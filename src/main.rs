use anyhow::Context;
use anyhow::Result;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::params::LlamaPoolingType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use tokio::time;
use std::error::Error;
use std::io::Write;
use std::num::NonZero;

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, "/home/jodie/opensearch_rs/nomic-embed-text-v2-moe.Q4_K_M.gguf", &model_params)?;
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true)
        .with_n_seq_max(16)
        .with_n_threads(std::thread::available_parallelism()?.get().try_into()?)
        .with_n_ctx(Some(NonZero::new(512*16).unwrap()));
        //.with_pooling_type(LlamaPoolingType::Mean);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // Split the prompt to display the batching functionality
    let prompt = "search_document: The quick brown fox jumps over the lazy dog.\n\
search_document: I am Jodie.\n\
search_document: I am a stegosaurus.\n\
search_document: I am dad.\n\
search_document: Lorem ipsum\n\
search_document: My father was a carpenter\n\
search_document: I am Groot.\n\
search_document: To be, or not to be, that is the question.\n\
search_document: When in the course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another.\n\
search_document: All that glitters is not gold.";
    let prompt_lines = prompt.lines();

    let time = std::time::Instant::now();
    // tokenize the prompt
    let tokens_lines_list = prompt_lines
        .map(|line| model.str_to_token(line, AddBos::Always))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to tokenize {prompt}"))?;
    eprintln!("Tokenization took {:?}", time.elapsed());

    let n_ctx = ctx.n_ctx() as usize;
    let n_ctx_train = model.n_ctx_train();

    eprintln!("n_ctx = {n_ctx}, n_ctx_train = {n_ctx_train}");

    if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
        println!(
            "Warning: prompt length exceeds model context size ({} > {})",
            tokens_lines_list.iter().map(|tok| tok.len()).max().unwrap_or(0),
            n_ctx
        );
        return Ok(());
    }

    // print the prompt token-by-token
    eprintln!();

    for (i, token_line) in tokens_lines_list.iter().enumerate() {
        eprintln!("Prompt {i}");
        for token in token_line {
            // Attempt to convert token to string and print it; if it fails, print the token instead
            match model.token_to_str(*token, Special::Tokenize) {
                Ok(token_str) => eprintln!("{token} --> {token_str}"),
                Err(e) => {
                    eprintln!("Failed to convert token to string, error: {e}");
                    eprintln!("Token value: {token}");
                }
            }
        }
        eprintln!();
    }

    std::io::stderr().flush()?;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    let time = std::time::Instant::now();
    for tokens in &tokens_lines_list {
        // Create a fresh batch for each sequence
        let mut batch = LlamaBatch::new(n_ctx, 1);
        batch.add_sequence(tokens, 0, false)?;
        batch_decode(
            &mut ctx,
            &mut batch,
            1, // Only one sequence in this batch
            &mut output,
            true,
        )?;
    }
    eprintln!("Embedding computation took {:?}", time.elapsed());

    // try batch all together
    
    let time = std::time::Instant::now();
    let mut output = Vec::with_capacity(tokens_lines_list.len());
    let mut batch = LlamaBatch::new(n_ctx, tokens_lines_list.len() as i32);
    for (i,tokens) in tokens_lines_list.iter().enumerate() {
        batch.add_sequence(tokens, i as i32, false)?;
    }
    batch_decode(
        &mut ctx,
        &mut batch,
        tokens_lines_list.len() as i32,
        &mut output,
        true,
    )?;
    eprintln!("Batch embedding computation took {:?}", time.elapsed());


    for (i, embeddings) in output.iter().enumerate() {
        eprintln!("Embeddings {i}: {:?}", &embeddings.iter().take(256).map(|f| (f*127.0).round() as i8).collect::<Vec<_>>());
        eprintln!();
    }
    Ok(())
}
