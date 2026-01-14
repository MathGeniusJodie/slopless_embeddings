use anyhow::Context;
use anyhow::Result;
use embed_anything::embeddings;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::params::LlamaPoolingType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::{AddBos, Special};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::CompressionRatio;
use qdrant_client::qdrant::CreateCollectionBuilder;
use qdrant_client::qdrant::HnswConfigDiff;
use qdrant_client::qdrant::OptimizersConfigDiff;
use qdrant_client::qdrant::ProductQuantizationBuilder;
use qdrant_client::qdrant::SearchParams;
use qdrant_client::qdrant::SearchPointsBuilder;
use qdrant_client::qdrant::UpsertPointsBuilder;
use qdrant_client::qdrant::quantization_config::Quantization;
use std::error::Error;
use std::io::Write;
use std::num::NonZero;

use qdrant_client::prelude::*;
use qdrant_client::qdrant::{
    Distance, QuantizationConfig, QuantizationType, VectorParams, VectorsConfig,
    vectors_config::Config,
};
use std::collections::HashMap;
use tokio;

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
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = embedding.to_vec();

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(
        &backend,
        "/home/jodie/opensearch_rs/mdbr-leaf-ir-q8_0.gguf",
        &model_params,
    )?;
    //let model = LlamaModel::load_from_file(&backend, "/home/jodie/opensearch_rs/granite-embedding-english-r2.Q6_K.gguf", &model_params)?;
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true)
        .with_n_seq_max(32)
        .with_n_threads(std::thread::available_parallelism()?.get().try_into()?)
        .with_n_ctx(Some(NonZero::new(512 * 32).unwrap()));
    //.with_pooling_type(LlamaPoolingType::Mean);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // Split the prompt to display the batching functionality
    let prompt = "The quick brown fox jumps over the lazy dog.\n\
I am Jodie.\n\
I am a stegosaurus.\n\
At first glance humans aren't very equipped for survival.\n\
We have no claws to defend ourselves, no fur to protect us from the cold and the sun.\n\
But what we have is one unique trait that allowed us to thrive: imagining the future.\n\
Through predictive modelling, long term planning, and a fair bit of wishful thinking, we didn't just hope a better world was possible,\n\
we were able to execute on our plans and mould the world into the shape of our fantasies.\n\
We imagined a world where a mother didn't have to watch her child die of smallpox... and then we made that world real.\n\
Our ancestors looked at the sky and imagined touching it. And their grandchildren left footprints on the moon.\n\
Most of our problems are a failure of imagination.\n\
If we can only imagine dystopia, how can we plan for utopia?\n\
We have a moral duty to use our powers to imagine futures that should exist.\n\
Tonight, on this darkest day, we do what our ancestors did: we gather, we speculate, and we steal warmth from the future.\n\
May the worlds we conjure tonight be the ones we build tomorrow. \n\
Your dad is Jerry.\n\
I am dad.\n\
I am father\n\
Your mom i jane\n\
Your mother is dad\n\
My father was a carpenter\n\
My father is John Smith\n\
To be, or not to be, that is the question.\n\
When in the course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another.\n\
Represent this sentence for searching relevant passages: Who is my dad?";
    let prompt_lines = prompt.lines();

    let time = std::time::Instant::now();
    // tokenize the prompt
    let tokens_lines_list = prompt_lines
        .clone()
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
            tokens_lines_list
                .iter()
                .map(|tok| tok.len())
                .max()
                .unwrap_or(0),
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
        )?;
    }
    eprintln!("Embedding computation took {:?}", time.elapsed());

    // try batch all together
    let time = std::time::Instant::now();
    let mut output = Vec::with_capacity(tokens_lines_list.len());
    let mut batch = LlamaBatch::new(n_ctx, tokens_lines_list.len() as i32);
    for (i, tokens) in tokens_lines_list.iter().enumerate() {
        batch.add_sequence(tokens, i as i32, false)?;
    }
    batch_decode(
        &mut ctx,
        &mut batch,
        tokens_lines_list.len() as i32,
        &mut output,
    )?;
    eprintln!("Batch embedding computation took {:?}", time.elapsed());

    for emb in output.iter_mut() {
        emb.truncate(256);
    }

    // Connect to Qdrant (assumes local docker: http://localhost:6333)
    let client = Qdrant::from_url("http://localhost:6334")
        .compression(None)
        .build()?;

    let collection_name = "pq_c_256d";

    let pq_config = ProductQuantizationBuilder::new(CompressionRatio::X32.into()).always_ram(true);

    // Vectors config
    let vectors = VectorsConfig {
        config: Some(Config::Params(VectorParams {
            size: 256,
            distance: Distance::Cosine.into(),
            hnsw_config: Some(HnswConfigDiff {
                m: Some(8), // smaller M for lower memory usage
                ef_construct: Some(200), // higher ef_construct for better indexing quality
                full_scan_threshold: Some(1),
                on_disk: Some(true),
                ..Default::default()
            }),
            quantization_config: Some(QuantizationConfig {
                quantization: Some(Quantization::Product(pq_config.clone().build())),
            }),
            on_disk: Some(true), // store vectors on disk (memmap)
            ..Default::default()
        })),
    };

    client.delete_collection(collection_name).await.ok();

    let response = client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(vectors)
                .quantization_config(pq_config)
                .optimizers_config(OptimizersConfigDiff {
                    indexing_threshold: Some(1),
                    flush_interval_sec: Some(1),
                    ..Default::default()
                }),
        )
        .await?;



    println!("Collection '{}' with PQ-C created!", collection_name);
    client
        .upsert_points(UpsertPointsBuilder::new(
            collection_name,
            output
                .iter()
                .enumerate()
                .map(|(i, emb)| PointStruct::new(i as u64, normalize(emb), Payload::default()))
                .collect::<Vec<_>>(),
        ))
        .await?;

    let test_search = client
    .search_points(
        SearchPointsBuilder::new(collection_name, output.last().unwrap().clone(), 5).params(SearchParams {
            hnsw_ef: Some(200),
            ..Default::default()
        })
    )
    .await?;

    test_search.result.iter().for_each(|res| {
        println!("Found point ID: {:?} with score: {}", prompt_lines.clone().nth(match res.id.clone().unwrap().point_id_options.unwrap(){
            qdrant_client::qdrant::point_id::PointIdOptions::Num(id) => id as usize,
            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id) => id.parse().unwrap(),
        }), res.score);
    });

    let info = client.collection_info(collection_name).await?.result.unwrap();
    println!("Collection info: {:?}", info);

    // 4 bit quantization simulation
    let output = output
        .iter()
        .map(|emb| {
            let mut max = 0.0;
            for &f in &emb[..256] {
                if f.abs() > max {
                    max = f.abs();
                }
            }
            if max == 0.0 {
                max = 1.0;
            }
            const RCP_MAX_I4_F32: f32 = 1.0 / 7.5; // 1.5 for 2 bit quantization is not terrible
            let norm = max * RCP_MAX_I4_F32;
            let quantised = emb[..256]
                .iter()
                .map(|&f| f / norm + 8.0)
                .map(|f| f.round() as u8)
                .collect::<Vec<_>>();

            let quantized_i4: Vec<u8> = quantised
                .chunks(2)
                .map(|chunk| (chunk[0] << 4) | chunk[1])
                .collect();

            let length = quantised
                .iter()
                .map(|&x| (x as f32 - 8.0).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm = 1.0 / length;
            (quantized_i4, norm)
        })
        .collect::<Vec<_>>();

    Ok(())
}