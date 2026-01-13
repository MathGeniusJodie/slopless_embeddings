use anyhow::Context;
use anyhow::Result;
use embed_anything::embeddings;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::params::LlamaPoolingType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
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
    output: &mut Vec<Vec<f32>>
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
fn main() -> Result<(), Box<dyn Error>> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, "/home/jodie/opensearch_rs/mdbr-leaf-ir-q8_0.gguf", &model_params)?;
    //let model = LlamaModel::load_from_file(&backend, "/home/jodie/opensearch_rs/granite-embedding-english-r2.Q6_K.gguf", &model_params)?;
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true)
        .with_n_seq_max(32)
        .with_n_threads(std::thread::available_parallelism()?.get().try_into()?)
        .with_n_ctx(Some(NonZero::new(512*32).unwrap()));
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
My father was a carpenter\n\
My father is John Smith\n\
To be, or not to be, that is the question.\n\
When in the course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another.\n\
Represent this sentence for searching relevant passages: Who is my dad?";
    let prompt_lines = prompt.lines();

    let time = std::time::Instant::now();
    // tokenize the prompt
    let tokens_lines_list = prompt_lines.clone()
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
    )?;
    eprintln!("Batch embedding computation took {:?}", time.elapsed());

    // 4 bit quantization simulation
    let output  = output.iter().map(|emb| {
        let mut max = 0.0;
        for &f in &emb[..256] {
            if f.abs() > max {
                max = f.abs();
            }
        };
        if max == 0.0 {
            max = 1.0;
        }
        const RCP_MAX_I4_F32: f32 = 1.0/7.5; // 1.5 for 2 bit quantization is not terrible
        let norm = max*RCP_MAX_I4_F32;
        let quantised = emb[..256]
            .iter()
            .map(|&f| f / norm + 8.0)
            .map(|f| f.round() as u8)
            .collect::<Vec<_>>();

        let quantized_i4: Vec<u8> = quantised.chunks(2).map(|chunk| {
            (chunk[0] << 4) | chunk[1]
        }).collect();

        let length = quantised.iter().map(|&x| (x as f32 -8.0).powi(2)).sum::<f32>().sqrt();
        let norm = 1.0 / length;
        (quantized_i4,norm)
    }).collect::<Vec<_>>();

    for (i, (embeddings,_)) in output.iter().enumerate() {
        eprintln!("Embeddings {i}: {:?} Length: {}", &embeddings, embeddings.len());
        eprintln!();
    }


    for (i, embeddings) in output.iter().enumerate() {
        // calc distance to last embedding
        let mut a_combined = embeddings.0.clone();
        a_combined.extend_from_slice(embeddings.1.to_le_bytes().as_slice());
        let mut b_combined = output.last().unwrap().0.clone();
        b_combined.extend_from_slice(output.last().unwrap().1.to_le_bytes().as_slice());
        let dist = cosine_distance_groundtruth(&a_combined[..], &b_combined[..]);
        let text = prompt_lines.clone().nth(i).unwrap_or("");
        if dist<0.1 {
            continue;
        }
        eprintln!("Distance of Embeddings {text} to last: {}", dist);
        println!("Length of combined embedding: {}", a_combined.len());
    }

    Ok(())
}

fn cosine_distance_groundtruth(a_i4: &[u8], b_i4: &[u8]) -> f32 {
    let norm_a = f32::from_le_bytes(a_i4[128..132].try_into().unwrap());
    let norm_b = f32::from_le_bytes(b_i4[128..132].try_into().unwrap());
    let dot_product = unsafe {
        dot_i4_256_nibbles_unrolled(
            a_i4.as_ptr(),
            b_i4.as_ptr(),
        ) as f32
    };
    dot_product * norm_a * norm_b
}

use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn dot_i4_256_nibbles_unrolled(a_ptr: *const u8, b_ptr: *const u8) -> i32 {
    // 1. Constants
    let mask = _mm256_set1_epi8(0x0F);
    let sub_bias = _mm256_set1_epi8(0x08);

    // 2. Accumulators
    // acc_dot stores the main product sums (16 x i16)
    let mut acc_dot = _mm256_setzero_si256();
    // acc_a_sum stores the sum of 'A' weights for correction (32 x i8)
    // We can use i8 because 4 iterations of range [-16, 14] will not overflow i8.
    let mut acc_a_sum = _mm256_setzero_si256();

    // 3. Fully Unrolled Execution (4 steps of 32 bytes)
    // Using a macro here just to keep the code DRY, but it effectively inlines 4 times.
    for i in 0..4 {
        let a = _mm256_loadu_si256(a_ptr.add(i * 32) as *const __m256i);
        let b = _mm256_loadu_si256(b_ptr.add(i * 32) as *const __m256i);

        //turn i4 to u8
        //let xor_bias = _mm256_set1_epi8(-120); // 0x88
        //let a = _mm256_xor_si256(a, xor_bias);
        //let b = _mm256_xor_si256(b, xor_bias);

        // Split Nibbles
        // A_lo: ((a & mask) - 8)
        let a_lo = _mm256_sub_epi8(_mm256_and_si256(a, mask), sub_bias);
        // B_lo: (b & mask) -> represents (B_real + 8)
        let b_lo = _mm256_and_si256(b, mask);
        
        // High nibbles
        let a_hi_shifted = _mm256_srli_epi16(a, 4);
        let b_hi_shifted = _mm256_srli_epi16(b, 4);
        let a_hi = _mm256_sub_epi8(_mm256_and_si256(a_hi_shifted, mask), sub_bias);
        let b_hi = _mm256_and_si256(b_hi_shifted, mask);

        // Main Dot Product (expensive part)
        let dot_lo = _mm256_maddubs_epi16(b_lo, a_lo);
        let dot_hi = _mm256_maddubs_epi16(b_hi, a_hi);
        
        // Accumulate main dot product
        acc_dot = _mm256_add_epi16(acc_dot, _mm256_add_epi16(dot_lo, dot_hi));

        // Lazy Correction Accumulation (Cheap part)
        // Just add the raw i8 A-values. We multiply by 8 later.
        // Note: We combine lo/hi immediately to save an add operation on the accumulator
        let sum_a_iter = _mm256_add_epi8(a_lo, a_hi);
        acc_a_sum = _mm256_add_epi8(acc_a_sum, sum_a_iter);
    }

    // 4. Finalize Correction
    // Now we do the expensive multiplication just once for the whole block.
    // Correction = 8 * sum(A)
    let correction = _mm256_maddubs_epi16(sub_bias, acc_a_sum);
    
    // Apply correction: result = dot - correction
    let final_vec = _mm256_sub_epi16(acc_dot, correction);

    // 5. Horizontal Sum (Same as before)
    let ones = _mm256_set1_epi16(1);
    let vec_i32 = _mm256_madd_epi16(final_vec, ones);
    
    let hi_128 = _mm256_extracti128_si256(vec_i32, 1);
    let lo_128 = _mm256_castsi256_si128(vec_i32);
    let sum_128 = _mm_add_epi32(lo_128, hi_128);
    let sum_64 = _mm_hadd_epi32(sum_128, sum_128);
    let sum_32 = _mm_hadd_epi32(sum_64, sum_64);
    
    _mm_cvtsi128_si32(sum_32)
}