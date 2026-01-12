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
use ort::session::output;
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
        emb[..256].iter().map(|&f| f / max).collect::<Vec<_>>()
    }).collect::<Vec<_>>();
    let output  = output.iter().cloned().map(|emb| {
        let emb = emb.iter().map(|&f| (f*7.0).round() as i8).collect::<Vec<i8>>();
        emb
    }).collect::<Vec<Vec<i8>>>();

    for (i, embeddings) in output.iter().enumerate() {
        eprintln!("Embeddings {i}: {:?} Length: {}", &embeddings, embeddings.len());
        eprintln!();
    }
/*
    let output  = output.iter().cloned().map(|emb| {
        let emb = emb.iter().map(|&f| f as f32).collect::<Vec<_>>();
        normalize(&emb)
    }).collect::<Vec<_>>();
*/
    for (i, embeddings) in output.iter().enumerate() {
        // calc distance to last embedding
        let dist = cosine_distance_groundtruth(embeddings, output.last().unwrap());
        if dist<0.1 {
            continue;
        }
        let text = prompt_lines.clone().nth(i).unwrap_or("");
        eprintln!("Distance of Embeddings {text} to last: {}", dist);
    }
    let i4_table: [i8;16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];
    let mut mul_table:Vec<i8> = Vec::with_capacity(256);
    for i in 0..16 {
        for j in 0..16 {
            mul_table.push(i4_table[i] * i4_table[j]);
        }
    }
    // print the mul_table as binary digits
    for i in 0..16 {
        for j in 0..16 {
            let val = mul_table[i*16 + j];
            print!("{:3},", val);
        }
        println!();
    }
    Ok(())
}

fn cosine_distance_groundtruth(a: &[i8], b: &[i8]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&val_a, &val_b) in a.iter().zip(b.iter()) {
        dot_product += (val_a as f32) * (val_b as f32);
        norm_a += (val_a as f32).powi(2);
        norm_b += (val_b as f32).powi(2);
    }

    let norm_a = 1.0 / norm_a.sqrt();
    let norm_b = 1.0 / norm_b.sqrt();

    dot_product * norm_a * norm_b
}

fn cosine_distance_i4(a: &[i8], b: &[i8]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&val_a, &val_b) in a.iter().zip(b.iter()) {
        dot_product += (val_a as f32) * (val_b as f32);
        norm_a += (val_a as f32).powi(2);
        norm_b += (val_b as f32).powi(2);
    }
    // we should precompute these
    let norm_a = 1.0 / norm_a.sqrt();
    let norm_b = 1.0 / norm_b.sqrt();

    let i4_table: [i8;16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];
    let mut mul_table:Vec<i8> = Vec::with_capacity(256);
    for i in 0..16 {
        for j in 0..16 {
            mul_table.push(i4_table[i] * i4_table[j]);
        }
    }
    1.0
}

use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn dot_i4_optimized(a: __m256i, b: __m256i) -> __m256i {
    // 1. Precompute constants (Hoist these out if calling in a loop)
    // 0x88 (-120) adds 8 to each nibble: [-8, 7] -> [0, 15]
    let xor_bias = _mm256_set1_epi8(0x88_u8 as i8); 
    let mask_low = _mm256_set1_epi8(0x0F);
    let eights = _mm256_set1_epi8(8);
    
    // 2. Bias inputs to convert Signed-4bit to Unsigned-4bit (0..15)
    // We assume inputs are packed 4-bit (two per byte).
    let a_u = _mm256_xor_si256(a, xor_bias);
    let b_u = _mm256_xor_si256(b, xor_bias);

    // 3. Unpack Low/High nibbles
    // AND is port 0/1/5, Shifts are port 0/1. 
    // We interleave instructions to allow better pipelining.
    let a_lo = _mm256_and_si256(a_u, mask_low);
    let b_lo = _mm256_and_si256(b_u, mask_low);
    
    // Note: srli_epi16 operates on 16-bit lanes. 
    // It shifts [H1 L1 H0 L0] -> [0 H1 L1 H0]. 
    // We mask immediately to isolate the nibbles.
    let a_hi = _mm256_and_si256(_mm256_srli_epi16(a_u, 4), mask_low);
    let b_hi = _mm256_and_si256(_mm256_srli_epi16(b_u, 4), mask_low);

    // 4. Main Term: (a+8)*(b+8)
    // maddubs performs: (u8 * i8) + (u8 * i8) -> i16 saturated
    // Since our inputs are 0..15, they fit in both u8 and i8 ranges.
    // Result is Sum( (a+8)(b+8) )
    let term_lo = _mm256_maddubs_epi16(a_lo, b_lo);
    let term_hi = _mm256_maddubs_epi16(a_hi, b_hi);
    
    // 5. Correction Term: 8 * Sum(a+8 + b+8)
    // Optimization: Calculate sum of nibbles first.
    // Max sum per byte = 15+15+15+15 = 60 (Fits safely in i8)
    let sum_a = _mm256_add_epi8(a_lo, a_hi);
    let sum_b = _mm256_add_epi8(b_lo, b_hi);
    let sum_biased = _mm256_add_epi8(sum_a, sum_b);
    
    // We use maddubs by 1 to effectively widen i8 sum to i16 without separate unpacks,
    // then shift left by 3 (multiply by 8).
    // Or simply maddubs by 8 directly.
    let correction = _mm256_maddubs_epi16(sum_biased, eights);

    // 6. Final Assembly
    // Algebra: 
    // Term = Sum( (a+8)(b+8) ) = Sum( ab + 8a + 8b + 64 )
    // Corr = Sum( 8(a+8 + b+8) ) = Sum( 8a + 8b + 128 )
    // Term - Corr = Sum( ab - 64 )
    // Since maddubs sums PAIRS of products, we effectively subtracted 64*2 = 128 per lane.
    // We must add 128 back.
    
    let term_total = _mm256_add_epi16(term_lo, term_hi);
    let diff = _mm256_sub_epi16(term_total, correction);
    
    // Combine 128 offset addition.
    _mm256_add_epi16(diff, _mm256_set1_epi16(128))
}


use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn dot_i4_optimized_v2(a: __m256i, b: __m256i) -> __m256i {
    // 1. Constants
    // mask: Selects lower nibbles (0x0F)
    let mask = _mm256_set1_epi8(0x0F);
    
    // xor_bias: 0x88 (0x80 | 0x08)
    // - Toggles MSB (0x80) for sign extension prep.
    // - Toggles nibble MSB (0x08) for offset prep.
    let xor_bias = _mm256_set1_epi8(-120); // 0x88 as i8
    
    // sub_bias: 8
    // We can derive this to save a register load if register pressure is high in the calling loop.
    // In isolation, the compiler handles this, but explicit reuse ensures we don't waste a register.
    let sub_bias = _mm256_and_si256(xor_bias, mask); // Results in 0x08

    // 2. Pre-process inputs (The "0x88" Trick)
    // This flips the sign bit and the 4-bit MSB simultaneously for both nibbles.
    let a_prep = _mm256_xor_si256(a, xor_bias);
    let b_prep = _mm256_xor_si256(b, xor_bias);

    // 3. Low Nibbles
    // A: Sign-extend 4-bit to 8-bit. 
    // Logic: ((a_nib ^ 8) - 8) effectively extends the sign.
    let a_lo = _mm256_sub_epi8(_mm256_and_si256(a_prep, mask), sub_bias);
    
    // B: Convert to unsigned 0..15 range (biased by +8).
    // Logic: (b_nib ^ 8). We effectively compute (B_real + 8).
    let b_lo = _mm256_and_si256(b_prep, mask);

    // 4. High Nibbles (Interleaved to allow pipelining)
    // Shift right by 4 bits. Note: srli_epi16 shifts 16-bit words, but since we mask 
    // immediately with 0x0F, cross-byte boundary bits are cleaned up correctly.
    let a_hi_shifted = _mm256_srli_epi16(a_prep, 4);
    let b_hi_shifted = _mm256_srli_epi16(b_prep, 4);

    let a_hi = _mm256_sub_epi8(_mm256_and_si256(a_hi_shifted, mask), sub_bias);
    let b_hi = _mm256_and_si256(b_hi_shifted, mask);

    // 5. Compute Dot Product and Correction
    // We are computing: sum( A_real * (B_real + 8) )
    // Expansion:        sum( A_real * B_real ) + 8 * sum( A_real )
    
    // Main MAC (Multiply Accumulate)
    // maddubs: Multiplies unsigned u8 (B) with signed i8 (A) and adds adjacent pairs.
    let dot_lo = _mm256_maddubs_epi16(b_lo, a_lo);
    let dot_hi = _mm256_maddubs_epi16(b_hi, a_hi);
    let dot_combined = _mm256_add_epi16(dot_lo, dot_hi);

    // Correction Term: 8 * sum( A_real )
    // Optimization: Add low/high A first, then multiply by 8 once.
    let a_sum = _mm256_add_epi8(a_lo, a_hi);
    let correction = _mm256_maddubs_epi16(sub_bias, a_sum); // bias(8) * sum(A)

    // 6. Final Result
    // sum(A*B) = (sum(A*B) + 8*sum(A)) - 8*sum(A)
    _mm256_sub_epi16(dot_combined, correction)
}