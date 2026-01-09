use anyhow::Result;
use ndarray::Array2;
use ort::{
    inputs, session::Session, session::builder::GraphOptimizationLevel,
    session::builder::SessionBuilder,
};
use tokenizers::Tokenizer;

pub struct NeuralSparseModel {
    session: Session,
    tokenizer: Tokenizer,
    // Pre-allocated buffer to avoid allocation every request
    vocab_buffer: Vec<f32>,
}

impl NeuralSparseModel {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        // Initialize ONNX Runtime Session
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(20)? // Adjust based on CPU cores
            .commit_from_file(model_path)?;

        for output in session.outputs() {
            println!("Output Name: {}", output.name());
            println!("Output Type: {:?}", output.dtype());
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        // Get vocab size directly from tokenizer to size our buffer
        let vocab_size = tokenizer.get_vocab_size(true);

        Ok(Self {
            session,
            tokenizer,
            vocab_buffer: vec![0.0; vocab_size],
        })
    }

    pub fn generate_sparse_vector(&mut self, text: &str) -> Result<Vec<(String, f32)>> {
        // --- 1. Tokenization ---
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let batch_size = 1;
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask)?;
        let token_type_ids_array = Array2::<i64>::zeros((batch_size, seq_len));

        // --- 2. Inference ---
        let inputs = inputs![
            "input_ids" => ort::value::Value::from_array(input_ids_array)?,
            "attention_mask" => ort::value::Value::from_array(attention_mask_array.clone())?,
            "token_type_ids" => ort::value::Value::from_array(token_type_ids_array)?
        ];

        let outputs = self.session.run(inputs)?;
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (dims, output_data) = output_tensor;
        // Dims: [Batch, Seq, Vocab]
        let vocab_size = dims[2] as usize;

        // Ensure buffer is clean and sized correctly
        if self.vocab_buffer.len() != vocab_size {
            self.vocab_buffer.resize(vocab_size, f32::NEG_INFINITY);
        } else {
            self.vocab_buffer.fill(f32::NEG_INFINITY);
        }

        // --- 3. Optimized Max-Pooling (The "Dense" Approach) ---
        // We iterate linearly through the data.
        // Data layout is [Seq 0 [Vocab...], Seq 1 [Vocab...]]

        let mut max_activation = 0.0f32;

        // Iterate over sequence steps
        for seq_idx in 0..seq_len {
            // Skip logic for padding tokens (based on attention mask)
            if attention_mask_array[[0, seq_idx]] == 0 {
                continue;
            }

            // Get the slice of logits for this specific token in the sequence
            let offset = seq_idx * vocab_size;
            let logits_slice = &output_data[offset..offset + vocab_size];

            // Hot loop: pure array indexing, extremely fast cache access
            for (vocab_idx, &logit) in logits_slice.iter().enumerate() {
                if logit > self.vocab_buffer[vocab_idx] {
                    self.vocab_buffer[vocab_idx] = logit;
                }
            }
        }

        // --- 4. Activation, Thresholding & Decoding ---
        let mut result_vector: Vec<(String, f32)> = Vec::new();

        // Compute actual weights and find global max for thresholding
        // We do this in one pass over the vocab
        for val in self.vocab_buffer.iter_mut() {
            // Apply ReLU: if *val < 0, make it 0
            if *val < 0.0 {
                *val = 0.0;
            }

            // Apply Log saturation: log(1 + val)
            if *val > 0.0 {
                *val = (1.0 + *val).ln();
                if *val > max_activation {
                    max_activation = *val;
                }
            }
        }

        let threshold = max_activation * 0.25; // Filter noise

        for (vocab_id, &weight) in self.vocab_buffer.iter().enumerate() {
            if weight >= threshold {
                // Only decode significant tokens
                let token = self
                    .tokenizer
                    .decode(&[vocab_id as u32], false)
                    .unwrap_or_default();
                result_vector.push((token, weight));
            }
        }

        // Sort by score
        result_vector.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // --- 5. Clean Subwords (Optional Polish) ---
        // If result has "run" and "##ning", this is where you'd handle it.
        // For SPLADE (Bag of Words), strict merging is hard because order is lost,
        // but we can clean the "##" visuals.
        let cleaned_vector: Vec<(String, f32)> = result_vector
            .into_iter()
            .map(|(tok, w)| (tok.replace("##", ""), w))
            .collect();

        Ok(cleaned_vector)
    }
}

fn main() -> Result<()> {
    let model_path = "/home/jodie/opensearch_rs/onnx_output/model_quantized.onnx";
    let tokenizer_path = "/home/jodie/opensearch_rs/onnx_output/tokenizer.json";

    let mut splade_model = NeuralSparseModel::new(model_path, tokenizer_path)?;
    // Warmup (optional)
    let _ = splade_model.generate_sparse_vector("warmup");

    let start = std::time::Instant::now();
    let sparse_features = splade_model.generate_sparse_vector("Rust is a systems programming language that runs blazingly fast.")?;
    let duration = start.elapsed();

    println!("Inference took: {:?}", duration);
    println!("\nTop Sparse Features:");
    for (token, weight) in sparse_features.iter(){
        println!("  {:<15} : {:.4}", token, weight);
    }

    println!("{:?}",splade_model.generate_sparse_vector(
        "To be or not to be, that is the question."
    )?.iter().collect::<Vec<_>>());

    Ok(())
}
