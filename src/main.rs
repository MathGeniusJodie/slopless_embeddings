use anyhow::Result;
use ndarray::{Array2, ArrayView3};
use ort::{inputs, session::builder::GraphOptimizationLevel, session::builder::SessionBuilder, session::Session};
use std::collections::HashMap;
use tokenizers::Tokenizer;

// --- 1. Struct to hold the "Setup" state ---
pub struct NeuralSparseModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl NeuralSparseModel {
    // --- Setup Logic: Loads Model and Tokenizer ---
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        println!("Loading model from {}...", model_path);

        // Initialize ONNX Runtime Session
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Load Tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self { session, tokenizer })
    }

    // --- Sentence to Tokens Logic: Runs inference and calculates weights ---
    pub fn generate_sparse_vector(&mut self, text: &str) -> Result<Vec<(String, f32)>> {
        // 1. Tokenization
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        
        let batch_size = 1;
        let seq_len = input_ids.len();

        // Create ndarray tensors (Batch, Seq)
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask.clone())?;
        let token_type_ids_array = Array2::<i64>::zeros((batch_size, seq_len));

        // 2. Run Inference
        let inputs = inputs![
            "input_ids" => ort::value::Value::from_array(input_ids_array)?,
            "attention_mask" => ort::value::Value::from_array(attention_mask_array.clone())?,
            "token_type_ids" => ort::value::Value::from_array(token_type_ids_array)?
        ];

        let outputs = self.session.run(inputs)?;
        
        // Extract output (logits)
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (output_shape, output_data) = output_tensor; // [Batch, Seq, Vocab_Size]
        let vocab_size = output_shape[2] as usize;

        let output_array = ArrayView3::<f32>::from_shape(
            (output_shape[0] as usize, output_shape[1] as usize, output_shape[2] as usize),
            output_data
        )?;

        // 3. SPLADE / Neural Sparse Logic
        // Logic: Weight = max(log(1 + relu(activation))) over the sequence dimension
        let batch_idx = 0;
        let mut sparse_vector: HashMap<usize, f32> = HashMap::new();

        // Iterate over Sequence dimension
        for seq_idx in 0..seq_len {
            // Skip padding tokens
            if attention_mask_array[[batch_idx, seq_idx]] == 0 {
                continue;
            }

            for vocab_idx in 0..vocab_size {
                let logit = output_array[[batch_idx, seq_idx, vocab_idx]];
                
                // ReLU
                let relu_out = if logit > 0.0 { logit } else { 0.0 };
                
                if relu_out > 0.0 {
                    // log(1 + x)
                    let weight = (1.0f32 + relu_out).ln();

                    // Max Pooling
                    let entry = sparse_vector.entry(vocab_idx).or_insert(0.0);
                    if weight > *entry {
                        *entry = weight;
                    }
                }
            }
        }

        // 4. Pruning
        if sparse_vector.is_empty() {
            return Ok(Vec::new());
        }

        let max_weight = sparse_vector.values().fold(0.0f32, |a, &b| a.max(b));
        let threshold = max_weight * 0.1;

        // Decode and filter
        let mut result_vector: Vec<(String, f32)> = Vec::new();

        for (vocab_id, weight) in sparse_vector {
            if weight >= threshold {
                let token = self.tokenizer.decode(&[vocab_id as u32], false).unwrap_or_else(|_| "???".to_string());
                result_vector.push((token, weight));
            }
        }

        // Sort by weight descending
        result_vector.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(result_vector)
    }
}

// --- Main execution ---
fn main() -> Result<()> {
    // Paths
    let model_path = "/home/jodie/opensearch_rs/onnx_output/model_quantized.onnx";
    let tokenizer_path = "/home/jodie/opensearch_rs/onnx_output/tokenizer.json";

    // 1. Initialization Phase
    let mut splade_model = NeuralSparseModel::new(model_path, tokenizer_path)?;

    // 2. Input
    let text = "Rust is a systems programming language that runs blazingly fast.";
    println!("Input: \"{}\"", text);

    // 3. Inference Phase
    let sparse_features = splade_model.generate_sparse_vector(text)?;

    // 4. Display Results
    println!("\nTop Sparse Features:");
    for (token, weight) in sparse_features.iter() {
        println!("  {:<15} : {:.4}", token, weight);
    }

    Ok(())
}