import shutil
import os
# CHANGE 1: Import ORTModelForMaskedLM instead of CustomTasks
from optimum.onnxruntime import ORTModelForMaskedLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

# This model has a vocabulary of ~105,000 tokens (Multilingual)
model_id = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
save_dir = "./onnx_output"

print(f"1. Exporting {model_id} to ONNX (MaskedLM task)...")

# CHANGE 2: Use ORTModelForMaskedLM to ensure we get Logits (Vocab Size), not Embeddings (768)
model = ORTModelForMaskedLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("2. Quantizing to Int8 (Dynamic AVX2)...")
quantizer = ORTQuantizer.from_pretrained(save_dir, file_name="model.onnx")

# Dynamic quantization is best for CPU inference
q_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

quantizer.quantize(
    save_dir=save_dir,
    quantization_config=q_config,
    # This will save the file as 'model_quantized.onnx' inside save_dir
)

print("3. Organizing files for Rust...")
# Move tokenizer
if os.path.exists(f"{save_dir}/tokenizer.json"):
    shutil.copy(f"{save_dir}/tokenizer.json", "tokenizer.json")

# Move quantized model
if os.path.exists(f"{save_dir}/model_quantized.onnx"):
    shutil.copy(f"{save_dir}/model_quantized.onnx", "model_quantized.onnx")
    print("Success! Created 'model_quantized.onnx' and 'tokenizer.json'")
else:
    print(f"Error: model_quantized.onnx not found in {save_dir}")