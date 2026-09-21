import inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-E2B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    
    text_model = model.model.language_model
    print("=== KV Sharing status across all 35 layers ===")
    for i, layer in enumerate(text_model.layers):
        attn = layer.self_attn
        is_shared = getattr(attn, "is_kv_shared_layer", False)
        layer_type = getattr(attn, "layer_type", "unknown")
        print(f"Layer {i:2d} | type: {layer_type:20s} | is_kv_shared_layer: {is_shared}")

if __name__ == "__main__":
    main()
