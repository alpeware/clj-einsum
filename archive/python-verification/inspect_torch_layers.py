import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-E2B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("=== PyTorch Gemma 4 Model Submodules ===")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0: # Leaf module
            print(f"  {name:60s} | {module}")

if __name__ == "__main__":
    main()
