import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-E2B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.eval()
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1]
        
    top_10 = torch.topk(logits, k=10)
    print("=== Top 10 PyTorch HuggingFace Predicted Tokens ===")
    for idx, (tid, val) in enumerate(zip(top_10.indices, top_10.values)):
        t_str = tokenizer.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(val):8.4f} | text: {repr(t_str)}")
        
    print("\nModel Config / Layer Architecture:")
    print(model.config)

if __name__ == "__main__":
    main()
