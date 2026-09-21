import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-E2B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1] # last token logits
        
    top_10 = torch.topk(logits, k=10)
    print("=== PyTorch HuggingFace Reference Output for 'The capital of France is' ===")
    for idx, (tid, val) in enumerate(zip(top_10.indices, top_10.values)):
        t_str = tokenizer.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(val):8.4f} | text: {repr(t_str)}")
        
    np.save("torch_logits.npy", logits.cpu().numpy())
    print("\nSaved PyTorch last token logits to torch_logits.npy")

if __name__ == "__main__":
    main()
