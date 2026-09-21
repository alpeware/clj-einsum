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
        
    hidden_states = outputs.hidden_states # tuple of (embed_out, layer0_out, layer1_out, ..., layer34_out)
    
    os.makedirs("torch_dumps", exist_ok=True)
    for i, h in enumerate(hidden_states):
        name = "embed" if i == 0 else f"layer_{i-1}"
        vec = h[0, -1].numpy()
        norm = float(np.linalg.norm(vec))
        print(f"State {name:10s} | last token vector norm: {norm:10.4f}")
        np.save(f"torch_dumps/{name}.npy", vec)
        
    print("\nSaved all layer last-token vectors to torch_dumps/")

if __name__ == "__main__":
    main()
