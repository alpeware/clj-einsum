import os
import sys
import kagglehub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("Downloading Gemma 4 E2B from Kaggle via kagglehub...")
    path = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e2b")
    print(f"Downloaded model to path: {path}")
    
    print("\nFiles in Kaggle model path:")
    for f in os.listdir(path):
        print(f"  {f}")
        
    print("\nLoading tokenizer and model with HuggingFace transformers...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cpu")
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"\nPrompt: '{prompt}'")
    print(f"Token IDs: {inputs.input_ids.tolist()}")
    
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated Output:\n{generated_text}")

if __name__ == "__main__":
    main()
