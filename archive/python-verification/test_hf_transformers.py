import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, "..", ".models", "gemma-4-E2B-it"))
    if not os.path.exists(model_dir):
        model_dir = os.path.abspath(os.path.join(script_dir, "..", ".models", "gemma-4-E2B"))
        
    print(f"Loading tokenizer and model from local path '{model_dir}' using HuggingFace Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    
    prompt_str = "<bos><|turn|>user\nWhat is the capital of France?<turn|>\n<|turn|>model\n"
    inputs = tokenizer(prompt_str, return_tensors="pt")
    print(f"\nPrompt: '{prompt_str}'")
    print(f"Input Token IDs: {inputs.input_ids.tolist()}")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\nHuggingFace Transformers Generated Output:\n{repr(generated_text)}")

if __name__ == "__main__":
    main()
