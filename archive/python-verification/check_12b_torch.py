import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-12B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # We can pass raw text or exact token IDs
    prompt = "The capital of France is"
    # Format with Gemma 4 IT prompt structure: <bos><|turn>user\nThe capital of France is<turn|><|turn>model\n
    formatted_prompt = f"<bos><|turn>user\n{prompt}<turn|><|turn>model\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    
    print("Formatted prompt:", repr(formatted_prompt))
    print("Token IDs:", inputs.input_ids[0].tolist())
    
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1] # last token logits
        
    top_10 = torch.topk(logits, k=10)
    print("\n=== PyTorch 12B Top 10 Predicted Tokens ===")
    for idx, (tid, val) in enumerate(zip(top_10.indices, top_10.values)):
        t_str = tokenizer.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(val):8.4f} | text: {repr(t_str)}")
        
    # Also try without IT chat template tags
    raw_inputs = tokenizer(prompt, return_tensors="pt")
    print("\nRaw Prompt Token IDs:", raw_inputs.input_ids[0].tolist())
    with torch.no_grad():
        raw_out = model(**raw_inputs)
        raw_logits = raw_out.logits[0, -1]
    raw_top_10 = torch.topk(raw_logits, k=10)
    print("\n=== PyTorch 12B Raw Prompt Top 10 Predicted Tokens ===")
    for idx, (tid, val) in enumerate(zip(raw_top_10.indices, raw_top_10.values)):
        t_str = tokenizer.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(val):8.4f} | text: {repr(t_str)}")

if __name__ == "__main__":
    main()
