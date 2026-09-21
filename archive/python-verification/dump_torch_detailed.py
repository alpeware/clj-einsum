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
    
    print("\nModel structure:")
    for name, child in model.named_children():
        print(f"  child: {name}")
        for n2, c2 in child.named_children():
            print(f"    sub-child: {n2}")
            
    # Find text layers
    text_model = getattr(model.model, "language_model", None) or getattr(model.model, "text_decoder", None) or model.model
    print(f"\nText model: {text_model.__class__.__name__}")
    
    layer0 = text_model.layers[0]
    submodule_outputs = {}
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                submodule_outputs[name] = output[0].detach()
            else:
                submodule_outputs[name] = output.detach()
        return hook
        
    hooks = []
    hooks.append(layer0.input_layernorm.register_forward_hook(get_hook("l0_input_layernorm")))
    hooks.append(layer0.self_attn.register_forward_hook(get_hook("l0_self_attn")))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(get_hook("l0_post_attention_layernorm")))
    hooks.append(layer0.pre_feedforward_layernorm.register_forward_hook(get_hook("l0_pre_ffw_layernorm")))
    hooks.append(layer0.mlp.register_forward_hook(get_hook("l0_mlp")))
    hooks.append(layer0.post_feedforward_layernorm.register_forward_hook(get_hook("l0_post_ffw_layernorm")))
    hooks.append(layer0.register_forward_hook(get_hook("l0_output")))
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        
    for h in hooks:
        h.remove()
        
    os.makedirs("torch_detailed", exist_ok=True)
    print("\n=== PyTorch Layer 0 Internal Vectors (Last Token) ===")
    for k, v in submodule_outputs.items():
        vec = v[0, -1].numpy()
        norm = float(np.linalg.norm(vec))
        print(f"  {k:30s} | norm: {norm:10.4f} | mean: {float(np.mean(vec)):10.4f} | min: {float(np.min(vec)):10.4f} | max: {float(np.max(vec)):10.4f}")
        np.save(f"torch_detailed/{k}.npy", vec)

if __name__ == "__main__":
    main()
