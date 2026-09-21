import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-12B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    
    input_ids = torch.tensor([[2, 105, 2364, 107, 669, 5279, 529, 7001, 563, 106, 105, 4368, 107]])
    print(f"Input IDs ({input_ids.shape[1]} tokens): {input_ids[0].tolist()}")
    
    text_model = getattr(model, "language_model", None) or getattr(model.model, "language_model", None) or model.model
    if hasattr(text_model, "model"):
        text_model = text_model.model
    print(f"Text model: {text_model.__class__.__name__}")
    
    layer_outputs = {}
    
    def get_layer_hook(l_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                layer_outputs[f"layer_{l_idx}"] = output[0].detach()
            else:
                layer_outputs[f"layer_{l_idx}"] = output.detach()
        return hook
        
    sub_outputs = {}
    def get_sub_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                sub_outputs[name] = output[0].detach()
            else:
                sub_outputs[name] = output.detach()
        return hook
        
    hooks = []
    if hasattr(text_model, "embed_tokens"):
        hooks.append(text_model.embed_tokens.register_forward_hook(get_sub_hook("embed_tokens")))
    
    l0 = text_model.layers[0]
    for sub_name in ["input_layernorm", "self_attn", "post_attention_layernorm", "pre_feedforward_layernorm", "mlp", "post_feedforward_layernorm"]:
        if hasattr(l0, sub_name):
            hooks.append(getattr(l0, sub_name).register_forward_hook(get_sub_hook(f"layer0_{sub_name}")))

    for idx, layer in enumerate(text_model.layers):
        hooks.append(layer.register_forward_hook(get_layer_hook(idx)))
        
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]
        
    for h in hooks:
        h.remove()
        
    print("\n=== PyTorch 12B Sub-module Vector Norms (Last Token) ===")
    for sub_name, tensor in sub_outputs.items():
        vec = tensor[0, -1].numpy()
        norm = float(np.linalg.norm(vec))
        mean = float(np.mean(vec))
        print(f"  {sub_name:30s} | norm: {norm:10.4f} | mean: {mean:10.4f}")

    print("\n=== PyTorch 12B Intermediate Layer Vector Norms (Last Token) ===")
    for l_idx in [0, 1, 2, 5, 10, 20, 30, 40, 47]:
        name = f"layer_{l_idx}"
        if name in layer_outputs:
            vec = layer_outputs[name][0, -1].numpy()
            norm = float(np.linalg.norm(vec))
            mean = float(np.mean(vec))
            print(f"  {name:10s} | norm: {norm:10.4f} | mean: {mean:10.4f}")

    top_10 = torch.topk(logits, k=10)
    print("\n=== PyTorch 12B Top 10 Predicted Tokens ===")
    for idx, (tid, val) in enumerate(zip(top_10.indices, top_10.values)):
        t_str = tokenizer.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(val):8.4f} | text: {repr(t_str)}")

if __name__ == "__main__":
    main()
