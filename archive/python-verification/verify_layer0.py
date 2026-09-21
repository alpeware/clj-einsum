import os
import sys
import jax
import jax.numpy as jnp
from safetensors import safe_open
from gemma import gm
from tokenizers import Tokenizer
from verify_inference import load_safetensors_flax_params

def main():
    model_dir = "../.models/gemma-4-E2B-it"
        
    print(f"Loading weights from {model_dir}...")
    variables = load_safetensors_flax_params(model_dir)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    encoded = tok.encode("What is the capital of France?")
    prompt_ids = [2, 105, 2364, 107] + encoded.ids + [106, 107, 105, 4368, 107]
    print(f"Prompt IDs ({len(prompt_ids)} tokens):", prompt_ids)
    
    tokens = jnp.array([prompt_ids], dtype=jnp.int32)
    positions = jnp.arange(len(prompt_ids), dtype=jnp.int32)[None, :]
    
    model = gm.nn.Gemma4_E2B()
    out = model.apply(variables, tokens, positions=positions)
    logits = out.logits[0, -1]
    
    top_10 = jnp.argsort(logits)[::-1][:10]
    print("\n=== Top 10 Reference IT Model Predicted Tokens ===")
    for tid in top_10:
        t_str = tok.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(logits[tid]):8.4f} | text: {repr(t_str)}")

if __name__ == "__main__":
    main()
