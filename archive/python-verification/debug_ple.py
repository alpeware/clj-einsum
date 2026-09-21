import os
import sys
import jax
import jax.numpy as jnp
from safetensors import safe_open
from tokenizers import Tokenizer
from gemma import gm

def main():
    model_dir = "../.models/gemma-4-E2B-it"
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    encoded = tok.encode("What is the capital of France?")
    prompt_ids = [2, 105, 2364, 107] + encoded.ids + [106, 107, 105, 4368, 107]
    tokens = jnp.array([prompt_ids], dtype=jnp.int32)
    
    sf_path = os.path.join(model_dir, "model.safetensors")
    sf_tensors = {}
    with safe_open(sf_path, framework="np") as f:
        for k in f.keys():
            sf_tensors[k] = f.get_tensor(k)
            
    emb_table = sf_tensors["model.language_model.embed_tokens.weight"]
    emb_pl = sf_tensors["model.language_model.embed_tokens_per_layer.weight"] # (262144, 8960)
    pl_model_proj = sf_tensors["model.language_model.per_layer_model_projection.weight"] # (8960, 1536)
    pl_proj_norm_scale = sf_tensors["model.language_model.per_layer_projection_norm.weight"] # (256,)
    
    # 1. tok_embed (with sqrt(1536))
    tok_embed = emb_table[tokens] * jnp.sqrt(1536.0) # (1, 16, 1536)
    print("Python tok_embed norm:", float(jnp.linalg.norm(tok_embed)))
    
    # 2. PLE
    raw_pl_tok = emb_pl[tokens] # (1, 16, 8960)
    pl_tok_scaled = raw_pl_tok * jnp.sqrt(256.0) # (1, 16, 8960)
    pl_context_raw = jnp.dot(tok_embed, pl_model_proj.T) # (1, 16, 8960)
    
    pl_tok_4d = pl_tok_scaled.reshape(1, 16, 35, 256)
    pl_context_4d = pl_context_raw.reshape(1, 16, 35, 256)
    
    # RMSNorm on pl_context_4d
    ms = jnp.mean(jnp.square(pl_context_4d), axis=-1, keepdims=True)
    pl_context_norm = pl_context_4d * jax.lax.rsqrt(ms + 1e-6) * pl_proj_norm_scale
    ple_all = (pl_context_norm + pl_tok_4d) / jnp.sqrt(2.0)
    
    print("Python ple_all norm:", float(jnp.linalg.norm(ple_all)))
    print("Python ple layer 0 norm:", float(jnp.linalg.norm(ple_all[:, :, 0, :])))

if __name__ == "__main__":
    main()
