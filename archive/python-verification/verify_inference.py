import os
import sys
import types
import importlib.machinery

tf = types.ModuleType('tensorflow')
tf.Tensor = object
tf.dtypes = types.ModuleType('dtypes')
tf.dtypes.DType = type('DType', (), {})
tf.__spec__ = importlib.machinery.ModuleSpec('tensorflow', None)
sys.modules['tensorflow'] = tf

import jax
import jax.numpy as jnp
from safetensors import safe_open
from gemma import gm
from tokenizers import Tokenizer

def load_safetensors_flax_params(model_dir):
    sf_path = os.path.join(model_dir, "model.safetensors")
    sf_tensors = {}
    with safe_open(sf_path, framework="np") as f:
        for k in f.keys():
            sf_tensors[k] = f.get_tensor(k)
            
    params = {
        "embedder": {
            "input_embedding": sf_tensors["model.language_model.embed_tokens.weight"],
            "per_layer_embeddings": sf_tensors["model.language_model.embed_tokens_per_layer.weight"].reshape(262144, 35, 256),
            "per_layer_model_projection": {"w": sf_tensors["model.language_model.per_layer_model_projection.weight"].reshape(35, 256, 1536).transpose(2, 0, 1)},
            "per_layer_projection_norm": {"scale": sf_tensors["model.language_model.per_layer_projection_norm.weight"]},
        },
        "final_norm": {"scale": sf_tensors["model.language_model.norm.weight"]},
    }
    
    for i in range(35):
        prefix = f"model.language_model.layers.{i}."
        is_global = ((i + 1) % 5 == 0)
        q_dim = 4096 if is_global else 2048
        kv_dim = 512 if is_global else 256
        head_dim = 512 if is_global else 256
        mlp_dim = 12288 if i >= 15 else 6144
        
        layer_dict = {
            "pre_attention_norm": {"scale": sf_tensors[prefix + "input_layernorm.weight"]},
            "skip_scale": sf_tensors[prefix + "layer_scalar"],
            "post_attention_norm": {"scale": sf_tensors[prefix + "post_attention_layernorm.weight"]},
            "pre_ffw_norm": {"scale": sf_tensors[prefix + "pre_feedforward_layernorm.weight"]},
            "post_ffw_norm": {"scale": sf_tensors[prefix + "post_feedforward_layernorm.weight"]},
            "per_layer_input_gate": {"w": sf_tensors[prefix + "per_layer_input_gate.weight"].T},
            "per_layer_projection": {"w": sf_tensors[prefix + "per_layer_projection.weight"].T},
            "post_per_layer_input_norm": {"scale": sf_tensors[prefix + "post_per_layer_input_norm.weight"]},
            "attn": {
                "q_einsum": {"w": sf_tensors[prefix + "self_attn.q_proj.weight"].T.reshape(8, 1536, head_dim)},
                "kv_einsum": {"w": jnp.stack([
                    sf_tensors[prefix + "self_attn.k_proj.weight"].T.reshape(1, 1536, head_dim),
                    sf_tensors[prefix + "self_attn.v_proj.weight"].T.reshape(1, 1536, head_dim)
                ], axis=0)},
                "attn_vec_einsum": {"w": sf_tensors[prefix + "self_attn.o_proj.weight"].reshape(8, head_dim, 1536)},
                "query_norm": {"scale": sf_tensors[prefix + "self_attn.q_norm.weight"]},
                "key_norm": {"scale": sf_tensors[prefix + "self_attn.k_norm.weight"]},
            },
            "mlp": {
                "gating_einsum": {"w": jnp.stack([
                    sf_tensors[prefix + "mlp.gate_proj.weight"],
                    sf_tensors[prefix + "mlp.up_proj.weight"]
                ], axis=0)},
                "linear": {"w": sf_tensors[prefix + "mlp.down_proj.weight"].T},
            }
        }
        params[f"layer_{i}"] = layer_dict
        
    return {"params": params}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, "..", ".models", "gemma-4-E2B-it"))
    if not os.path.exists(model_dir):
        model_dir = os.path.abspath(os.path.join(script_dir, "..", ".models", "gemma-4-E2B"))
    print(f"Loading Flax params from safetensors in {model_dir}...")
    variables = load_safetensors_flax_params(model_dir)
    model = gm.nn.Gemma4_E2B()
    
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    encoded = tok.encode("What is the capital of France?")
    clean_ids = encoded.ids
    if clean_ids[0] == 2:
        clean_ids = clean_ids[1:]
    prompt_ids = [2, 105, 2364, 107] + clean_ids + [106, 107, 105, 4368, 107]
        
    print(f"Prompt IDs ({len(prompt_ids)} tokens): {prompt_ids}")
    cur_tokens = list(prompt_ids)
    
    print("\nGenerating tokens autoregressively with Flax Gemma 4 model...")
    for step in range(20):
        seq_len = len(cur_tokens)
        tokens = jnp.array([cur_tokens], dtype=jnp.int32)
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        out = model.apply(variables, tokens, positions=positions)
        logits = out.logits[0, -1] # last token logits
        next_id = int(jnp.argmax(logits))
        next_str = tok.decode([next_id])
        cur_tokens.append(next_id)
        print(f"Step {step+1:2d} | Token ID: {next_id:6d} | logit: {float(logits[next_id]):8.4f} | text: {repr(next_str)}")
        if next_id == 1 or next_id == 106:
            break
            
    full_text = tok.decode(cur_tokens)
    print("\n=== Reference Autoregressive Generated Text ===")
    print(full_text)

if __name__ == "__main__":
    main()
