import os
import sys
import jax
import jax.numpy as jnp
from safetensors import safe_open
from tokenizers import Tokenizer
from verify_inference import load_safetensors_flax_params
from gemma.gm.nn.gemma4 import _modules, _config

def main():
    model_dir = "../.models/gemma-4-E2B"
    if not os.path.exists(model_dir):
        model_dir = "../.models/gemma-4-E2B-it"
        
    variables = load_safetensors_flax_params(model_dir)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    encoded = tok.encode("The capital of France is")
    prompt_ids = encoded.ids
    tokens = jnp.array([prompt_ids], dtype=jnp.int32)
    positions = jnp.arange(len(prompt_ids), dtype=jnp.int32)[None, :]
    
    cfg = _config.TransformerConfig(
        num_embed=262144,
        embed_dim=1536,
        hidden_dim=6144,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=30.0,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attention_types=_config.make_attention_layers_types(
            (_modules.AttentionType.LOCAL_SLIDING,) * 4 + (_modules.AttentionType.GLOBAL,),
            num_layers=35
        ),
        sliding_window_size=512,
        qk_norm_with_scale=True,
        global_rope_proportion=0.25,
        local_rope_proportion=1.0,
    )
    
    model = _transformer.Transformer(config=cfg)
    
    res = model.apply(variables, tokens, positions=positions)
    logits = res.logits[0, -1]
    
    top_10 = jnp.argsort(logits)[::-1][:10]
    print("\n=== Top 10 Reference Base Model Predicted Tokens ===")
    for tid in top_10:
        t_str = tok.decode([int(tid)])
        print(f"  token {int(tid):6d} | logit: {float(logits[tid]):8.4f} | text: {repr(t_str)}")

if __name__ == "__main__":
    from gemma.gm.nn.gemma4 import _transformer
    main()
