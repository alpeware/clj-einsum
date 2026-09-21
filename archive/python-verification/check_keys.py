import os
from safetensors import safe_open

sf_path = "../.models/gemma-4-E2B/model.safetensors"
with safe_open(sf_path, framework="np") as f:
    keys = list(f.keys())

print(f"Total safetensor keys: {len(keys)}")
for i in [0, 14, 15, 19, 20, 34]:
    l_keys = [k for k in keys if f"layers.{i}." in k]
    print(f"\n--- Layer {i} keys ({len(l_keys)} keys) ---")
    for k in sorted(l_keys):
        print(f"  {k}")
