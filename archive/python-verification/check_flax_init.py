import jax
import jax.numpy as jnp
from gemma import gm
from kauldron import kd

model = gm.nn.Gemma4_E2B()
tokens = jnp.zeros((1, 4), dtype=jnp.int32)
variables = model.init(jax.random.PRNGKey(0), tokens)

print("\n=== Official Flax Parameter Keys and Shapes ===")
def print_keys(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print_keys(v, prefix + k + ".")
        else:
            print(f"  {prefix + k:50s} : {v.shape}")

print_keys(variables["params"])
