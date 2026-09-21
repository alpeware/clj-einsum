import jax
import jax.numpy as jnp

x = jnp.array([[2, 105, 2364]], dtype=jnp.int32)
embed = jnp.zeros((262144, 1536), dtype=jnp.float32)

def f(emb, idx):
    return emb[idx]

jaxpr = jax.make_jaxpr(f)(embed, x)
print(jaxpr)
