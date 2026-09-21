import jax
import jax.numpy as jnp
from flax import linen as nn

batch, seq, features, hidden = 1, 4, 1536, 6144
x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, features))

gate_proj = jax.random.normal(jax.random.PRNGKey(1), (hidden, features))
up_proj = jax.random.normal(jax.random.PRNGKey(2), (hidden, features))
down_proj = jax.random.normal(jax.random.PRNGKey(3), (features, hidden))

# Method 1: Flax FeedForward
gating_w = jnp.stack([gate_proj, up_proj], axis=0) # (2, hidden, features)
gate_flax = jnp.einsum('...F,NHF->...NH', x, gating_w) # (batch, seq, 2, hidden)
act_flax = jax.nn.gelu(gate_flax[..., 0, :]) * gate_flax[..., 1, :]
out_flax = jnp.einsum('...H,HF->...F', act_flax, down_proj.T)

# Method 2: Clojure geglu
gate_clj = jnp.dot(x, gate_proj.T)
up_clj = jnp.dot(x, up_proj.T)

# Clojure GELU formula: 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
c_sqrt = 0.7978845608
act_gate_clj = 0.5 * gate_clj * (1.0 + jnp.tanh(c_sqrt * (gate_clj + 0.044715 * (gate_clj ** 3))))
act_clj = act_gate_clj * up_clj
out_clj = jnp.dot(act_clj, down_proj.T)

diff = jnp.max(jnp.abs(out_flax - out_clj))
print(f"Max diff for FeedForward / GeGLU: {diff}")
