import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

print("=== Gemma4TextAttention.forward Source Code ===")
print(inspect.getsource(Gemma4TextAttention.forward))
