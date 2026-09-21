import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

print("=== Gemma4TextDecoderLayer Source Code ===")
print(inspect.getsource(Gemma4TextDecoderLayer.forward))
