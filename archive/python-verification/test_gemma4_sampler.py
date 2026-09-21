import os
import sys
import jax
from gemma import gm
from tokenizers import Tokenizer
import sentencepiece as spm
from verification.verify_inference import load_safetensors_flax_params

def main():
    model_dir = "../.models/gemma-4-E2B-it"
    if not os.path.exists(model_dir):
        model_dir = "../.models/gemma-4-E2B"
        
    print(f"Loading model and parameters from {model_dir}...")
    variables = load_safetensors_flax_params(model_dir)
    model = gm.nn.Gemma4_E2B()
    
    # Check turn format by running Gemma4Sampler
    print("Testing prompt generation...")
    prompt = "<|turn|>user\nWhat is the capital of France?<turn|>\n<|turn|>model\n"
    
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    encoded = tok.encode("What is the capital of France?")
    print("Encoded prompt IDs without turn tags:", encoded.ids)

if __name__ == "__main__":
    main()
