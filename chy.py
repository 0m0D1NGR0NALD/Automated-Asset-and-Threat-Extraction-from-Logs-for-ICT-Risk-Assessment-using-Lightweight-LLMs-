from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Suppress symlink warning (optional)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Fix for torch.float8_e8m0fnu attribute error
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Test generation - FIX: move inputs to same device as model
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
# Move inputs to the model's device (CUDA)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))