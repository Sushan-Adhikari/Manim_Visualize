"""
Debug script to test generation with different parameters
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
model_path = "./derivative-animator-deepseek-1.3b/merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Test prompt
function = "x^2"
prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate Manim animation code for the derivative of: f(x) = {function}

### Response:
"""

print("\n" + "="*70)
print("PROMPT:")
print("="*70)
print(prompt)
print("="*70)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to('cuda') for k, v in inputs.items()}

print(f"\nPrompt tokens: {inputs['input_ids'].shape[1]}")

# Test 1: Very conservative generation
print("\n" + "="*70)
print("TEST 1: Conservative (temp=0.1, max_new_tokens=100)")
print("="*70)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated.split("### Response:")[-1].strip()
print(response[:500])

# Test 2: Check what's in the first 200 tokens
print("\n" + "="*70)
print("TEST 2: First 200 tokens (temp=0.3)")
print("="*70)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated.split("### Response:")[-1].strip()
print(response[:800])

print("\n" + "="*70)
print("Checking if it starts with 'from manim import *'...")
if response.strip().startswith('from manim import'):
    print("✓ YES! Starts correctly")
else:
    print(f"✗ NO! Starts with: {response[:50]}")
print("="*70)
