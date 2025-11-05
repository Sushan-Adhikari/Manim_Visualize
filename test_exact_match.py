"""Test with EXACT instruction from training data"""
from inference_finetuned import DerivativeAnimatorInference
import torch
import json

generator = DerivativeAnimatorInference()

# Load EXACT example from training
with open('derivative_dataset_537/finetuning/hf_train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if 'x^2' in d['input'] and len(d['input']) < 10:  # Find simple x^2
            exact_instruction = d['instruction']
            expected_output = d['output']
            break

print("Using EXACT training instruction:")
print(exact_instruction)
print("="*70)

# Build prompt EXACTLY as training
prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{exact_instruction}

### Response:
"""

inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(generator.device) for k, v in inputs.items()}

print("\nGenerating with greedy decoding...")
with torch.no_grad():
    outputs = generator.model.generate(
        **inputs,
        max_new_tokens=1500,
        do_sample=False,
        pad_token_id=generator.tokenizer.pad_token_id,
    )

generated = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated.split("### Response:")[-1].strip()

print("\nGenerated (first 800 chars):")
print("="*70)
print(response[:800])
print("="*70)

# Check key elements
checks = {
    'from manim import *': 'from manim import *' in response,
    'import numpy': 'import numpy' in response,
    'DerivativeVisualization': 'DerivativeVisualization' in response,
    'axes = Axes': 'axes = Axes' in response,
}

print("\nChecks:")
for key, result in checks.items():
    print(f"  {'✓' if result else '✗'} {key}")

if all(checks.values()):
    print("\n✅ SUCCESS! Model generates correct template!")
else:
    print("\n❌ Still not working")
