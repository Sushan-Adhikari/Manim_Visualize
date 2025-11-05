"""
Verify what the model actually learned during training
"""

import json
from transformers import AutoTokenizer

print("="*70)
print("VERIFYING TRAINING SETUP")
print("="*70)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./derivative-animator-deepseek-1.3b/merged_model",
    trust_remote_code=True
)

# Load a training example
with open('derivative_dataset_537/finetuning/hf_train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'x^2' in data['input'] and data['input'].strip() == 'x^2':
            sample = data
            break

print("\n1. TRAINING EXAMPLE (x^2):")
print("-"*70)
print(f"Instruction: {sample['instruction']}")
print(f"Input: {sample['input']}")
print(f"Output length: {len(sample['output'])} chars")
print(f"Output starts with: {sample['output'][:200]}")

# Reconstruct the exact training prompt
instruction_text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate Manim animation code for the derivative of: f(x) = {sample['input']}

### Response:
"""

response_text = sample['output'] + tokenizer.eos_token

full_training_text = instruction_text + response_text

print("\n2. FULL TRAINING SEQUENCE:")
print("-"*70)
print(f"Total length: {len(full_training_text)} chars")
print(f"Instruction length: {len(instruction_text)} chars")
print(f"Response length: {len(response_text)} chars")

# Tokenize it
tokens = tokenizer(full_training_text, truncation=True, max_length=2048)
print(f"\nTokenized length: {len(tokens['input_ids'])} tokens")
print(f"Was truncated: {len(full_training_text.encode('utf-8')) > 2048 * 4}")  # Rough check

# Check if response fits
instruction_tokens = tokenizer(instruction_text, truncation=False)
response_tokens = tokenizer(response_text, truncation=False)
print(f"Instruction tokens: {len(instruction_tokens['input_ids'])}")
print(f"Response tokens: {len(response_tokens['input_ids'])}")
print(f"Total if not truncated: {len(instruction_tokens['input_ids']) + len(response_tokens['input_ids'])}")

if len(tokens['input_ids']) < (len(instruction_tokens['input_ids']) + len(response_tokens['input_ids'])):
    print("\n⚠️  WARNING: Training data was TRUNCATED!")
    print(f"   Lost {len(response_tokens['input_ids']) - (len(tokens['input_ids']) - len(instruction_tokens['input_ids']))} tokens")
    print("   Model never saw the full template!")
else:
    print("\n✓ Full sequence fit within max_length")

# Check what was in the truncated output
if len(tokens['input_ids']) == 2048:
    print("\n3. WHAT MODEL SAW (decoded from 2048 tokens):")
    print("-"*70)
    decoded = tokenizer.decode(tokens['input_ids'])
    print(decoded[-500:])  # Last 500 chars
    print("\n^ This is what the model learned to generate!")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
if len(tokens['input_ids']) == 2048 and "self.wait(1)" not in decoded[-500:]:
    print("❌ PROBLEM: Training data was truncated!")
    print("   Model never saw the complete template during training")
    print("   It learned to generate whatever fit in the truncated portion")
else:
    print("✓ Training data should have been complete")
