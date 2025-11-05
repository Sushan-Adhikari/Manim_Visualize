"""
Convert instruction dataset to HuggingFace JSONL format for training
"""

import json
from pathlib import Path

def convert_to_hf_format():
    """Convert instruction_dataset.json to hf_train.jsonl and hf_validation.jsonl"""

    print("Loading instruction dataset...")
    with open('derivative_dataset_537/finetuning/instruction_dataset.json') as f:
        data = json.load(f)

    # Process training data
    print(f"Processing {len(data['train'])} training samples...")
    with open('derivative_dataset_537/finetuning/hf_train.jsonl', 'w') as f:
        for item in data['train']:
            # Extract function from instruction
            # "Build an animated derivative visualization for: f(x) = x^2"
            instruction = item['instruction']
            func_match = instruction.split('f(x) = ')[-1] if 'f(x) = ' in instruction else "unknown"

            hf_item = {
                'instruction': instruction,
                'input': func_match,
                'output': item['output']
            }
            f.write(json.dumps(hf_item) + '\n')

    # Process validation data
    print(f"Processing {len(data['validation'])} validation samples...")
    with open('derivative_dataset_537/finetuning/hf_validation.jsonl', 'w') as f:
        for item in data['validation']:
            instruction = item['instruction']
            func_match = instruction.split('f(x) = ')[-1] if 'f(x) = ' in instruction else "unknown"

            hf_item = {
                'instruction': instruction,
                'input': func_match,
                'output': item['output']
            }
            f.write(json.dumps(hf_item) + '\n')

    # Process test data
    print(f"Processing {len(data['test'])} test samples...")
    with open('derivative_dataset_537/finetuning/hf_test.jsonl', 'w') as f:
        for item in data['test']:
            instruction = item['instruction']
            func_match = instruction.split('f(x) = ')[-1] if 'f(x) = ' in instruction else "unknown"

            hf_item = {
                'instruction': instruction,
                'input': func_match,
                'output': item['output']
            }
            f.write(json.dumps(hf_item) + '\n')

    print("✓ Created hf_train.jsonl")
    print("✓ Created hf_validation.jsonl")
    print("✓ Created hf_test.jsonl")
    print("\nDataset ready for fine-tuning!")

if __name__ == "__main__":
    convert_to_hf_format()
