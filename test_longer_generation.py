"""
Test with progressively more tokens to see if model follows template
"""

from inference_finetuned import DerivativeAnimatorInference
import torch

print("Loading model...")
generator = DerivativeAnimatorInference()

function = "x^2"
prompt = generator.format_prompt(function)

inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(generator.device) for k, v in inputs.items()}

# Test with increasing token counts
for num_tokens in [100, 300, 500, 1000]:
    print(f"\n{'='*70}")
    print(f"GENERATING {num_tokens} TOKENS (greedy)")
    print('='*70)

    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,  # Greedy
            pad_token_id=generator.tokenizer.pad_token_id,
        )

    generated_text = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("### Response:")[-1].strip()

    print(f"Generated {len(response)} characters")

    # Check key elements
    checks = {
        'import numpy': 'import numpy' in response,
        'DerivativeVisualization': 'DerivativeVisualization' in response,
        'axes = Axes': 'axes = Axes' in response,
        'calc_step': 'calc_step' in response,
        'MathTex': 'MathTex' in response,
    }

    print("\nKey elements present:")
    for key, present in checks.items():
        print(f"  {'✓' if present else '✗'} {key}")

    # Show first 500 chars
    print(f"\nFirst 500 characters:")
    print(response[:500])

    if num_tokens == 1000:
        print(f"\n... (showing first 500 of {len(response)} total chars)")

print("\n" + "="*70)
print("If we see the template elements appearing as tokens increase,")
print("we just need to generate MORE tokens!")
print("="*70)
