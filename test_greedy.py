"""Quick test with greedy decoding"""
from inference_finetuned import DerivativeAnimatorInference
import torch

generator = DerivativeAnimatorInference()

function = "x^2"
prompt = generator.format_prompt(function)

print("Prompt:")
print(prompt)
print("="*70)

inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(generator.device) for k, v in inputs.items()}

# GREEDY - most deterministic
print("\nGenerating with GREEDY decoding (500 tokens)...")
with torch.no_grad():
    outputs = generator.model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,  # Pure greedy
        pad_token_id=generator.tokenizer.pad_token_id,
    )

generated = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated.split("### Response:")[-1].strip()

print("\nGenerated:")
print("="*70)
print(response[:1000])
print("="*70)

# Check
if "from manim import" in response:
    print("\n✓ Generates Manim code!")
elif "class DerivativeVisualization" in response:
    print("\n✓ Has DerivativeVisualization!")
elif "###" in response or "Instruction:" in response:
    print("\n❌ Generating more instructions/prompts")
elif "github" in response.lower() or "task" in response.lower():
    print("\n❌ Generating random text/markdown")
else:
    print(f"\n⚠️  Unexpected output")
