"""Quick test of model with simple parameters"""

from inference_finetuned import DerivativeAnimatorInference

print("Loading model...")
generator = DerivativeAnimatorInference()

print("\n" + "="*70)
print("TESTING DIFFERENT GENERATION PARAMETERS")
print("="*70)

function = "x^2"

# Test 1: Very low temperature, greedy
print("\nTest 1: Greedy decoding (temp=0.01, do_sample=False)")
print("-"*70)
code1 = generator.generate_code(
    function=function,
    max_length=3072,
    temperature=0.01,
    do_sample=False,  # Greedy
    max_new_tokens=500,
    min_new_tokens=100
)
print(f"Generated {len(code1)} chars")
print(f"Starts with: {code1[:200]}")

# Test 2: Higher temperature
print("\n\nTest 2: Higher temperature (temp=0.7)")
print("-"*70)
code2 = generator.generate_code(
    function=function,
    max_length=3072,
    temperature=0.7,
    do_sample=True,
    max_new_tokens=500,
    min_new_tokens=100,
    top_p=0.9,
    repetition_penalty=1.2
)
print(f"Generated {len(code2)} chars")
print(f"Starts with: {code2[:200]}")

# Check which one looks better
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Test 1 has 'from manim import *': {'from manim import *' in code1}")
print(f"Test 1 has 'import numpy': {'import numpy' in code1}")
print(f"Test 2 has 'from manim import *': {'from manim import *' in code2}")
print(f"Test 2 has 'import numpy': {'import numpy' in code2}")
