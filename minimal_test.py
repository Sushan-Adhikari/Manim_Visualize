"""
Minimal test - try to generate ANYTHING reasonable from the model
"""

try:
    from inference_finetuned import DerivativeAnimatorInference

    print("Loading model...")
    generator = DerivativeAnimatorInference()

    # Try with absolute minimal generation to see what we get
    import torch

    function = "x^2"
    prompt = generator.format_prompt(function)

    print("\nPrompt being used:")
    print("="*70)
    print(prompt)
    print("="*70)

    # Tokenize
    inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(generator.device) for k, v in inputs.items()}

    print(f"\nInput tokens: {inputs['input_ids'].shape[1]}")

    # Try VERY basic generation - just 50 tokens with greedy decoding
    print("\nGenerating with greedy decoding (50 tokens)...")
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy - take most likely token each time
            pad_token_id=generator.tokenizer.pad_token_id,
        )

    generated_text = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("### Response:")[-1].strip()

    print("\nGenerated (first 50 tokens):")
    print("="*70)
    print(response)
    print("="*70)

    # Check if it's reasonable
    if "from manim" in response or "import" in response or "class" in response:
        print("\n✓ Model is generating code-like output!")
    elif "Dot()" in response and response.count("Dot()") > 3:
        print("\n❌ Model is stuck in repetition loop")
    else:
        print(f"\n⚠️  Unexpected output type")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
