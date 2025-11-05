"""
Inference script for fine-tuned DerivativeAnimator model
Uses the DeepSeek-Coder-1.3B model fine-tuned on derivative animations
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import tempfile
import os
import re

class DerivativeAnimatorInference:
    def __init__(self, model_path: str = "./derivative-animator-deepseek-1.3b/merged_model"):
        """Initialize the fine-tuned model for inference

        Args:
            model_path: Path to the fine-tuned model (use merged_model for best results)
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n🤖 Loading fine-tuned DerivativeAnimator model...")
        print(f"   Model path: {model_path}")
        print(f"   Device: {self.device}")

        self.load_model()

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        model_path = Path(self.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                "Please train the model first using: python finetuning_train.py"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        print("✓ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    def format_prompt(self, function: str) -> str:
        """Format the prompt for inference - MUST match training format exactly

        Args:
            function: Mathematical function (e.g., "x^2", "sin(x)")

        Returns:
            Formatted prompt string
        """
        # This MUST match the EXACT format used in the training data!
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a complete Manim scene that demonstrates the derivative of: f(x) = {function}

### Response:
"""
        return prompt

    def generate_code(
        self,
        function: str,
        max_length: int = 3072,  # Increased from 2048 to fit full template
        temperature: float = 0.3,  # Slightly increased for more complete generation
        top_p: float = 0.95,  # Increased for more diversity
        top_k: int = 50,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,  # Reduced to allow template patterns
        do_sample: bool = True,
    ) -> str:
        """Generate Manim code for the given function

        Args:
            function: Mathematical function (e.g., "x^2", "sin(x)")
            max_length: Maximum length of generated code
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search (1 = greedy)
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (False = greedy)

        Returns:
            Generated Manim code as string
        """
        prompt = self.format_prompt(function)

        print(f"\n📝 Generating code for: f(x) = {function}")
        print(f"   Parameters: temp={temperature}, top_p={top_p}, top_k={top_k}")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Prompt should be short
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Generate up to 2048 NEW tokens (not including prompt)
                min_new_tokens=800,   # Ensure at least 800 tokens generated
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (after "### Response:")
        if "### Response:" in generated_text:
            code = generated_text.split("### Response:")[1].strip()
        else:
            code = generated_text.strip()

        print(f"✓ Generated {len(code)} characters")

        return code

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate the generated Manim code

        Args:
            code: Generated Python code

        Returns:
            (is_valid, message) tuple
        """
        # Check for required imports
        if "from manim import" not in code:
            return False, "Missing 'from manim import *'"

        if "import numpy as np" not in code:
            return False, "Missing 'import numpy as np'"

        # Check for class definition
        if "class DerivativeVisualization(Scene):" not in code:
            return False, "Missing DerivativeVisualization class"

        # Check for construct method
        if "def construct(self):" not in code:
            return False, "Missing construct method"

        return True, "Code structure valid"

    def test_execution(self, code: str, timeout: int = 45) -> tuple[bool, str]:
        """Test if the generated code can be executed by Manim

        Args:
            code: Python code to test
            timeout: Maximum execution time in seconds

        Returns:
            (success, message) tuple
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Extract class name
            class_match = re.search(r'class (\w+)\(Scene\):', code)
            if not class_match:
                return False, "Could not find Scene class"

            class_name = class_match.group(1)

            # Run Manim dry-run
            result = subprocess.run(
                ['manim', '-pql', '--dry_run', temp_file, class_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                error_lines = result.stderr.split('\n')
                relevant_errors = [line for line in error_lines if 'Error' in line]
                error_msg = '\n'.join(relevant_errors[-5:]) if relevant_errors else result.stderr[-500:]
                return False, f"Execution failed:\n{error_msg}"

            return True, "Code executed successfully"

        except subprocess.TimeoutExpired:
            return False, f"Execution timeout (>{timeout}s)"
        except Exception as e:
            return False, f"Test error: {str(e)}"
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    def save_code(self, code: str, function: str, output_dir: str = "generated_manim") -> str:
        """Save generated code to file

        Args:
            code: Generated code
            function: Function name for filename
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        Path(output_dir).mkdir(exist_ok=True)

        # Create safe filename
        safe_name = re.sub(r'[^\w\s-]', '', function).strip().replace(' ', '_')
        if not safe_name:
            safe_name = "function"
        filename = f"{output_dir}/derivative_{safe_name}.py"

        with open(filename, 'w') as f:
            f.write(code)

        return filename


def main():
    """Interactive CLI for generating derivative animations"""
    print("\n" + "="*70)
    print("DERIVATIVE ANIMATOR - Fine-tuned Model Inference")
    print("="*70)

    # Initialize model
    try:
        generator = DerivativeAnimatorInference()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n📚 Examples: x^2, sin(x), x^3+2*x^2-x+1, x*sin(x), e^x")
    print("-"*70)

    while True:
        function = input("\n🔢 Enter function (or 'quit' to exit): ").strip()

        if function.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not function:
            print("❌ Please enter a function")
            continue

        try:
            # Generate code
            code = generator.generate_code(
                function=function,
                max_length=3072,  # Ensure full template generation
                temperature=0.3,  # Slightly higher for complete output
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,  # Lower to allow template patterns
                do_sample=True
            )

            # Validate
            is_valid, msg = generator.validate_code(code)
            if not is_valid:
                print(f"\n❌ Validation failed: {msg}")
                print("\nGenerated code preview:")
                print(code[:500])
                continue

            print(f"✓ {msg}")

            # Test execution
            print("\n🧪 Testing code execution...")
            can_execute, test_msg = generator.test_execution(code)

            if not can_execute:
                print(f"❌ {test_msg}")
                print("\nCode preview:")
                print(code[:500])

                # Ask if user wants to save anyway
                save = input("\nSave code anyway? (y/n): ").strip().lower()
                if save == 'y':
                    filename = generator.save_code(code, function)
                    print(f"✓ Saved to: {filename}")
                continue

            print(f"✓ {test_msg}")

            # Save code
            filename = generator.save_code(code, function)
            print(f"\n✅ Code saved to: {filename}")
            print(f"\nTo render:")
            print(f"   manim -pql {filename} DerivativeVisualization")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
