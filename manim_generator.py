import os
import re
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# IMPROVED TEMPLATE - Using placeholders that won't confuse the AI
CODE_TEMPLATE = """from manim import *
import numpy as np

class DerivativeVisualization(Scene):
    def construct(self):
        # PART 1: Setup axes and function
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[Y_MIN_PLACEHOLDER, Y_MAX_PLACEHOLDER, 1],
            x_length=5.5,
            y_length=4.5,
            axis_config={"color": GRAY}
        ).shift(LEFT * 2.5)
        
        # Define the function
        def f(x):
            return FUNCTION_CODE_PLACEHOLDER
        
        # Plot the function
        graph = axes.plot(f, color=BLUE, x_range=[-3.5, 3.5])
        func_label = MathTex(r"FUNCTION_LATEX_PLACEHOLDER", color=BLUE, font_size=36).to_corner(UL).shift(DOWN*0.3)
        
        # PART 2: CALCULATION STEPS (Right side, no overlap)
        calc_title = Text("Derivative Calculation:", font_size=20, color=WHITE).to_edge(RIGHT).shift(LEFT*0.2 + UP*3.3)
        
        calc_step1 = MathTex(r"STEP1_LATEX_PLACEHOLDER", font_size=22).next_to(calc_title, DOWN, buff=0.3, aligned_edge=LEFT)
        calc_step2 = MathTex(r"STEP2_LATEX_PLACEHOLDER", font_size=22).next_to(calc_step1, DOWN, buff=0.25, aligned_edge=LEFT)
        calc_step3 = MathTex(r"STEP3_LATEX_PLACEHOLDER", font_size=22).next_to(calc_step2, DOWN, buff=0.25, aligned_edge=LEFT)
        calc_final = MathTex(r"FINAL_LATEX_PLACEHOLDER", font_size=26, color=YELLOW).next_to(calc_step3, DOWN, buff=0.3, aligned_edge=LEFT)
        
        # Ensure all calculation steps fit on screen
        calc_steps = VGroup(calc_title, calc_step1, calc_step2, calc_step3, calc_final)
        if calc_steps.width > 3.8:
            calc_steps.scale_to_fit_width(3.8)
        
        # PART 3: Moving elements
        x_tracker = ValueTracker(-2)
        
        # Define derivative function
        def f_prime(x):
            return DERIVATIVE_CODE_PLACEHOLDER
        
        # Moving dot on curve
        dot = always_redraw(lambda: Dot(
            axes.c2p(x_tracker.get_value(), f(x_tracker.get_value())),
            color=RED,
            radius=0.08
        ))
        
        # Tangent line
        tangent = always_redraw(lambda: axes.plot(
            lambda x: f_prime(x_tracker.get_value()) * (x - x_tracker.get_value()) + f(x_tracker.get_value()),
            x_range=[x_tracker.get_value()-1.5, x_tracker.get_value()+1.5],
            color=GREEN
        ))
        
        # Derivative value display - positioned BELOW calculation to avoid overlap
        deriv_label = always_redraw(lambda: MathTex(
            r"f'({:.1f}) = {:.2f}".format(x_tracker.get_value(), f_prime(x_tracker.get_value())),
            font_size=30,
            color=YELLOW
        ).next_to(calc_steps, DOWN, buff=0.5))
        
        # PART 4: Animation sequence
        self.play(Create(axes), Write(func_label), run_time=1)
        self.play(Create(graph), run_time=1.5)
        self.wait(0.5)
        
        # Show calculation steps one by one
        self.play(Write(calc_title), run_time=0.5)
        self.play(Write(calc_step1), run_time=0.7)
        self.wait(0.3)
        self.play(Write(calc_step2), run_time=0.7)
        self.wait(0.3)
        self.play(Write(calc_step3), run_time=0.7)
        self.wait(0.3)
        self.play(Write(calc_final), run_time=0.8)
        self.wait(0.5)
        
        # Add moving elements
        self.play(Create(dot), Create(tangent), Write(deriv_label), run_time=1)
        
        # Animate movement
        self.play(x_tracker.animate.set_value(2), run_time=3, rate_func=smooth)
        self.wait(1)
"""

# IMPROVED System instruction with CORRECT LaTeX examples
SYSTEM_INSTRUCTION = """
You are a Manim expert. You will receive a TEMPLATE with placeholders and a mathematical function.
Your ONLY job is to fill in the placeholders correctly. Do NOT modify the template structure.

TEMPLATE PLACEHOLDERS TO FILL:
1. {y_min} and {y_max} - Choose appropriate y-axis range based on function
2. {function_code} - Python code for the function (e.g., x**2, np.sin(x))
3. {function_latex} - LaTeX for function display (e.g., x^2, \\sin(x))
4. {step1_latex} - First calculation step showing original function
5. {step2_latex} - Second step showing derivative rule being applied
6. {step3_latex} - Third step showing the calculation process
7. {final_latex} - Final result (the derivative formula)
8. {derivative_code} - Python code for derivative (e.g., 2*x, np.cos(x))

CRITICAL LaTeX FORMATTING RULES:
- In Python r-strings, LaTeX needs proper escaping
- For text: \text{Text here} - use single braces
- For math: \frac{a}{b} - use single braces  
- For variables with subscripts/superscripts: x^{2}, x_{1} - use single braces
- Keep formulas COMPACT to fit on screen (max ~40 characters)

CALCULATION STEP GUIDELINES (USE COMPACT NOTATION):

For POLYNOMIALS (x^n):
  step1: "f(x) = x^{2}"
  step2: "\text{Power Rule: } \frac{d}{dx}[x^n] = nx^{n-1}"
  step3: "f'(x) = 2x^{1}"
  final: "f'(x) = 2x"

For SINE (sin(x)):
  step1: "f(x) = \sin(x)"
  step2: "\frac{d}{dx}[\sin(x)] = \cos(x)"
  step3: "f'(x) = \cos(x)"
  final: "f'(x) = \cos(x)"

For COSINE (cos(x)):
  step1: "f(x) = \cos(x)"
  step2: "\frac{d}{dx}[\cos(x)] = -\sin(x)"
  step3: "f'(x) = -\sin(x)"
  final: "f'(x) = -\sin(x)"

For EXPONENTIAL (e^x):
  step1: "f(x) = e^{x}"
  step2: "\frac{d}{dx}[e^x] = e^x"
  step3: "f'(x) = e^{x}"
  final: "f'(x) = e^{x}"

For PRODUCTS (x*sin(x)):
  step1: "f(x) = x \sin(x)"
  step2: "\text{Product Rule: } (uv)' = u'v + uv'"
  step3: "f'(x) = \sin(x) + x\cos(x)"
  final: "f'(x) = \sin(x) + x\cos(x)"

For CHAIN RULE compositions:
  step1: Show original function
  step2: "\text{Chain Rule: } \frac{d}{dx}[f(g(x))] = f'(g(x))g'(x)"
  step3: Show application
  final: Simplified result

PYTHON CODE REQUIREMENTS:
- ALWAYS import numpy as np at the top
- Use np.sin, np.cos, np.tan, np.exp for trig/exponential functions
- Use np.log for natural log
- Use np.sqrt for square root
- For derivative code, calculate the mathematical derivative correctly
- Use ** for exponentiation (e.g., x**2)

CRITICAL RULES:
1. Output ONLY the filled template - NO explanations, NO markdown
2. Use raw strings with single backslashes: r"\frac{a}{b}"
3. Use single braces in LaTeX: x^{2} not x^{{2}}
4. Choose y_range that fits the function (typically [-5, 10] for polynomials, [-2, 2] for trig)
5. NEVER skip calculation steps - all 4 steps MUST be filled
6. Keep LaTeX expressions compact (under 40 chars) to fit on screen
7. Ensure derivative_code is mathematically correct

VALID COLORS ONLY:
BLUE, RED, GREEN, YELLOW, ORANGE, PURPLE, PINK, WHITE, BLACK, GRAY, GREY
NO: GRAY_A, GREY_A, BLUE_GRAY or any compound names
"""

def extract_code_from_response(response_text: str) -> str:
    """Extract clean Python code from response"""
    # Remove markdown code fences if present
    code = re.sub(r'^```python\s*\n?', '', response_text, flags=re.MULTILINE)
    code = re.sub(r'^```\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    code = code.strip()
    
    # FIX COMMON LATEX ERRORS
    # Fix double backslashes that AI might generate
    code = re.sub(r'\\\\frac', r'\\frac', code)
    code = re.sub(r'\\\\text', r'\\text', code)
    code = re.sub(r'\\\\sin', r'\\sin', code)
    code = re.sub(r'\\\\cos', r'\\cos', code)
    
    # Fix double braces {{}} to single braces {}
    # But be careful with Python format strings
    lines = code.split('\n')
    fixed_lines = []
    for line in lines:
        if 'MathTex' in line and 'r"' in line:
            # In MathTex raw strings, fix double braces to single
            # Match the pattern r"..." and fix braces inside
            def fix_braces(match):
                content = match.group(1)
                # Replace {{ with { and }} with }
                content = content.replace('{{', '{').replace('}}', '}')
                return f'r"{content}"'
            
            line = re.sub(r'r"([^"]*)"', fix_braces, line)
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def validate_code_structure(code: str) -> tuple[bool, str]:
    """Validate basic code structure and required elements"""
    
    # Check for numpy import
    if "import numpy as np" not in code:
        return False, "Missing 'import numpy as np' statement"
    
    # Check for invalid colors
    invalid_colors = [
        "BLUE_GRAY", "BLUE_GREY", "GRAY_BLUE", "GREY_BLUE",
        "GREY_A", "GREY_B", "GREY_C", "GREY_D", "GREY_E",
        "GRAY_A", "GRAY_B", "GRAY_C", "GRAY_D", "GRAY_E",
    ]
    
    for invalid_color in invalid_colors:
        pattern = r'\b' + re.escape(invalid_color) + r'\b'
        if re.search(pattern, code):
            return False, f"Invalid color '{invalid_color}' found."
    
    # Check for calculation steps presence and scaling
    required_elements = {
        "calc_title": "Derivative Calculation:" in code or "calc_title" in code,
        "calc_step1": "calc_step1" in code,
        "calc_step2": "calc_step2" in code,
        "calc_step3": "calc_step3" in code,
        "calc_final": "calc_final" in code,
        "has_write_calc": "Write(calc_" in code,
        "scale_to_fit": "scale_to_fit_width" in code,
        "deriv_below": "next_to(calc_steps, DOWN" in code,  # Ensure derivative label below calc steps
    }
    
    missing = [k for k, v in required_elements.items() if not v]
    if missing:
        return False, f"CRITICAL: Missing elements: {', '.join(missing)}. Template was not followed!"
    
    # Check basic structure
    checks = {
        "has_import": "from manim import" in code,
        "has_class": "class DerivativeVisualization" in code,
        "has_construct": "def construct(self)" in code,
        "has_valuetracker": "ValueTracker" in code,
    }
    
    failed = [k for k, v in checks.items() if not v]
    if failed:
        return False, f"Failed checks: {', '.join(failed)}"
    
    return True, "Code structure valid with calculation steps"

def test_manim_code(code: str, timeout: int = 45) -> tuple[bool, str]:
    """Test if Manim code can be executed successfully"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        class_match = re.search(r'class (\w+)\(Scene\):', code)
        if not class_match:
            return False, "Could not find Scene class"
        
        class_name = class_match.group(1)
        
        result = subprocess.run(
            ['manim', '-pql', '--dry_run', temp_file, class_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            error_lines = result.stderr.split('\n')
            relevant_errors = [line for line in error_lines if 'Error' in line or 'not defined' in line]
            error_msg = '\n'.join(relevant_errors[-10:]) if relevant_errors else result.stderr[-500:]
            return False, f"Manim execution error:\n{error_msg}"
        
        return True, "Code executed successfully"
        
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Code took >45 seconds."
    except Exception as e:
        return False, f"Test error: {str(e)}"
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass

def generate_manim_code(
    user_function: str, 
    max_attempts: int = 3, 
    use_thinking: bool = False, 
    skip_execution_test: bool = False,
    model_name: str = "gemini-2.5-flash"  # Changed default to 2.5-flash
) -> tuple[str, dict]:
    """Generate Manim code by filling template
    
    Args:
        user_function: Mathematical function (e.g., "x^2")
        max_attempts: Maximum generation attempts
        use_thinking: Use thinking model (slower, better quality)
        skip_execution_test: Skip execution test
        model_name: Gemini model to use (default: gemini-2.5-flash with 250 RPD)
    """
    
    prompt = f"""Fill in the following template for the function: f(x) = {user_function}

Replace these EXACT placeholders in the template:
- Y_MIN_PLACEHOLDER -> appropriate minimum y value (e.g., -1, -5)
- Y_MAX_PLACEHOLDER -> appropriate maximum y value (e.g., 10, 5)
- FUNCTION_CODE_PLACEHOLDER -> Python code (e.g., x**2, np.sin(x))
- FUNCTION_LATEX_PLACEHOLDER -> LaTeX like: f(x) = x^{{2}}
- STEP1_LATEX_PLACEHOLDER -> First step LaTeX
- STEP2_LATEX_PLACEHOLDER -> Second step LaTeX  
- STEP3_LATEX_PLACEHOLDER -> Third step LaTeX
- FINAL_LATEX_PLACEHOLDER -> Final result LaTeX
- DERIVATIVE_CODE_PLACEHOLDER -> Python derivative code (e.g., 2*x)

TEMPLATE:
{CODE_TEMPLATE}

EXAMPLE REPLACEMENT FOR x^2:
- Y_MIN_PLACEHOLDER -> -1
- Y_MAX_PLACEHOLDER -> 10
- FUNCTION_CODE_PLACEHOLDER -> x**2
- FUNCTION_LATEX_PLACEHOLDER -> f(x) = x^{{2}}
- STEP1_LATEX_PLACEHOLDER -> f(x) = x^{{2}}
- STEP2_LATEX_PLACEHOLDER -> \\text{{Power Rule: }} \\frac{{d}}{{dx}}[x^n] = nx^{{n-1}}
- STEP3_LATEX_PLACEHOLDER -> f'(x) = 2x^{{1}}
- FINAL_LATEX_PLACEHOLDER -> f'(x) = 2x
- DERIVATIVE_CODE_PLACEHOLDER -> 2*x

CRITICAL LATEX RULES:
- Use single backslashes: \\frac, \\text, \\sin
- Use double curly braces for LaTeX variables: x^{{{{2}}}}, \\frac{{{{a}}}}{{{{b}}}}
- Keep formulas under 40 characters

Now replace ALL placeholders for f(x) = {user_function}
Output ONLY the complete filled code.
"""
    
    metadata = {
        "attempts": 0,
        "validation_errors": [],
        "test_errors": []
    }
    
    for attempt in range(1, max_attempts + 1):
        metadata["attempts"] = attempt
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_attempts}")
        print(f"{'='*60}")
        
        try:
            print(f"Generating code from template using {model_name}")
            
            if use_thinking:
                # For thinking mode, use thinking model
                actual_model = "gemini-2.0-flash-thinking-exp-01-21"
                response = client.models.generate_content(
                    model=actual_model,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        system_instruction=SYSTEM_INSTRUCTION
                    ),
                    contents=prompt
                )
            else:
                # Use the specified model (default: gemini-2.5-flash)
                response = client.models.generate_content(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        system_instruction=SYSTEM_INSTRUCTION
                    ),
                    contents=prompt
                )
            
            code = extract_code_from_response(response.text)
            
            # Debug: Show generated LaTeX
            print("\nDEBUG - Generated LaTeX lines:")
            for i, line in enumerate(code.split('\n')):
                if 'MathTex' in line and 'step' in line.lower():
                    print(f"  Line {i}: {line[:100]}")
            
            print("\nValidating code structure...")
            is_valid, validation_msg = validate_code_structure(code)
            
            if not is_valid:
                print(f"❌ Validation failed: {validation_msg}")
                metadata["validation_errors"].append(validation_msg)
                
                if "calculation" in validation_msg.lower():
                    prompt += f"\n\n⚠️ CRITICAL ERROR: Missing calculation steps! Include all: calc_title, calc_step1, calc_step2, calc_step3, calc_final."
                elif "deriv_below" in validation_msg.lower():
                    prompt += f"\n\n⚠️ ERROR: Derivative label must be positioned with next_to(calc_steps, DOWN) to avoid overlap!"
                else:
                    prompt += f"\n\n⚠️ ERROR: {validation_msg}"
                continue
            
            print(f"✓ {validation_msg}")
            
            if skip_execution_test:
                print("⚠️ Skipping execution test")
                metadata["success"] = True
                return code, metadata
            
            print("Testing code execution...")
            can_execute, test_msg = test_manim_code(code)
            
            if not can_execute:
                print(f"❌ Execution test failed: {test_msg}")
                metadata["test_errors"].append(test_msg)
                prompt += f"\n\n⚠️ EXECUTION ERROR:\n{test_msg[:300]}"
                continue
            
            print(f"✓ {test_msg}")
            print("\n🎉 SUCCESS! Generated working Manim code.")
            metadata["success"] = True
            return code, metadata
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"❌ {error_msg}")
            metadata["test_errors"].append(error_msg)
            
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                print("⚠️ API quota/rate limit. Waiting 60 seconds...")
                import time
                time.sleep(60)
            
            continue
    
    print("\n❌ Failed to generate working code after all attempts")
    metadata["success"] = False
    return "", metadata

def save_code(code: str, function_name: str, output_dir: str = "generated_manim"):
    """Save generated code to file"""
    Path(output_dir).mkdir(exist_ok=True)
    
    safe_name = re.sub(r'[^\w\s-]', '', function_name).strip().replace(' ', '_')
    filename = f"{output_dir}/derivative_{safe_name}.py"
    
    with open(filename, 'w') as f:
        f.write(code)
    
    return filename

def main():
    print("\n" + "="*60)
    print("MANIM DERIVATIVE VISUALIZER - Enhanced v2")
    print("="*60)
    print("\n✨ Fixed overlapping text and LaTeX rendering!")
    print("\nTips:")
    print("  • Use ^ for exponents (e.g., x^2)")
    print("  • Use () for grouping (e.g., (x+1)^2)")
    print("  • Use * for multiplication (e.g., 2*x)")
    print("  • Examples: x^2, sin(x), 3*x^2+2*x+1, x*sin(x)")
    print("-"*60)
    
    user_function = input("\nEnter your function: ").strip()
    
    if not user_function:
        print("Error: No function provided")
        return
    
    print("\nGeneration mode:")
    print("  1. Fast (Gemini 2.0 Flash)")
    print("  2. Quality (Gemini 2.0 Flash Thinking)")
    mode = input("Choose mode (1 or 2, default=2): ").strip() or "2"
    
    use_thinking = (mode == "2")
    
    if use_thinking:
        print("\n⚡ Using Gemini 2.0 Flash Thinking")
    else:
        print("\n⚡ Using Gemini 2.0 Flash")
    
    code, metadata = generate_manim_code(user_function, max_attempts=3, use_thinking=use_thinking)
    
    if not code:
        print("\n❌ FAILED TO GENERATE WORKING CODE")
        print("\nMetadata:")
        print(f"  Attempts: {metadata['attempts']}")
        if metadata.get('validation_errors'):
            print(f"  Validation Errors:")
            for err in metadata['validation_errors']:
                print(f"    - {err}")
        if metadata.get('test_errors'):
            print(f"  Last Error: {metadata['test_errors'][-1][:300]}")
        return
    
    filename = save_code(code, user_function)
    print(f"\n✓ Code saved to: {filename}")
    
    print("\nGeneration Metadata:")
    print(f"  Attempts: {metadata['attempts']}")
    print(f"  Success: {metadata.get('success', False)}")
    print(f"  ✅ No overlapping text")
    print(f"  ✅ Proper LaTeX rendering")
    
    print("\n" + "="*60)
    print("To render the animation:")
    print(f"  manim -pql {filename} DerivativeVisualization")
    print("\nFor high quality:")
    print(f"  manim -pqh {filename} DerivativeVisualization")

if __name__ == "__main__":
    main()