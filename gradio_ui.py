import gradio as gr
import os
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Import your generator
from manim_generator import generate_manim_code

# Load environment
load_dotenv()

# Pre-defined function templates
FUNCTION_TEMPLATES = {
    "Polynomials": {
        "Linear (x)": "x",
        "Quadratic (x²)": "x^2",
        "Cubic (x³)": "x^3",
        "Quartic (x⁴)": "x^4",
        "Custom Polynomial": ""
    },
    "Trigonometric": {
        "Sine": "sin(x)",
        "Cosine": "cos(x)",
        "Tangent": "tan(x)",
        "Scaled Sine": "2*sin(x)",
        "Custom Trig": ""
    },
    "Exponential & Logarithmic": {
        "e^x": "e^x",
        "Natural Log": "ln(x)",
        "Scaled Exponential": "2*e^x",
        "Custom Exp/Log": ""
    },
    "Advanced": {
        "x·sin(x)": "x*sin(x)",
        "x²·e^x": "x^2*e^x",
        "sin(x²)": "sin(x^2)",
        "Custom Advanced": ""
    }
}

def get_function_buttons():
    """Generate buttons for each function template"""
    buttons_html = []
    for category, functions in FUNCTION_TEMPLATES.items():
        buttons_html.append(f"### {category}")
        for name, func in functions.items():
            if func:  # Skip custom entries
                buttons_html.append(f"- **{name}**: `{func}`")
    return "\n".join(buttons_html)

def generate_animation(function_input: str, 
                      use_thinking: bool,
                      quality: str) -> tuple:
    """Generate Manim animation from function"""
    
    if not function_input.strip():
        return None, "❌ Please enter a function!", None
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        return None, "❌ GEMINI_API_KEY not found in environment!", None
    
    try:
        status = f"🔄 Generating code for f(x) = {function_input}..."
        
        # Generate code
        code, metadata = generate_manim_code(
            function_input,
            max_attempts=3,
            use_thinking=use_thinking,
            skip_execution_test=False
        )
        
        if not code:
            error_msg = "❌ Failed to generate code\n\n"
            if metadata.get('validation_errors'):
                error_msg += "Validation Errors:\n" + "\n".join(metadata['validation_errors'])
            if metadata.get('test_errors'):
                error_msg += "\n\nLast Error:\n" + metadata['test_errors'][-1][:500]
            return None, error_msg, None
        
        status += f"\n✓ Code generated successfully ({metadata['attempts']} attempts)"
        
        # Save code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        status += f"\n🎬 Rendering animation ({quality} quality)..."
        
        # Render animation
        quality_flag = {
            "Preview (Low)": "-ql",
            "Medium": "-qm",
            "High": "-qh"
        }[quality]
        
        result = subprocess.run(
            ['manim', quality_flag, temp_file, 'DerivativeVisualization'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return None, f"❌ Rendering failed:\n{result.stderr[:500]}", code
        
        # Find generated video
        media_dir = Path("media/videos")
        video_files = list(media_dir.rglob("*.mp4"))
        
        if not video_files:
            return None, "❌ Video file not found after rendering", code
        
        # Get most recent video
        latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
        
        status += f"\n✅ Animation rendered successfully!"
        status += f"\n📊 Metadata: {metadata['attempts']} attempts"
        
        # Cleanup
        try:
            os.unlink(temp_file)
        except:
            pass
        
        return str(latest_video), status, code
        
    except subprocess.TimeoutExpired:
        return None, "❌ Rendering timeout (>60s)", None
    except Exception as e:
        return None, f"❌ Error: {str(e)}", None

def select_template(category: str, template_name: str) -> str:
    """Return function from template selection"""
    if category in FUNCTION_TEMPLATES and template_name in FUNCTION_TEMPLATES[category]:
        return FUNCTION_TEMPLATES[category][template_name]
    return ""

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
.function-input textarea {
    font-family: 'Courier New', monospace;
    font-size: 16px;
}
.title-text {
    text-align: center;
    color: #2c3e50;
}
.example-box {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="DerivativeAnimator") as demo:
    
    gr.Markdown("""
    # 🎬 DerivativeAnimator
    ### Automated Manim Code Generation for Calculus Derivatives
    
    Generate professional mathematical animations instantly - no programming required!
    """, elem_classes="title-text")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📝 Function Input")
            
            # Template selector
            gr.Markdown("### Quick Templates")
            category_dropdown = gr.Dropdown(
                choices=list(FUNCTION_TEMPLATES.keys()),
                label="Category",
                value="Polynomials"
            )
            template_dropdown = gr.Dropdown(
                choices=list(FUNCTION_TEMPLATES["Polynomials"].keys()),
                label="Template",
                value="Quadratic (x²)"
            )
            load_template_btn = gr.Button("📥 Load Template", variant="secondary")
            
            gr.Markdown("---")
            
            # Function input
            function_input = gr.Textbox(
                label="Mathematical Function",
                placeholder="Enter your function (e.g., x^2, sin(x), x*e^x)",
                value="x^2",
                lines=2,
                elem_classes="function-input"
            )
            
            gr.Markdown("""
            **Syntax Guide:**
            - Powers: `x^2`, `x^3`
            - Trig: `sin(x)`, `cos(x)`, `tan(x)`
            - Exponential: `e^x`, `2*e^x`
            - Logarithm: `ln(x)`, `log(x)`
            - Products: `x*sin(x)`, `x^2*e^x`
            """, elem_classes="example-box")
            
            # Options
            gr.Markdown("### ⚙️ Generation Options")
            use_thinking = gr.Checkbox(
                label="Use Thinking Mode (slower, higher quality)",
                value=True
            )
            quality = gr.Radio(
                choices=["Preview (Low)", "Medium", "High"],
                label="Video Quality",
                value="Preview (Low)"
            )
            
            # Generate button
            generate_btn = gr.Button("🚀 Generate Animation", variant="primary", size="lg")
            
            # Examples
            gr.Markdown("### 📚 Examples")
            gr.Examples(
                examples=[
                    ["x^2"],
                    ["x^3"],
                    ["sin(x)"],
                    ["e^x"],
                    ["x*sin(x)"],
                    ["x^2*e^x"]
                ],
                inputs=function_input,
                label="Click to try"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("## 🎥 Output")
            
            # Video output
            video_output = gr.Video(
                label="Generated Animation",
                autoplay=True
            )
            
            # Status
            status_output = gr.Textbox(
                label="Generation Status",
                lines=5,
                interactive=False
            )
            
            # Code output
            with gr.Accordion("📄 Generated Code", open=False):
                code_output = gr.Code(
                    label="Manim Code",
                    language="python",
                    lines=20
                )
    
    # Event handlers
    def update_templates(category):
        templates = list(FUNCTION_TEMPLATES[category].keys())
        return gr.Dropdown(choices=templates, value=templates[0])
    
    category_dropdown.change(
        update_templates,
        inputs=[category_dropdown],
        outputs=[template_dropdown]
    )
    
    load_template_btn.click(
        select_template,
        inputs=[category_dropdown, template_dropdown],
        outputs=[function_input]
    )
    
    generate_btn.click(
        generate_animation,
        inputs=[function_input, use_thinking, quality],
        outputs=[video_output, status_output, code_output]
    )
    
    # Footer
    gr.Markdown("""
    ---
    **DerivativeAnimator** - Making mathematical visualization accessible to everyone
    
    *Research by: Sushan Adhikari, Sunidhi Sharma, Darshan Lamichhane, Usan Adhikari*
    
    Kathmandu University | 2025
    """, elem_classes="title-text")

# Launch instructions
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DERIVATIVEANIMATOR WEB UI")
    print("="*70)
    print("\n✨ Features:")
    print("  • Template-based function selection")
    print("  • Real-time animation generation")
    print("  • Multiple quality options")
    print("  • Code export functionality")
    
    # Check dependencies
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  WARNING: GEMINI_API_KEY not found!")
        print("Please set it in your .env file")
    
    print("\n🚀 Launching interface...")
    print("="*70 + "\n")
    
    demo.launch(
        share=False,  # Set to True to create public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )