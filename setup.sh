#!/bin/bash

# DerivativeAnimator Setup Script
# Automates installation and environment setup

echo "======================================================================"
echo "DerivativeAnimator - Complete Setup"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python $python_version detected"
else
    echo "❌ Python 3.8+ required. Current: $python_version"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  venv already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ All dependencies installed"

# Install system dependencies for Manim (if needed)
echo ""
echo "Checking system dependencies..."
if command -v apt-get &> /dev/null; then
    echo "Detected apt package manager"
    echo "Installing system dependencies (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg libcairo2-dev pkg-config python3-dev
    echo "✓ System dependencies installed"
elif command -v brew &> /dev/null; then
    echo "Detected Homebrew"
    echo "Installing system dependencies..."
    brew install ffmpeg cairo pkg-config
    echo "✓ System dependencies installed"
else
    echo "⚠️  Could not detect package manager"
    echo "Please manually install: ffmpeg, cairo, pkg-config"
fi

# Create .env template
echo ""
echo "Creating .env template..."
if [ -f ".env" ]; then
    echo "⚠️  .env already exists. Skipping."
else
    cat > .env << EOF
# DerivativeAnimator Environment Variables
GEMINI_API_KEY=your_gemini_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
EOF
    echo "✓ .env template created"
    echo "⚠️  Please edit .env and add your API keys"
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p derivative_dataset_537/{code,metadata,finetuning,evaluation,visualizations}
mkdir -p derivative_dataset_537/code/{foundation,conceptual,application,advanced}
echo "✓ Directory structure created"

# Test installation
echo ""
echo "Testing installation..."
python3 -c "import manim; import torch; import transformers; print('✓ Core packages imported successfully')"

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Generate dataset:"
echo "   python data_generation_pipeline.py"
echo ""
echo "3. Create visualizations:"
echo "   python dataset_visualizer.py"
echo ""
echo "4. Prepare fine-tuning data:"
echo "   python finetuning_data_preparation.py"
echo ""
echo "5. Launch web UI:"
echo "   python gradio_ui.py"
echo ""
echo "6. Fine-tune model (optional):"
echo "   python finetuning_train.py"
echo ""
echo "7. Evaluate results:"
echo "   python evaluation_framework.py"
echo ""
echo "======================================================================"
echo "✅ Ready to go!"
echo "======================================================================"