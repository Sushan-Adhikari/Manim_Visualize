"""
Complete training pipeline for DerivativeAnimator
Runs all steps needed to train and test the model
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False

    print(f"\n✅ Completed: {description}")
    return True

def main():
    print("\n" + "="*70)
    print("DERIVATIVE ANIMATOR - COMPLETE TRAINING PIPELINE")
    print("="*70)

    # Step 1: Prepare training data
    if not run_command(
        "python3 prepare_training_data.py",
        "Step 1: Preparing training data (JSONL format)"
    ):
        return

    # Step 2: Verify dataset
    dataset_files = [
        "derivative_dataset_537/finetuning/hf_train.jsonl",
        "derivative_dataset_537/finetuning/hf_validation.jsonl"
    ]

    print("\n📊 Verifying dataset files...")
    for file in dataset_files:
        if not Path(file).exists():
            print(f"❌ Missing: {file}")
            return
        size = Path(file).stat().st_size / 1024 / 1024
        print(f"✓ {file} ({size:.2f} MB)")

    # Step 3: Check GPU
    print("\n🔍 Checking GPU availability...")
    result = subprocess.run(
        "python3 -c 'import torch; print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout.strip())

    if "None" in result.stdout:
        print("⚠️  No GPU detected. Training will be very slow.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return

    # Step 4: Train model
    print("\n" + "="*70)
    print("Starting fine-tuning...")
    print("This will take approximately 1-3 hours depending on your GPU")
    print("="*70)

    proceed = input("\nStart training now? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Training cancelled")
        return

    if not run_command(
        "python3 finetuning_train.py",
        "Step 4: Fine-tuning DeepSeek-Coder-1.3B"
    ):
        return

    # Step 5: Test inference
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test the model:")
    print("   python3 inference_finetuned.py")
    print("\n2. Run batch evaluation:")
    print("   python3 evaluate_model.py")
    print("\n3. Compare with baseline:")
    print("   python3 compare_models.py")

if __name__ == "__main__":
    main()
