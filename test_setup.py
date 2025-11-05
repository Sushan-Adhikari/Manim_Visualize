"""
Quick test to verify the training setup is correct
"""

import json
from pathlib import Path
import torch

def test_dataset_files():
    """Test that all required dataset files exist"""
    print("\n" + "="*70)
    print("Testing Dataset Files")
    print("="*70)

    required_files = [
        'derivative_dataset_537/finetuning/hf_train.jsonl',
        'derivative_dataset_537/finetuning/hf_validation.jsonl',
        'derivative_dataset_537/finetuning/hf_test.jsonl'
    ]

    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size / 1024 / 1024
            print(f"✓ {file} ({size:.2f} MB)")

            # Count lines
            with open(file) as f:
                count = sum(1 for _ in f)
            print(f"  → {count} samples")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_exist = False

    return all_exist

def test_dataset_format():
    """Test that dataset is in correct format"""
    print("\n" + "="*70)
    print("Testing Dataset Format")
    print("="*70)

    try:
        with open('derivative_dataset_537/finetuning/hf_train.jsonl') as f:
            first_line = f.readline()
            sample = json.loads(first_line)

        required_keys = ['instruction', 'input', 'output']
        has_all_keys = all(key in sample for key in required_keys)

        if has_all_keys:
            print("✓ Dataset has all required keys")
            print(f"\n📝 Sample:")
            print(f"   Instruction: {sample['instruction'][:60]}...")
            print(f"   Input: {sample['input']}")
            print(f"   Output length: {len(sample['output'])} characters")
            return True
        else:
            print(f"❌ Missing keys. Found: {list(sample.keys())}")
            return False

    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*70)
    print("Testing GPU")
    print("="*70)

    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  → VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"  → Compute Capability: {props.major}.{props.minor}")
        return True
    else:
        print("⚠️  No GPU detected")
        print("   Training will be VERY slow on CPU")
        print("   Recommend using a GPU (Google Colab, AWS, etc.)")
        return False

def test_dependencies():
    """Test that all required packages are installed"""
    print("\n" + "="*70)
    print("Testing Dependencies")
    print("="*70)

    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('peft', 'peft'),
        ('bitsandbytes', 'bitsandbytes'),
    ]

    all_installed = True
    for package_name, import_name in required_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: NOT INSTALLED")
            all_installed = False

    return all_installed

def test_training_script():
    """Test that training script can be imported"""
    print("\n" + "="*70)
    print("Testing Training Script")
    print("="*70)

    try:
        # Try to import the training script
        import finetuning_train
        print("✓ finetuning_train.py can be imported")
        return True
    except Exception as e:
        print(f"❌ Error importing training script: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("DERIVATIVE ANIMATOR - SETUP VERIFICATION")
    print("="*70)
    print("\nThis script will verify that everything is ready for training.")

    results = []

    # Run all tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("GPU", test_gpu()))
    results.append(("Dataset Files", test_dataset_files()))
    results.append(("Dataset Format", test_dataset_format()))
    results.append(("Training Script", test_training_script()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed and test_name != "GPU":  # GPU is optional
            all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou're ready to train the model:")
        print("   python3 run_training.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before training.")
        print("\nIf dataset files are missing, run:")
        print("   python3 prepare_training_data.py")
        print("\nIf dependencies are missing, run:")
        print("   pip install -r requirements.txt")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()
