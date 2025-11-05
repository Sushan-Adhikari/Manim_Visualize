"""
Evaluate fine-tuned DerivativeAnimator model on test set
"""

import json
from pathlib import Path
from inference_finetuned import DerivativeAnimatorInference
import time

def load_test_cases():
    """Load test cases from dataset"""
    with open('derivative_dataset_537/finetuning/instruction_dataset.json') as f:
        data = json.load(f)

    test_cases = []
    for item in data['test'][:20]:  # Test on first 20 examples
        instruction = item['instruction']
        func = instruction.split('f(x) = ')[-1] if 'f(x) = ' in instruction else "unknown"
        test_cases.append({
            'function': func,
            'reference_code': item['output'],
            'instruction': instruction
        })

    return test_cases

def evaluate():
    """Run evaluation on test set"""
    print("\n" + "="*70)
    print("DERIVATIVE ANIMATOR - MODEL EVALUATION")
    print("="*70)

    # Load model
    print("\n🤖 Loading fine-tuned model...")
    try:
        generator = DerivativeAnimatorInference()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease train the model first:")
        print("   python3 run_training.py")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return

    # Load test cases
    print("\n📚 Loading test cases...")
    test_cases = load_test_cases()
    print(f"✓ Loaded {len(test_cases)} test cases")

    # Run evaluation
    results = {
        'total': len(test_cases),
        'valid': 0,
        'executable': 0,
        'failed': 0,
        'errors': []
    }

    print("\n" + "="*70)
    print("Running evaluation...")
    print("="*70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: f(x) = {test['function']}")
        print("-"*70)

        try:
            # Generate code
            start_time = time.time()
            code = generator.generate_code(
                function=test['function'],
                temperature=0.2,
                top_p=0.9,
                do_sample=True
            )
            gen_time = time.time() - start_time

            # Validate structure
            is_valid, msg = generator.validate_code(code)
            if not is_valid:
                print(f"❌ Validation failed: {msg}")
                results['failed'] += 1
                results['errors'].append({
                    'function': test['function'],
                    'error': f"Validation: {msg}"
                })
                continue

            print(f"✓ Validation passed")
            results['valid'] += 1

            # Test execution
            can_execute, test_msg = generator.test_execution(code, timeout=45)
            if not can_execute:
                print(f"❌ Execution failed: {test_msg[:200]}")
                results['errors'].append({
                    'function': test['function'],
                    'error': f"Execution: {test_msg[:200]}"
                })
            else:
                print(f"✓ Execution passed")
                results['executable'] += 1

            # Save result
            filename = generator.save_code(code, f"test_{i}_{test['function']}", "evaluation_results")
            print(f"✓ Saved to: {filename}")
            print(f"⏱️  Generation time: {gen_time:.2f}s")

        except Exception as e:
            print(f"❌ Error: {str(e)[:200]}")
            results['failed'] += 1
            results['errors'].append({
                'function': test['function'],
                'error': str(e)[:200]
            })

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Total cases: {results['total']}")
    print(f"   Valid structure: {results['valid']} ({results['valid']/results['total']*100:.1f}%)")
    print(f"   Executable: {results['executable']} ({results['executable']/results['total']*100:.1f}%)")
    print(f"   Failed: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")

    # Success rate for the research paper
    success_rate = results['executable'] / results['total'] * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("   ✅ EXCELLENT - Model performs well!")
    elif success_rate >= 60:
        print("   ✓ GOOD - Model is functional with room for improvement")
    elif success_rate >= 40:
        print("   ⚠️  FAIR - Model needs more training or data")
    else:
        print("   ❌ POOR - Model needs significant improvement")

    # Save results
    results_file = "evaluation_results/summary.json"
    Path("evaluation_results").mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {results_file}")

    if results['errors']:
        print(f"\n⚠️  Errors encountered:")
        for err in results['errors'][:5]:  # Show first 5 errors
            print(f"   - {err['function']}: {err['error'][:100]}")

    print("\n" + "="*70)

if __name__ == "__main__":
    evaluate()
