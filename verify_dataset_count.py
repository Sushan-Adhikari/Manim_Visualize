#!/usr/bin/env python3
"""
Verify the dataset has exactly 537 functions
"""

# Copy the DATASET_FUNCTIONS from data_generation_pipeline.py
from data_generation_pipeline import DATASET_FUNCTIONS

def verify_dataset():
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70 + "\n")
    
    # Count by level
    counts = {}
    for level, functions in DATASET_FUNCTIONS.items():
        counts[level] = len(functions)
    
    # Print breakdown
    print("📊 Function Count by Level:")
    print()
    for level, count in counts.items():
        print(f"  {level.capitalize():15} {count:3} functions")
    
    total = sum(counts.values())
    print(f"  {'-'*24}")
    print(f"  {'TOTAL':15} {total:3} functions")
    print()
    
    # Verify
    if total == 537:
        print("✅ VERIFIED: Dataset contains exactly 537 functions!")
    else:
        print(f"❌ ERROR: Expected 537, but found {total} functions")
        print(f"   Difference: {total - 537:+d}")
        
        if total < 537:
            print(f"\n   Need to add {537 - total} more functions")
        else:
            print(f"\n   Need to remove {total - 537} functions")
    
    print()
    
    # Check for duplicates
    print("🔍 Checking for duplicate functions...")
    all_functions = []
    for level, functions in DATASET_FUNCTIONS.items():
        for func, desc in functions:
            all_functions.append(func)
    
    unique_functions = set(all_functions)
    duplicates = len(all_functions) - len(unique_functions)
    
    if duplicates == 0:
        print("   ✅ No duplicates found")
    else:
        print(f"   ⚠️  Found {duplicates} duplicate(s)")
        # Find and print duplicates
        from collections import Counter
        counts = Counter(all_functions)
        dupes = [func for func, count in counts.items() if count > 1]
        for func in dupes[:5]:  # Show first 5
            print(f"      - {func} (appears {counts[func]} times)")
    
    print()
    
    # Sample functions from each level
    print("📝 Sample Functions:")
    for level, functions in DATASET_FUNCTIONS.items():
        print(f"\n  {level.capitalize()}:")
        for i, (func, desc) in enumerate(functions[:3], 1):  # Show first 3
            print(f"    {i}. f(x) = {func:20} ({desc})")
        if len(functions) > 3:
            print(f"    ... and {len(functions) - 3} more")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70 + "\n")
    
    return total == 537 and duplicates == 0

if __name__ == "__main__":
    success = verify_dataset()
    
    if success:
        print("✅ Dataset is ready for generation!")
        print("\nRun: python data_generation_pipeline.py")
    else:
        print("❌ Please fix the issues above before generating")