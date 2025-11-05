import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import re

class FineTuningDataPreparation:
    """Prepare dataset for LLM fine-tuning (OpenAI, HuggingFace formats)"""
    
    def __init__(self, dataset_dir: str = "derivative_dataset_537"):
        self.dataset_dir = Path(dataset_dir)
        self.code_dir = self.dataset_dir / "code"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.finetuning_dir = self.dataset_dir / "finetuning"
        self.finetuning_dir.mkdir(exist_ok=True)
        
    def prepare_all_formats(self, 
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1):
        """Prepare data in multiple formats for different platforms"""
        
        print("\n" + "="*70)
        print("FINE-TUNING DATA PREPARATION")
        print("="*70 + "\n")
        
        # Collect all successful generations
        data_samples = self._collect_successful_samples()
        print(f"✓ Collected {len(data_samples)} successful samples")
        
        if len(data_samples) == 0:
            print("\n❌ No successful samples found!")
            print("Please ensure data_generation_pipeline.py completed successfully.")
            print(f"Check if files exist in: {self.code_dir}")
            return
        
        # Split data
        train, val, test = self._split_data(data_samples, train_ratio, val_ratio, test_ratio)
        print(f"✓ Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        # Generate different formats
        self._generate_openai_format(train, val, test)
        self._generate_huggingface_format(train, val, test)
        self._generate_jsonl_format(train, val, test)
        self._generate_instruction_format(train, val, test)
        
        # Generate statistics
        self._generate_dataset_info(train, val, test)
        
        # Generate quality report
        self._generate_quality_report(data_samples)
        
        print("\n✅ Fine-tuning data preparation complete!")
        print(f"📁 Output: {self.finetuning_dir}")
    
    def _collect_successful_samples(self) -> List[Dict]:
        """Collect all successful code generations"""
        samples = []
        
        # Load generation report to get successful samples
        report_file = self.metadata_dir / "generation_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Get successful samples
            for result in report['results']:
                if result['success'] and 'code_file' in result:
                    code_file = self.dataset_dir / result['code_file']
                    if code_file.exists():
                        with open(code_file, 'r') as f:
                            code = f.read()
                        
                        samples.append({
                            'function': result['function'],
                            'code': code,
                            'level': result['level'],
                            'description': result.get('description', ''),
                            'filename': code_file.name,
                            'attempts': result.get('attempts', 1),
                            'code_length': result.get('code_length', len(code))
                        })
        else:
            # Fallback: scan directories
            print("⚠️  No report found, scanning directories...")
            for level in ['foundation', 'conceptual', 'application', 'advanced']:
                level_dir = self.code_dir / level
                if not level_dir.exists():
                    continue
                
                for code_file in level_dir.glob("*.py"):
                    with open(code_file, 'r') as f:
                        code = f.read()
                    
                    # Extract function from code
                    function = self._extract_function_from_code(code)
                    if not function:
                        continue
                    
                    samples.append({
                        'function': function,
                        'code': code,
                        'level': level,
                        'description': '',
                        'filename': code_file.name,
                        'attempts': 1,
                        'code_length': len(code)
                    })
        
        print(f"  Found samples in:")
        for level in ['foundation', 'conceptual', 'application', 'advanced']:
            count = len([s for s in samples if s['level'] == level])
            if count > 0:
                print(f"    • {level.capitalize()}: {count}")
        
        return samples
    
    def _extract_function_from_code(self, code: str) -> str:
        """Extract the mathematical function from generated code"""
        patterns = [
            r'def f\(x\):\s*\n\s*return (.+)',
            r'f = lambda x:\s*(.+)',
            r'function.*?=.*?lambda x:\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code, re.MULTILINE)
            if match:
                python_expr = match.group(1).strip()
                python_expr = re.sub(r'\s*#.*$', '', python_expr)
                
                # Convert Python to math notation
                math_expr = python_expr
                math_expr = math_expr.replace('**', '^')
                math_expr = math_expr.replace('np.sin', 'sin')
                math_expr = math_expr.replace('np.cos', 'cos')
                math_expr = math_expr.replace('np.tan', 'tan')
                math_expr = math_expr.replace('np.exp', 'exp')
                math_expr = math_expr.replace('np.log', 'ln')
                math_expr = math_expr.replace('np.sqrt', 'sqrt')
                return math_expr
    
        return None
        
    def _split_data(self, data: List[Dict], 
                   train_ratio: float, 
                   val_ratio: float, 
                   test_ratio: float) -> Tuple[List, List, List]:
        """Split data into train/val/test sets with stratification by level"""
        random.seed(42)
        
        # Stratified split by level
        train, val, test = [], [], []
        
        for level in ['foundation', 'conceptual', 'application', 'advanced']:
            level_data = [s for s in data if s['level'] == level]
            random.shuffle(level_data)
            
            n = len(level_data)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio)
            
            train.extend(level_data[:train_size])
            val.extend(level_data[train_size:train_size + val_size])
            test.extend(level_data[train_size + val_size:])
        
        # Shuffle combined sets
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return train, val, test
    
    def _generate_openai_format(self, train: List, val: List, test: List):
        """Generate OpenAI fine-tuning format (JSONL)"""
        print("\n📝 Generating OpenAI format...")
        
        def create_openai_sample(sample: Dict) -> Dict:
            system_prompt = """You are an expert Manim code generator specializing in mathematical derivative visualizations. 
You generate complete, syntactically correct, and executable Manim code that:
1. Visualizes the function graph
2. Shows step-by-step derivative calculation
3. Demonstrates the derivative with a moving tangent line
4. Uses proper mathematical notation in LaTeX
5. Follows the established template structure"""
            user_prompt = f"Generate complete Manim code to visualize the derivative of: f(x) = {sample['function']}"
            
            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": sample['code']}
                ]
            }
        
        # Save train
        train_file = self.finetuning_dir / "openai_train.jsonl"
        with open(train_file, 'w') as f:
            for sample in train:
                f.write(json.dumps(create_openai_sample(sample)) + '\n')
        
        # Save val
        val_file = self.finetuning_dir / "openai_val.jsonl"
        with open(val_file, 'w') as f:
            for sample in val:
                f.write(json.dumps(create_openai_sample(sample)) + '\n')
        
        print(f"  ✓ {train_file.name} ({len(train)} samples)")
        print(f"  ✓ {val_file.name} ({len(val)} samples)")
    
    def _generate_huggingface_format(self, train: List, val: List, test: List):
        """Generate HuggingFace datasets format - FIXED"""
        print("\n📝 Generating HuggingFace format...")
        
        def create_hf_sample(sample: Dict) -> Dict:
            instruction = f"""Below is a mathematical function. Generate complete, executable Manim code to visualize its derivative.
Function: f(x) = {sample['function']}
Level: {sample['level']}
Requirements:
- Show the function graph
- Display derivative calculation steps
- Animate a moving tangent line
- Use proper LaTeX formatting"""
            return {
                "instruction": instruction,
                "input": sample['function'],
                "output": sample['code'],
                "level": sample['level']
            }
        
        # ✅ FIX: Save as separate JSONL files instead of nested JSON
        # This is the correct format for HuggingFace load_dataset('json')
        
        train_file = self.finetuning_dir / "hf_train.jsonl"
        with open(train_file, 'w') as f:
            for sample in train:
                f.write(json.dumps(create_hf_sample(sample)) + '\n')
        
        val_file = self.finetuning_dir / "hf_validation.jsonl"
        with open(val_file, 'w') as f:
            for sample in val:
                f.write(json.dumps(create_hf_sample(sample)) + '\n')
        
        test_file = self.finetuning_dir / "hf_test.jsonl"
        with open(test_file, 'w') as f:
            for sample in test:
                f.write(json.dumps(create_hf_sample(sample)) + '\n')
        
        print(f"  ✓ {train_file.name} ({len(train)} samples)")
        print(f"  ✓ {val_file.name} ({len(val)} samples)")
        print(f"  ✓ {test_file.name} ({len(test)} samples)")
        
        # Also create a combined JSON file for reference (but not for direct loading)
        combined_file = self.finetuning_dir / "huggingface_dataset_combined.json"
        with open(combined_file, 'w') as f:
            json.dump({
                "train": [create_hf_sample(s) for s in train],
                "validation": [create_hf_sample(s) for s in val],
                "test": [create_hf_sample(s) for s in test]
            }, f, indent=2)
        print(f"  ✓ {combined_file.name} (reference only)")
    
    def _generate_jsonl_format(self, train: List, val: List, test: List):
        """Generate simple JSONL format for general use"""
        print("\n📝 Generating JSONL format...")
        
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            jsonl_file = self.finetuning_dir / f"{split_name}.jsonl"
            with open(jsonl_file, 'w') as f:
                for sample in split_data:
                    f.write(json.dumps({
                        "function": sample['function'],
                        "code": sample['code'],
                        "level": sample['level'],
                        "description": sample.get('description', ''),
                        "metadata": {
                            "attempts": sample.get('attempts', 1),
                            "code_length": sample.get('code_length', len(sample['code']))
                        }
                    }) + '\n')
            print(f"  ✓ {jsonl_file.name} ({len(split_data)} samples)")
    
    def _generate_instruction_format(self, train: List, val: List, test: List):
        """Generate instruction-tuning format (Alpaca/Vicuna style)"""
        print("\n📝 Generating instruction-tuning format...")
        
        instructions = [
            "Create a Manim animation showing the derivative calculation for the function",
            "Generate Manim code to visualize the derivative of",
            "Write a complete Manim scene that demonstrates the derivative of",
            "Produce a derivative visualization animation for the function",
            "Develop Manim code showing derivative calculation steps for",
            "Build an animated derivative visualization for",
            "Construct Manim code that illustrates the derivative of"
        ]
        
        def create_instruction_sample(sample: Dict) -> Dict:
            instruction = random.choice(instructions)
            return {
                "instruction": f"{instruction}: f(x) = {sample['function']}",
                "input": "",
                "output": sample['code'],
                "metadata": {
                    "level": sample['level'],
                    "function": sample['function'],
                    "description": sample.get('description', ''),
                    "attempts": sample.get('attempts', 1)
                }
            }
        
        instruction_data = {
            "train": [create_instruction_sample(s) for s in train],
            "validation": [create_instruction_sample(s) for s in val],
            "test": [create_instruction_sample(s) for s in test]
        }
        
        instruction_file = self.finetuning_dir / "instruction_dataset.json"
        with open(instruction_file, 'w') as f:
            json.dump(instruction_data, f, indent=2)
        
        print(f"  ✓ {instruction_file.name}")
    
    def _generate_quality_report(self, samples: List[Dict]):
        """Generate data quality report"""
        print("\n📝 Generating quality report...")
        
        total = len(samples)
        by_level = {level: len([s for s in samples if s['level'] == level]) 
                   for level in ['foundation', 'conceptual', 'application', 'advanced']}
        
        avg_code_length = sum(s['code_length'] for s in samples) / total
        avg_attempts = sum(s.get('attempts', 1) for s in samples) / total
        
        function_types = {}
        for sample in samples:
            func = sample['function'].lower()
            if 'sin' in func or 'cos' in func:
                ftype = 'trigonometric'
            elif 'e^' in func or 'exp' in func:
                ftype = 'exponential'
            elif 'ln' in func or 'log' in func:
                ftype = 'logarithmic'
            elif '^' in func:
                ftype = 'polynomial'
            else:
                ftype = 'linear'
            
            function_types[ftype] = function_types.get(ftype, 0) + 1
        
        quality_report = {
            "total_samples": total,
            "distribution_by_level": by_level,
            "distribution_by_type": function_types,
            "average_code_length": int(avg_code_length),
            "average_attempts": round(avg_attempts, 2),
            "quality_metrics": {
                "all_samples_valid": all('code' in s and len(s['code']) > 0 for s in samples),
                "all_have_functions": all('function' in s for s in samples),
                "code_length_range": [
                    min(s['code_length'] for s in samples),
                    max(s['code_length'] for s in samples)
                ]
            }
        }
        
        report_file = self.finetuning_dir / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        print(f"  ✓ {report_file.name}")
    
    def _generate_dataset_info(self, train: List, val: List, test: List):
        """Generate dataset information and statistics"""
        print("\n📝 Generating dataset info...")
        
        total = len(train) + len(val) + len(test)
        
        info = {
            "dataset_name": "DerivativeAnimator-Manim-537",
            "version": "1.0",
            "description": "Fine-tuning dataset for automated Manim code generation for derivative visualizations",
            "total_samples": total,
            "splits": {
                "train": len(train),
                "validation": len(val),
                "test": len(test)
            },
            "splits_percentage": {
                "train": f"{len(train)/total*100:.1f}%",
                "validation": f"{len(val)/total*100:.1f}%",
                "test": f"{len(test)/total*100:.1f}%"
            },
            "by_level": {
                level: {
                    "train": len([s for s in train if s['level'] == level]),
                    "val": len([s for s in val if s['level'] == level]),
                    "test": len([s for s in test if s['level'] == level])
                }
                for level in ['foundation', 'conceptual', 'application', 'advanced']
            },
            "formats_generated": [
                "hf_train.jsonl, hf_validation.jsonl, hf_test.jsonl - HuggingFace format (USE THESE)",
                "openai_train.jsonl, openai_val.jsonl - OpenAI GPT-3.5/4 fine-tuning",
                "train.jsonl, val.jsonl, test.jsonl - Universal JSONL format",
                "instruction_dataset.json - Instruction-tuning format (Alpaca/Vicuna)"
            ],
            "recommended_models": [
                "OpenAI GPT-3.5-turbo",
                "OpenAI GPT-4",
                "CodeLlama-7b/13b/34b",
                "DeepSeek-Coder-6.7b/33b",
                "StarCoder-15b",
                "WizardCoder-15b"
            ]
        }
        
        info_file = self.finetuning_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        readme_content = self._generate_readme(info)
        readme_file = self.finetuning_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"  ✓ {info_file.name}")
        print(f"  ✓ {readme_file.name}")
    
    def _generate_readme(self, info: Dict) -> str:
        """Generate comprehensive README"""
        return f"""# DerivativeAnimator Fine-Tuning Dataset

## Overview
High-quality dataset for fine-tuning large language models to generate Manim code for derivative visualizations.

**Version**: {info['version']}  
**Total Samples**: {info['total_samples']}  

## Usage with HuggingFace

```python
from datasets import load_dataset

# Load the dataset (CORRECT WAY)
dataset = load_dataset('json', data_files={{
    'train': 'hf_train.jsonl',
    'validation': 'hf_validation.jsonl',
    'test': 'hf_test.jsonl'
}})

print(dataset)
# DatasetDict({{
#     train: Dataset,
#     validation: Dataset,
#     test: Dataset
# }})
```

## Files Generated

- `hf_train.jsonl`, `hf_validation.jsonl`, `hf_test.jsonl` - **Use these for HuggingFace training**
- `openai_train.jsonl`, `openai_val.jsonl` - OpenAI format
- `train.jsonl`, `val.jsonl`, `test.jsonl` - Universal JSONL
- `instruction_dataset.json` - Instruction format

## Quick Start

See the main README for complete training instructions.
"""

def main():
    print("\n" + "="*70)
    print("FINE-TUNING DATA PREPARATION TOOL")
    print("="*70)
    
    dataset_dir = Path("derivative_dataset_537")
    if not dataset_dir.exists():
        print(f"\n❌ Dataset directory not found: {dataset_dir}")
        print("Please run data_generation_pipeline.py first")
        return
    
    prep = FineTuningDataPreparation(str(dataset_dir))
    prep.prepare_all_formats(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print("\n" + "="*70)
    print("FINE-TUNING DATA READY")
    print("="*70)
    print(f"\n📁 Output directory: {prep.finetuning_dir}")
    print("\n📋 Generated files:")
    print("  • hf_train.jsonl, hf_validation.jsonl, hf_test.jsonl - HuggingFace (USE THESE)")
    print("  • openai_train.jsonl, openai_val.jsonl - OpenAI format")
    print("  • train.jsonl, val.jsonl, test.jsonl - Universal JSONL")
    print("  • instruction_dataset.json - Instruction format")
    print("  • dataset_info.json, quality_report.json, README.md")
    
    print("\n✅ Ready for fine-tuning!")

if __name__ == "__main__":
    main()