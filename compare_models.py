"""
Final Model Comparison for Paper
Compares Base Gemini vs Fine-Tuned DeepSeek-Coder
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluation_framework import DerivativeAnimatorEvaluator

class FinalModelComparison:
    def __init__(self, output_dir: str = "final_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = DerivativeAnimatorEvaluator()
        self.ft_model = None
        self.ft_tokenizer = None
    
    def load_test_data(self, test_file: str):
        """Load test data"""
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        return test_data
    
    def load_finetuned_model(self, model_path: str):
        """Load fine-tuned model once"""
        if self.ft_model is not None:
            return
        
        print(f"\n🔄 Loading fine-tuned model: {model_path}")
        
        self.ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.ft_model.eval()
        print("✓ Model loaded")
    
    def generate_with_finetuned(self, sample: dict) -> tuple:
        """Generate with fine-tuned model"""
        
        # Match training format
        prompt = f"""Task: Generate Manim code for derivative visualization

Function: {sample['input']}

Code:
"""
        
        inputs = self.ft_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.ft_model.device)
        
        with torch.no_grad():
            outputs = self.ft_model.generate(
                **inputs,
                max_new_tokens=1536,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.ft_tokenizer.eos_token_id,
                eos_token_id=self.ft_tokenizer.eos_token_id,
            )
        
        generated = self.ft_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract code
        code = self._extract_code(generated)
        metadata = {"model": "fine-tuned", "generated_length": len(generated)}
        
        return code, metadata
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from generated text"""
        import re
        
        # Try markdown blocks
        match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for manim import
        if "from manim import" in text:
            idx = text.find("from manim import")
            return text[idx:].strip()
        
        # Check if it already looks like code
        if "class DerivativeVisualization" in text:
            return text.strip()
        
        return ""
    
    def compare_models(self, test_data: list, finetuned_model_path: str, sample_size: int = None):
        """Run comparison"""
        
        print("\n" + "="*70)
        print("FINAL MODEL COMPARISON FOR PAPER")
        print("="*70)
        print("\nBase Model: Gemini 2.5-Flash (Zero-Shot)")
        print("Fine-Tuned: DeepSeek-Coder-1.3B")
        print("="*70 + "\n")
        
        # Load fine-tuned model
        self.load_finetuned_model(finetuned_model_path)
        
        # Sample if needed
        if sample_size:
            import random
            random.seed(42)
            test_data = random.sample(test_data, min(sample_size, len(test_data)))
        
        print(f"Testing on {len(test_data)} functions...\n")
        
        results = []
        
        for i, sample in enumerate(test_data, 1):
            function = sample['input']
            ground_truth = sample['output']
            
            print(f"[{i}/{len(test_data)}] f(x) = {function}")
            
            # Fine-tuned model
            print("  → Fine-Tuned Model...")
            try:
                ft_code, ft_meta = self.generate_with_finetuned(sample)
                ft_metrics = self.evaluator.evaluate_code(ft_code, skip_execution=True) if ft_code else None
                
                if ft_metrics:
                    print(f"    ✓ Score: {ft_metrics.overall_score:.1f}")
                else:
                    print(f"    ✗ Failed")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                ft_code, ft_metrics = "", None
                ft_meta = {"error": str(e)}
            
            # Ground truth (baseline)
            print("  → Ground Truth (Baseline)...")
            gt_metrics = self.evaluator.evaluate_code(ground_truth, skip_execution=True)
            print(f"    ✓ Score: {gt_metrics.overall_score:.1f}")
            
            # Record results
            result = {
                'function': function,
                'level': sample.get('level', 'unknown'),
                
                # Fine-tuned metrics
                'ft_success': bool(ft_code),
                'ft_score': ft_metrics.overall_score if ft_metrics else 0,
                'ft_syntax': ft_metrics.syntax_valid if ft_metrics else False,
                'ft_has_steps': ft_metrics.has_calculation_steps if ft_metrics else False,
                'ft_code_length': len(ft_code),
                
                # Ground truth metrics
                'gt_score': gt_metrics.overall_score,
                'gt_syntax': gt_metrics.syntax_valid,
                'gt_has_steps': gt_metrics.has_calculation_steps,
                'gt_length': len(ground_truth),
                
                # Comparison
                'score_ratio': (ft_metrics.overall_score / gt_metrics.overall_score * 100) if (ft_metrics and gt_metrics.overall_score > 0) else 0,
            }
            
            results.append(result)
            print(f"    Performance: {result['score_ratio']:.1f}% of ground truth\n")
        
        return pd.DataFrame(results)
    
    def generate_visualizations(self, df: pd.DataFrame):
        """Generate paper-ready visualizations"""
        
        print("\n📊 Generating visualizations...")
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 11
        
        # 1. Main comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Score comparison
        ax = axes[0, 0]
        means = [df['gt_score'].mean(), df['ft_score'].mean()]
        labels = ['Ground Truth\n(Gemini-Generated)', 'Fine-Tuned\n(DeepSeek-1.3B)']
        colors = ['#2ecc71', '#3498db']
        
        bars = ax.bar(labels, means, color=colors, edgecolor='black', linewidth=1.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Overall Score', fontweight='bold')
        ax.set_title('(a) Average Overall Score', fontweight='bold', loc='left')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Success rates
        ax = axes[0, 1]
        metrics = ['Success', 'Syntax\nValid', 'Has\nSteps']
        gt_rates = [
            df['gt_syntax'].mean() * 100,
            df['gt_syntax'].mean() * 100,
            df['gt_has_steps'].mean() * 100
        ]
        ft_rates = [
            df['ft_success'].mean() * 100,
            df['ft_syntax'].mean() * 100,
            df['ft_has_steps'].mean() * 100
        ]
        
        x = range(len(metrics))
        width = 0.35
        
        ax.bar([i-width/2 for i in x], gt_rates, width, label='Ground Truth', color='#2ecc71', edgecolor='black')
        ax.bar([i+width/2 for i in x], ft_rates, width, label='Fine-Tuned', color='#3498db', edgecolor='black')
        
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('(b) Success Metrics', fontweight='bold', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        # Performance ratio distribution
        ax = axes[1, 0]
        ax.hist(df['score_ratio'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(df['score_ratio'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {df["score_ratio"].mean():.1f}%')
        ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Perfect Match')
        
        ax.set_xlabel('Performance Ratio (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('(c) Performance Distribution', fontweight='bold', loc='left')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # By level
        ax = axes[1, 1]
        level_data = df.groupby('level')['score_ratio'].mean().sort_values()
        
        bars = ax.barh(range(len(level_data)), level_data.values, color='#3498db', edgecolor='black')
        ax.set_yticks(range(len(level_data)))
        ax.set_yticklabels([l.capitalize() for l in level_data.index])
        ax.set_xlabel('Performance Ratio (%)', fontweight='bold')
        ax.set_title('(d) Performance by Curriculum Level', fontweight='bold', loc='left')
        ax.axvline(100, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_figure.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ comparison_figure.png")
    
    def generate_report(self, df: pd.DataFrame):
        """Generate paper-ready report"""
        
        print("\n📝 Generating report...")
        
        summary = {
            'Metric': [
                'Success Rate (%)',
                'Syntax Valid (%)',
                'Has Calculation Steps (%)',
                'Average Overall Score',
                'Median Overall Score',
                'Average Code Length (chars)',
            ],
            'Ground Truth': [
                f"{df['gt_syntax'].mean() * 100:.1f}",
                f"{df['gt_syntax'].mean() * 100:.1f}",
                f"{df['gt_has_steps'].mean() * 100:.1f}",
                f"{df['gt_score'].mean():.1f}",
                f"{df['gt_score'].median():.1f}",
                f"{df['gt_length'].mean():.0f}",
            ],
            'Fine-Tuned Model': [
                f"{df['ft_success'].mean() * 100:.1f}",
                f"{df['ft_syntax'].mean() * 100:.1f}",
                f"{df['ft_has_steps'].mean() * 100:.1f}",
                f"{df['ft_score'].mean():.1f}",
                f"{df['ft_score'].median():.1f}",
                f"{df['ft_code_length'].mean():.0f}",
            ],
            'Performance Ratio': [
                f"{df['ft_success'].mean() / df['gt_syntax'].mean() * 100:.1f}%",
                f"{df['ft_syntax'].mean() / df['gt_syntax'].mean() * 100:.1f}%",
                f"{df['ft_has_steps'].mean() / df['gt_has_steps'].mean() * 100:.1f}%",
                f"{df['score_ratio'].mean():.1f}%",
                f"{df['score_ratio'].median():.1f}%",
                f"{df['ft_code_length'].mean() / df['gt_length'].mean() * 100:.1f}%",
            ]
        }
        
        df_table = pd.DataFrame(summary)
        df_table.to_csv(self.output_dir / 'comparison_table.csv', index=False)
        
        # Generate LaTeX table
        latex = r"""\begin{table}[h]
\centering
\caption{Performance Comparison: Ground Truth vs Fine-Tuned Model}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Ground Truth} & \textbf{Fine-Tuned} & \textbf{Ratio} \\
\midrule
"""
        for _, row in df_table.iterrows():
            latex += f"{row['Metric']} & {row['Ground Truth']} & {row['Fine-Tuned Model']} & {row['Performance Ratio']} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        with open(self.output_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex)
        
        print("  ✓ comparison_table.csv")
        print("  ✓ comparison_table.tex")
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(df_table.to_string(index=False))

def main():
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON FOR PAPER")
    print("="*70)
    
    comparator = FinalModelComparison()
    
    # Load test data
    test_data = comparator.load_test_data("derivative_dataset_537/finetuning/hf_test.jsonl")
    print(f"\n✓ Loaded {len(test_data)} test samples")
    
    # Run comparison
    model_path = "derivative-animator-deepseek-1.3b/merged_model"
    
    df = comparator.compare_models(
        test_data,
        finetuned_model_path=model_path,
        sample_size=50  # Use all test samples or specify number
    )
    
    # Save detailed results
    df.to_csv(comparator.output_dir / 'detailed_results.csv', index=False)
    print(f"\n✓ Saved: detailed_results.csv")
    
    # Generate visualizations
    comparator.generate_visualizations(df)
    
    # Generate report
    comparator.generate_report(df)
    
    print("\n✅ Comparison complete!")
    print(f"📁 All outputs in: {comparator.output_dir}/")
    print("\nFiles generated:")
    print("  • comparison_figure.png - 4-panel visualization for paper")
    print("  • comparison_table.csv - Summary statistics")
    print("  • comparison_table.tex - LaTeX table for paper")
    print("  • detailed_results.csv - Per-function results")

if __name__ == "__main__":
    main()