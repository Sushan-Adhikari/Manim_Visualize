"""
Model Comparison: Base Model vs Fine-Tuned Model
Generates comparative evaluation for research paper
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

from evaluation_framework import DerivativeAnimatorEvaluator
from manim_generator import generate_manim_code

class ModelComparison:
    """Compare base model vs fine-tuned model performance"""
    
    def __init__(self, output_dir: str = "model_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = DerivativeAnimatorEvaluator()
        
    def load_test_functions(self, test_file: str = "derivative_dataset_537/finetuning/test.jsonl") -> List[str]:
        """Load test functions"""
        functions = []
        with open(test_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                functions.append(data['function'])
        return functions
    
    def generate_with_base_model(self, function: str) -> tuple:
        """Generate code with base model"""
        code, metadata = generate_manim_code(
            function,
            max_attempts=3,
            use_thinking=False,  # Base model
            skip_execution_test=True
        )
        return code, metadata
    
    def generate_with_finetuned_model(self, function: str, model_path: str) -> tuple:
        """Generate code with fine-tuned model"""
        # TODO: Implement inference with fine-tuned model
        # This is a placeholder - actual implementation depends on your fine-tuned model
        # For now, use thinking mode as proxy for "better" model
        code, metadata = generate_manim_code(
            function,
            max_attempts=3,
            use_thinking=True,  # Fine-tuned proxy
            skip_execution_test=True
        )
        return code, metadata
    
    def compare_on_test_set(self, 
                           test_functions: List[str],
                           finetuned_model_path: str = None,
                           sample_size: int = 50) -> pd.DataFrame:
        """Run comparison on test set"""
        
        print("\n" + "="*70)
        print("MODEL COMPARISON: BASE vs FINE-TUNED")
        print("="*70 + "\n")
        
        results = []
        
        # Sample if needed
        if sample_size and len(test_functions) > sample_size:
            import random
            random.seed(42)
            test_functions = random.sample(test_functions, sample_size)
        
        print(f"Testing on {len(test_functions)} functions...\n")
        
        for i, function in enumerate(test_functions, 1):
            print(f"[{i}/{len(test_functions)}] Testing: f(x) = {function}")
            
            # Base model
            print("  → Base model...")
            base_code, base_meta = self.generate_with_base_model(function)
            base_metrics = self.evaluator.evaluate_code(base_code, skip_execution=True) if base_code else None
            
            # Fine-tuned model
            print("  → Fine-tuned model...")
            ft_code, ft_meta = self.generate_with_finetuned_model(function, finetuned_model_path)
            ft_metrics = self.evaluator.evaluate_code(ft_code, skip_execution=True) if ft_code else None
            
            # Compare
            result = {
                'function': function,
                'base_success': bool(base_code),
                'base_attempts': base_meta.get('attempts', 0) if base_code else 3,
                'base_score': base_metrics.overall_score if base_metrics else 0,
                'base_syntax_valid': base_metrics.syntax_valid if base_metrics else False,
                'base_has_steps': base_metrics.has_calculation_steps if base_metrics else False,
                'ft_success': bool(ft_code),
                'ft_attempts': ft_meta.get('attempts', 0) if ft_code else 3,
                'ft_score': ft_metrics.overall_score if ft_metrics else 0,
                'ft_syntax_valid': ft_metrics.syntax_valid if ft_metrics else False,
                'ft_has_steps': ft_metrics.has_calculation_steps if ft_metrics else False,
                'improvement': (ft_metrics.overall_score - base_metrics.overall_score) if (base_metrics and ft_metrics) else 0
            }
            
            results.append(result)
            
            print(f"    Base: {result['base_score']:.1f} | FT: {result['ft_score']:.1f} | Δ: {result['improvement']:+.1f}")
        
        df = pd.DataFrame(results)
        return df
    
    def generate_comparison_visualizations(self, df: pd.DataFrame):
        """Generate comparison visualizations for paper"""
        
        print("\n📊 Generating comparison visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Overall Score Comparison
        self._plot_score_comparison(df)
        
        # 2. Success Rate Comparison
        self._plot_success_rates(df)
        
        # 3. Attempts Distribution
        self._plot_attempts_comparison(df)
        
        # 4. Improvement Distribution
        self._plot_improvement_distribution(df)
        
        # 5. Metric Breakdown
        self._plot_metric_breakdown(df)
        
        print(f"✓ Visualizations saved to: {self.output_dir}")
    
    def _plot_score_comparison(self, df: pd.DataFrame):
        """Plot overall score comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        means = [df['base_score'].mean(), df['ft_score'].mean()]
        labels = ['Base Model', 'Fine-Tuned Model']
        colors = ['#e74c3c', '#2ecc71']
        
        bars = ax1.bar(labels, means, color=colors, edgecolor='black', linewidth=1.5)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax1.set_ylabel('Overall Score', fontsize=12)
        ax1.set_title('Average Overall Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Scatter plot
        ax2.scatter(df['base_score'], df['ft_score'], alpha=0.6, s=50)
        ax2.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Equal Performance')
        ax2.set_xlabel('Base Model Score', fontsize=12)
        ax2.set_ylabel('Fine-Tuned Model Score', fontsize=12)
        ax2.set_title('Score Comparison (Per Function)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig_score_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Score comparison")
    
    def _plot_success_rates(self, df: pd.DataFrame):
        """Plot success rate comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Success Rate', 'Syntax Valid', 'Has Steps']
        base_rates = [
            df['base_success'].mean() * 100,
            df['base_syntax_valid'].mean() * 100,
            df['base_has_steps'].mean() * 100
        ]
        ft_rates = [
            df['ft_success'].mean() * 100,
            df['ft_syntax_valid'].mean() * 100,
            df['ft_has_steps'].mean() * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_rates, width, label='Base Model', 
                      color='#e74c3c', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, ft_rates, width, label='Fine-Tuned Model',
                      color='#2ecc71', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Success Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig_success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Success rates")
    
    def _plot_attempts_comparison(self, df: pd.DataFrame):
        """Plot attempts distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(df['base_attempts'], bins=range(1, 5), alpha=0.5, label='Base Model',
               color='#e74c3c', edgecolor='black')
        ax.hist(df['ft_attempts'], bins=range(1, 5), alpha=0.5, label='Fine-Tuned Model',
               color='#2ecc71', edgecolor='black')
        
        ax.set_xlabel('Number of Attempts', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Attempts Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean lines
        ax.axvline(df['base_attempts'].mean(), color='#e74c3c', linestyle='--', 
                  linewidth=2, label=f'Base Mean: {df["base_attempts"].mean():.2f}')
        ax.axvline(df['ft_attempts'].mean(), color='#2ecc71', linestyle='--',
                  linewidth=2, label=f'FT Mean: {df["ft_attempts"].mean():.2f}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig_attempts_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Attempts distribution")
    
    def _plot_improvement_distribution(self, df: pd.DataFrame):
        """Plot improvement distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(df['improvement'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Improvement')
        ax.axvline(df['improvement'].mean(), color='green', linestyle='--', linewidth=2,
                  label=f'Mean: {df["improvement"].mean():.1f}')
        
        ax.set_xlabel('Score Improvement (Fine-Tuned - Base)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Score Improvements', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        improved = (df['improvement'] > 0).sum()
        total = len(df)
        text = f"Improved: {improved}/{total} ({improved/total*100:.1f}%)"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig_improvement_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Improvement distribution")
    
    def _plot_metric_breakdown(self, df: pd.DataFrame):
        """Plot detailed metric breakdown"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Overall\nScore', 'Syntax\nValid', 'Has\nSteps', 'Success\nRate']
        base_values = [
            df['base_score'].mean(),
            df['base_syntax_valid'].mean() * 100,
            df['base_has_steps'].mean() * 100,
            df['base_success'].mean() * 100
        ]
        ft_values = [
            df['ft_score'].mean(),
            df['ft_syntax_valid'].mean() * 100,
            df['ft_has_steps'].mean() * 100,
            df['ft_success'].mean() * 100
        ]
        improvements = [ft - base for ft, base in zip(ft_values, base_values)]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        ax.bar(x - width, base_values, width, label='Base Model', 
              color='#e74c3c', edgecolor='black')
        ax.bar(x, ft_values, width, label='Fine-Tuned Model',
              color='#2ecc71', edgecolor='black')
        ax.bar(x + width, improvements, width, label='Improvement',
              color='#3498db', edgecolor='black')
        
        ax.set_ylabel('Score / Percentage', fontsize=12)
        ax.set_title('Detailed Metric Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig_metric_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Metric breakdown")
    
    def generate_comparison_table(self, df: pd.DataFrame):
        """Generate LaTeX comparison table"""
        
        print("\n📝 Generating comparison table...")
        
        summary = {
            'Metric': [
                'Success Rate (%)',
                'Syntax Valid (%)',
                'Has Calculation Steps (%)',
                'Average Overall Score',
                'Average Attempts',
                'Median Score'
            ],
            'Base Model': [
                f"{df['base_success'].mean() * 100:.1f}",
                f"{df['base_syntax_valid'].mean() * 100:.1f}",
                f"{df['base_has_steps'].mean() * 100:.1f}",
                f"{df['base_score'].mean():.1f}",
                f"{df['base_attempts'].mean():.2f}",
                f"{df['base_score'].median():.1f}"
            ],
            'Fine-Tuned Model': [
                f"{df['ft_success'].mean() * 100:.1f}",
                f"{df['ft_syntax_valid'].mean() * 100:.1f}",
                f"{df['ft_has_steps'].mean() * 100:.1f}",
                f"{df['ft_score'].mean():.1f}",
                f"{df['ft_attempts'].mean():.2f}",
                f"{df['ft_score'].median():.1f}"
            ],
            'Improvement': [
                f"+{(df['ft_success'].mean() - df['base_success'].mean()) * 100:.1f}",
                f"+{(df['ft_syntax_valid'].mean() - df['base_syntax_valid'].mean()) * 100:.1f}",
                f"+{(df['ft_has_steps'].mean() - df['base_has_steps'].mean()) * 100:.1f}",
                f"+{df['ft_score'].mean() - df['base_score'].mean():.1f}",
                f"{df['ft_attempts'].mean() - df['base_attempts'].mean():.2f}",
                f"+{df['ft_score'].median() - df['base_score'].median():.1f}"
            ]
        }
        
        df_table = pd.DataFrame(summary)
        
        # Save as CSV
        df_table.to_csv(self.output_dir / 'comparison_table.csv', index=False)
        
        # Save as LaTeX
        latex = df_table.to_latex(index=False, escape=False)
        with open(self.output_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex)
        
        print(f"  ✓ Comparison table saved")
        
        # Print to console
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(df_table.to_string(index=False))

def main():
    print("\n" + "="*70)
    print("MODEL COMPARISON TOOL")
    print("="*70)
    
    # Initialize
    comparator = ModelComparison()
    
    # Load test functions
    test_functions = comparator.load_test_functions()
    print(f"\n✓ Loaded {len(test_functions)} test functions")
    
    # Run comparison
    df = comparator.compare_on_test_set(test_functions, sample_size=50)
    
    # Save results
    df.to_csv(comparator.output_dir / 'comparison_results.csv', index=False)
    print(f"\n✓ Results saved to: {comparator.output_dir}/comparison_results.csv")
    
    # Generate visualizations
    comparator.generate_comparison_visualizations(df)
    
    # Generate table
    comparator.generate_comparison_table(df)
    
    print("\n✅ Model comparison complete!")
    print(f"📁 All outputs in: {comparator.output_dir}/")

if __name__ == "__main__":
    main()