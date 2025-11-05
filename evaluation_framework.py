"""
Comprehensive Evaluation Framework for DerivativeAnimator
Evaluates: Syntax correctness, mathematical accuracy, pedagogical quality
"""

import json
import re
import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Syntax metrics
    syntax_valid: bool = False
    manim_imports_correct: bool = False
    class_structure_correct: bool = False
    has_required_methods: bool = False
    
    # Execution metrics
    can_execute: bool = False
    execution_time: float = 0.0
    execution_error: Optional[str] = None
    
    # Mathematical metrics
    function_defined: bool = False
    derivative_defined: bool = False
    math_notation_correct: bool = False
    
    # Pedagogical metrics
    has_calculation_steps: bool = False
    has_visualization: bool = False
    has_animation: bool = False
    code_length: int = 0
    
    # BLEU score (if reference available)
    bleu_score: Optional[float] = None
    
    # Overall score
    overall_score: float = 0.0

class CodeSyntaxValidator:
    """Validates Python/Manim syntax"""
    
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code is valid Python"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    @staticmethod
    def check_imports(code: str) -> bool:
        """Check for required imports"""
        required = ['from manim import', 'import numpy']
        return all(req in code for req in required)
    
    @staticmethod
    def check_class_structure(code: str) -> bool:
        """Check for proper Manim Scene class"""
        has_class = re.search(r'class\s+\w+\(Scene\)', code) is not None
        has_construct = 'def construct(self):' in code
        return has_class and has_construct
    
    @staticmethod
    def check_required_elements(code: str) -> Dict[str, bool]:
        """Check for required Manim elements"""
        return {
            'axes': 'Axes(' in code,
            'graph': 'plot(' in code or 'graph' in code,
            'derivative': "f_prime" in code or "derivative" in code,
            'animation': 'self.play(' in code,
            'tangent': 'tangent' in code.lower(),
        }

class MathematicalValidator:
    """Validates mathematical correctness"""
    
    @staticmethod
    def extract_function(code: str) -> Optional[str]:
        """Extract function definition"""
        match = re.search(r'def f\(x\):\s*\n\s*return (.+)', code)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def extract_derivative(code: str) -> Optional[str]:
        """Extract derivative definition"""
        match = re.search(r'def f_prime\(x\):\s*\n\s*return (.+)', code)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def check_latex_notation(code: str) -> bool:
        """Check if LaTeX notation is present and formatted"""
        mathtex_count = code.count('MathTex')
        has_proper_escaping = r'\frac' in code or r'\sin' in code or 'f(x)' in code
        return mathtex_count >= 3 and has_proper_escaping

class PedagogicalValidator:
    """Validates pedagogical quality"""
    
    @staticmethod
    def check_calculation_steps(code: str) -> Dict[str, bool]:
        """Check for step-by-step calculation display"""
        return {
            'has_title': 'Derivative Calculation' in code or 'calc_title' in code,
            'has_step1': 'calc_step1' in code or 'step_1' in code,
            'has_step2': 'calc_step2' in code or 'step_2' in code,
            'has_final': 'calc_final' in code or 'final' in code,
            'steps_animated': code.count('Write(calc_') >= 3,
        }
    
    @staticmethod
    def check_visualization_quality(code: str) -> Dict[str, bool]:
        """Check visualization elements"""
        return {
            'moving_dot': 'Dot(' in code and 'always_redraw' in code,
            'tangent_line': 'tangent' in code.lower(),
            'value_tracker': 'ValueTracker' in code,
            'dynamic_label': 'deriv_label' in code or 'derivative' in code.lower(),
        }
    
    @staticmethod
    def check_animation_sequence(code: str) -> Dict[str, bool]:
        """Check animation sequencing"""
        play_count = code.count('self.play(')
        wait_count = code.count('self.wait(')
        
        return {
            'has_intro': play_count >= 1,
            'has_build_up': play_count >= 3,
            'has_pauses': wait_count >= 2,
            'smooth_transition': 'rate_func=smooth' in code,
        }

class ExecutionValidator:
    """Validates code execution"""
    
    @staticmethod
    def test_execution(code: str, timeout: int = 90) -> Tuple[bool, float, Optional[str]]:
        """Test if code can be executed"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Extract class name
            class_match = re.search(r'class (\w+)\(Scene\):', code)
            if not class_match:
                return False, 0.0, "No Scene class found"
            
            class_name = class_match.group(1)
            
            import time
            start_time = time.time()
            
            result = subprocess.run(
                ['manim', '-pql', '--dry_run', temp_file, class_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                return False, execution_time, error_msg
            
            return True, execution_time, None
            
        except subprocess.TimeoutExpired:
            return False, timeout, "Execution timeout"
        except Exception as e:
            return False, 0.0, str(e)
        finally:
            try:
                import os
                os.unlink(temp_file)
            except:
                pass

class BLEUScoreCalculator:
    """Calculate BLEU score for code similarity"""
    
    @staticmethod
    def tokenize_code(code: str) -> List[str]:
        """Tokenize code into words"""
        # Remove comments and strings
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'".*?"', '', code)
        code = re.sub(r"'.*?'", '', code)
        # Split into tokens
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [t.lower() for t in tokens if t.strip()]
    
    @staticmethod
    def calculate_bleu(reference: str, candidate: str) -> float:
        """Calculate simplified BLEU score"""
        ref_tokens = BLEUScoreCalculator.tokenize_code(reference)
        cand_tokens = BLEUScoreCalculator.tokenize_code(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Calculate precision for n-grams (n=1,2,3,4)
        precisions = []
        for n in range(1, 5):
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)]
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
            precision = matches / len(cand_ngrams)
            precisions.append(precision)
        
        # Geometric mean
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / len(cand_tokens)))
        
        return bp * bleu

class DerivativeAnimatorEvaluator:
    """Main evaluation class"""
    
    def __init__(self):
        self.syntax_validator = CodeSyntaxValidator()
        self.math_validator = MathematicalValidator()
        self.pedagogical_validator = PedagogicalValidator()
        self.execution_validator = ExecutionValidator()
        self.bleu_calculator = BLEUScoreCalculator()
    
    def evaluate_code(self, 
                     code: str, 
                     reference_code: Optional[str] = None,
                     skip_execution: bool = False) -> EvaluationMetrics:
        """Comprehensive evaluation of generated code"""
        
        metrics = EvaluationMetrics()
        
        # 1. Syntax validation
        syntax_valid, syntax_error = self.syntax_validator.validate_syntax(code)
        metrics.syntax_valid = syntax_valid
        metrics.manim_imports_correct = self.syntax_validator.check_imports(code)
        metrics.class_structure_correct = self.syntax_validator.check_class_structure(code)
        
        required_elements = self.syntax_validator.check_required_elements(code)
        metrics.has_required_methods = sum(required_elements.values()) >= 3
        
        # 2. Mathematical validation
        metrics.function_defined = self.math_validator.extract_function(code) is not None
        metrics.derivative_defined = self.math_validator.extract_derivative(code) is not None
        metrics.math_notation_correct = self.math_validator.check_latex_notation(code)
        
        # 3. Pedagogical validation
        calc_steps = self.pedagogical_validator.check_calculation_steps(code)
        metrics.has_calculation_steps = sum(calc_steps.values()) >= 3
        
        viz_quality = self.pedagogical_validator.check_visualization_quality(code)
        metrics.has_visualization = sum(viz_quality.values()) >= 2
        
        anim_sequence = self.pedagogical_validator.check_animation_sequence(code)
        metrics.has_animation = sum(anim_sequence.values()) >= 2
        
        metrics.code_length = len(code)
        
        # 4. Execution validation
        if not skip_execution and syntax_valid:
            can_exec, exec_time, exec_error = self.execution_validator.test_execution(code)
            metrics.can_execute = can_exec
            metrics.execution_time = exec_time
            metrics.execution_error = exec_error
        
        # 5. BLEU score
        if reference_code:
            metrics.bleu_score = self.bleu_calculator.calculate_bleu(reference_code, code)
        
        # 6. Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate weighted overall score"""
        weights = {
            'syntax': 0.20,
            'execution': 0.25,
            'mathematical': 0.25,
            'pedagogical': 0.30,
        }
        
        syntax_score = (
            metrics.syntax_valid * 0.4 +
            metrics.manim_imports_correct * 0.2 +
            metrics.class_structure_correct * 0.2 +
            metrics.has_required_methods * 0.2
        )
        
        execution_score = metrics.can_execute * 1.0
        
        math_score = (
            metrics.function_defined * 0.4 +
            metrics.derivative_defined * 0.4 +
            metrics.math_notation_correct * 0.2
        )
        
        pedagogical_score = (
            metrics.has_calculation_steps * 0.4 +
            metrics.has_visualization * 0.3 +
            metrics.has_animation * 0.3
        )
        
        overall = (
            syntax_score * weights['syntax'] +
            execution_score * weights['execution'] +
            math_score * weights['mathematical'] +
            pedagogical_score * weights['pedagogical']
        )
        
        return overall * 100  # Convert to percentage

    def evaluate_dataset(self, 
                        dataset_dir: str = "derivative_dataset_537/code",
                        sample_size: Optional[int] = 50,
                        skip_execution: bool = False) -> pd.DataFrame:
        """Evaluate entire dataset"""
        
        print("\n" + "="*70)
        print("EVALUATING DATASET")
        print("="*70 + "\n")
        
        results = []
        code_files = list(Path(dataset_dir).rglob("*.py"))
        
        if sample_size:
            import random
            code_files = random.sample(code_files, min(sample_size, len(code_files)))
        
        total = len(code_files)
        
        for i, code_file in enumerate(code_files, 1):
            print(f"[{i}/{total}] Evaluating: {code_file.name}")
            
            with open(code_file, 'r') as f:
                code = f.read()
            
            metrics = self.evaluate_code(code, skip_execution=skip_execution)
            
            result = {
                'file': code_file.name,
                'level': code_file.parent.name,
                **asdict(metrics)
            }
            results.append(result)
            
            if i % 10 == 0:
                print(f"  Progress: {i}/{total} ({i/total*100:.1f}%)")
        
        df = pd.DataFrame(results)
        return df
    
    def generate_report(self, 
                       df: pd.DataFrame, 
                       output_dir: str = "derivative_dataset_537/evaluation"):
        """Generate comprehensive evaluation report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n📊 Generating evaluation report...")
        
        # Summary statistics
        summary = {
            'total_samples': len(df),
            'syntax_valid': (df['syntax_valid'].sum() / len(df) * 100),
            'can_execute': (df['can_execute'].sum() / len(df) * 100),
            'has_calculation_steps': (df['has_calculation_steps'].sum() / len(df) * 100),
            'has_visualization': (df['has_visualization'].sum() / len(df) * 100),
            'average_overall_score': df['overall_score'].mean(),
            'median_overall_score': df['overall_score'].median(),
            'average_execution_time': df[df['execution_time'] > 0]['execution_time'].mean(),
        }
        
        # By level
        by_level = df.groupby('level').agg({
            'overall_score': ['mean', 'std'],
            'can_execute': 'sum',
            'syntax_valid': 'sum'
        }).round(2)
        
        # Save results
        df.to_csv(output_path / 'evaluation_results.csv', index=False)
        
        with open(output_path / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        by_level.to_csv(output_path / 'evaluation_by_level.csv')
        
        # Generate LaTeX table
        latex_table = self._generate_latex_table(df)
        with open(output_path / 'evaluation_table.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"✓ Report saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\nTotal Samples: {summary['total_samples']}")
        print(f"Syntax Valid: {summary['syntax_valid']:.1f}%")
        print(f"Can Execute: {summary['can_execute']:.1f}%")
        print(f"Has Calculation Steps: {summary['has_calculation_steps']:.1f}%")
        print(f"Has Visualization: {summary['has_visualization']:.1f}%")
        print(f"Average Overall Score: {summary['average_overall_score']:.2f}/100")
        print(f"Median Overall Score: {summary['median_overall_score']:.2f}/100")
        if summary['average_execution_time'] > 0:
            print(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
    
    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table for paper"""
        by_level = df.groupby('level').agg({
            'overall_score': 'mean',
            'syntax_valid': lambda x: (x.sum() / len(x) * 100),
            'can_execute': lambda x: (x.sum() / len(x) * 100),
            'has_calculation_steps': lambda x: (x.sum() / len(x) * 100),
        }).round(1)
        
        latex = r"""\begin{table}[h]
\centering
\caption{Evaluation Metrics by Curriculum Level}
\begin{tabular}{lrrrr}
\toprule
Level & Overall Score & Syntax Valid & Can Execute & Has Steps \\
\midrule
"""
        for level, row in by_level.iterrows():
            latex += fr"{row['syntax_valid']:.1f}% & {row['can_execute']:.1f}% & {row['has_calculation_steps']:.1f}% \\\\"

        
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        return latex

def main():
    """Main evaluation function"""
    evaluator = DerivativeAnimatorEvaluator()
    
    # Evaluate dataset
    df = evaluator.evaluate_dataset(
        dataset_dir="derivative_dataset_537/code",
        sample_size=None,  # Evaluate all, or set number for sampling
        skip_execution=False  # Set True to skip execution tests
    )
    
    # Generate report
    evaluator.generate_report(df)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()