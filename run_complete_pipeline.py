#!/usr/bin/env python3
"""
Master Pipeline Orchestrator for DerivativeAnimator
Runs complete workflow: Dataset → Visualization → Fine-tuning Prep → Evaluation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

class PipelineOrchestrator:
    """Orchestrates the complete DerivativeAnimator pipeline"""
    
    def __init__(self, skip_dataset=False, skip_viz=False, skip_eval=False):
        self.skip_dataset = skip_dataset
        self.skip_viz = skip_viz
        self.skip_eval = skip_eval
        self.start_time = datetime.now()
        self.results = {}
        
    def print_header(self, title):
        """Print section header"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70 + "\n")
    
    def print_step(self, step_num, total, description):
        """Print step information"""
        print(f"\n{'─'*70}")
        print(f"STEP {step_num}/{total}: {description}")
        print(f"{'─'*70}")
    
    def run_command(self, cmd, description):
        """Run a command and capture result"""
        print(f"\n🚀 Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed")
            print(f"Error: {e.stderr[:500]}")
            return False, e.stderr
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            return False, f"Command not found: {cmd[0]}"
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        self.print_header("CHECKING PREREQUISITES")
        
        checks = {
            'Python 3.8+': self._check_python(),
            'Manim': self._check_manim(),
            'API Keys': self._check_api_keys(),
            'Dependencies': self._check_dependencies(),
        }
        
        all_passed = all(checks.values())
        
        print("\n📋 Prerequisites Check:")
        for name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
        
        if not all_passed:
            print("\n⚠️  Some prerequisites failed. Please fix them before continuing.")
            return False
        
        print("\n✅ All prerequisites satisfied")
        return True
    
    def _check_python(self):
        """Check Python version"""
        import sys
        version = sys.version_info
        return version.major >= 3 and version.minor >= 8
    
    def _check_manim(self):
        """Check if Manim is installed"""
        try:
            import manim
            return True
        except ImportError:
            return False
    
    def _check_api_keys(self):
        """Check if required API keys are set"""
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("GEMINI_API_KEY") is not None
    
    def _check_dependencies(self):
        """Check if required packages are installed"""
        required = ['pandas', 'matplotlib', 'seaborn', 'torch', 'transformers']
        try:
            for package in required:
                __import__(package)
            return True
        except ImportError:
            return False
    
    def run_dataset_generation(self):
        """Step 1: Generate dataset"""
        self.print_step(1, 5, "Dataset Generation (537 samples)")
        
        if self.skip_dataset:
            print("⏭️  Skipping dataset generation (--skip-dataset)")
            return True
        
        # Check if dataset already exists
        dataset_dir = Path("derivative_dataset_537/code")
        if dataset_dir.exists():
            file_count = len(list(dataset_dir.rglob("*.py")))
            if file_count > 100:
                print(f"⚠️  Dataset already exists ({file_count} files)")
                response = input("Regenerate? This will take 4-6 hours (y/N): ")
                if response.lower() != 'y':
                    print("Using existing dataset")
                    return True
        
        success, output = self.run_command(
            ['python', 'data_generation_pipeline.py'],
            "Dataset Generation"
        )
        
        self.results['dataset_generation'] = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        return success
    
    def run_visualizations(self):
        """Step 2: Generate visualizations"""
        self.print_step(2, 5, "Dataset Visualization")
        
        if self.skip_viz:
            print("⏭️  Skipping visualization (--skip-viz)")
            return True
        
        success, output = self.run_command(
            ['python', 'dataset_visualizer.py'],
            "Dataset Visualization"
        )
        
        self.results['visualization'] = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        return success
    
    def run_finetuning_prep(self):
        """Step 3: Prepare fine-tuning data"""
        self.print_step(3, 5, "Fine-Tuning Data Preparation")
        
        success, output = self.run_command(
            ['python', 'finetuning_data_preparation.py'],
            "Fine-Tuning Data Preparation"
        )
        
        self.results['finetuning_prep'] = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        return success
    
    def run_evaluation(self):
        """Step 4: Evaluate dataset"""
        self.print_step(4, 5, "Dataset Evaluation")
        
        if self.skip_eval:
            print("⏭️  Skipping evaluation (--skip-eval)")
            return True
        
        success, output = self.run_command(
            ['python', 'evaluation_framework.py'],
            "Dataset Evaluation"
        )
        
        self.results['evaluation'] = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        return success
    
    def generate_summary(self):
        """Step 5: Generate final summary"""
        self.print_step(5, 5, "Generating Summary Report")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'pipeline_run': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': elapsed,
                'duration_formatted': f"{elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m"
            },
            'steps': self.results
        }
        
        # Save summary
        summary_file = Path("pipeline_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_file}")
        
        return True
    
    def print_final_report(self):
        """Print final report"""
        self.print_header("PIPELINE COMPLETE")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"⏱️  Total Time: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s")
        print(f"📊 Steps Completed: {len([r for r in self.results.values() if r['success']])}/{len(self.results)}")
        
        print("\n📁 Generated Outputs:")
        outputs = [
            ("derivative_dataset_537/code/", "Generated Manim code (537 samples)"),
            ("derivative_dataset_537/visualizations/", "Dataset visualizations (8 figures)"),
            ("derivative_dataset_537/finetuning/", "Fine-tuning data (multiple formats)"),
            ("derivative_dataset_537/evaluation/", "Evaluation results"),
            ("pipeline_summary.json", "Pipeline execution summary"),
        ]
        
        for path, desc in outputs:
            if Path(path).exists():
                print(f"  ✅ {path} - {desc}")
            else:
                print(f"  ⚠️  {path} - {desc} (not found)")
        
        print("\n📝 Next Steps:")
        print("  1. Review visualizations in derivative_dataset_537/visualizations/")
        print("  2. Check evaluation results in derivative_dataset_537/evaluation/")
        print("  3. Fine-tune model: python finetuning_train.py")
        print("  4. Launch web UI: python gradio_ui.py")
        print("  5. Write paper using generated figures and statistics")
        
        print("\n" + "="*70)
        print("✅ DerivativeAnimator Pipeline Complete!")
        print("="*70)
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        self.print_header("DERIVATIVEANIMATOR - COMPLETE PIPELINE")
        
        print("This will execute the following steps:")
        print("  1. Dataset Generation (537 samples) - ~4-6 hours")
        print("  2. Dataset Visualization (8 figures)")
        print("  3. Fine-Tuning Data Preparation")
        print("  4. Dataset Evaluation")
        print("  5. Summary Report Generation")
        
        if not self.skip_dataset:
            print("\n⚠️  Step 1 will take 4-6 hours due to API rate limits")
            response = input("\nContinue? (y/N): ")
            if response.lower() != 'y':
                print("Pipeline cancelled")
                return False
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Run pipeline steps
        steps = [
            ('dataset', self.run_dataset_generation),
            ('visualization', self.run_visualizations),
            ('finetuning_prep', self.run_finetuning_prep),
            ('evaluation', self.run_evaluation),
            ('summary', self.generate_summary),
        ]
        
        for step_name, step_func in steps:
            success = step_func()
            if not success and step_name in ['dataset', 'finetuning_prep']:
                print(f"\n❌ Critical step '{step_name}' failed. Stopping pipeline.")
                return False
        
        # Print final report
        self.print_final_report()
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='DerivativeAnimator Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_complete_pipeline.py
  
  # Skip dataset generation (use existing)
  python run_complete_pipeline.py --skip-dataset
  
  # Skip visualization and evaluation
  python run_complete_pipeline.py --skip-viz --skip-eval
  
  # Run only fine-tuning preparation
  python run_complete_pipeline.py --skip-dataset --skip-viz --skip-eval
        """
    )
    
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset generation (use existing dataset)'
    )
    
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation (faster for testing)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(
        skip_dataset=args.skip_dataset,
        skip_viz=args.skip_viz,
        skip_eval=args.skip_eval
    )
    
    success = orchestrator.run_complete_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()