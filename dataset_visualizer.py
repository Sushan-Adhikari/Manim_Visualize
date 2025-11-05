import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
import re

class DatasetVisualizer:
    """Generate comprehensive visualizations for derivative dataset analysis"""
    
    def __init__(self, dataset_dir: str = "derivative_dataset_537"):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_dir = self.dataset_dir / "metadata"
        self.viz_dir = self.dataset_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        
        # Set style for publication quality
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)  # Better for papers
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.family'] = 'serif'  # Better for academic papers
        plt.rcParams['font.size'] = 11
        
    def _load_data(self) -> pd.DataFrame:
        """Load generation report data"""
        report_file = self.metadata_dir / "generation_report.json"
        
        if not report_file.exists():
            raise FileNotFoundError(f"Report not found: {report_file}")
        
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['results'])
        
        # Add function type classification
        df['function_type'] = df['function'].apply(self._classify_function)
        
        # Add complexity score (for analysis)
        df['complexity_score'] = df['function'].apply(self._calculate_complexity)
        
        return df
    
    def _classify_function(self, func: str) -> str:
        """Classify function by type"""
        func = func.lower()
        
        # Priority order matters
        if 'sin' in func and 'cos' in func:
            return 'Trig Product'
        elif 'sin' in func or 'cos' in func or 'tan' in func:
            return 'Trigonometric'
        elif 'e^' in func or 'exp' in func:
            return 'Exponential'
        elif 'ln' in func or 'log' in func:
            return 'Logarithmic'
        elif 'sqrt' in func:
            return 'Radical'
        elif '/' in func:
            return 'Rational'
        elif '^' in func:
            if any(op in func for op in ['sin', 'cos', 'e^', 'ln', '*']):
                return 'Composite'
            return 'Polynomial'
        else:
            return 'Linear'
    
    def _calculate_complexity(self, func: str) -> int:
        """Calculate rough complexity score for function"""
        score = 0
        score += func.count('^') * 2  # Powers
        score += func.count('*') * 1  # Multiplication
        score += func.count('/') * 2  # Division
        score += func.count('sin') * 3  # Trig functions
        score += func.count('cos') * 3
        score += func.count('tan') * 3
        score += func.count('ln') * 3  # Logarithm
        score += func.count('e^') * 3  # Exponential
        score += func.count('sqrt') * 2  # Square root
        score += len(func.split('+')) + len(func.split('-')) - 1  # Terms
        return score
    
    def generate_all_visualizations(self):
        """Generate all visualizations for the paper"""
        print("\n" + "="*70)
        print("GENERATING DATASET VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.plot_function_distribution()
        self.plot_complexity_levels()
        self.plot_success_rates()
        self.plot_attempts_distribution()
        self.plot_code_length_stats()
        self.plot_function_type_by_level()
        self.plot_success_heatmap()
        self.plot_generation_timeline()
        self.plot_complexity_analysis()  # NEW
        
        self.generate_summary_table()
        self.generate_dataset_statistics()  # NEW
        
        print(f"\n✅ All visualizations saved to: {self.viz_dir}")
        print(f"📊 Generated 9 figures + 2 tables")
    
    def plot_function_distribution(self):
        """Figure 1: Distribution of function types"""
        plt.figure(figsize=(12, 6))
        
        # Count by type
        type_counts = self.df['function_type'].value_counts()
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = sns.color_palette('Set3', len(type_counts))
        wedges, texts, autotexts = ax1.pie(type_counts.values, labels=type_counts.index, 
                                            autopct='%1.1f%%', startangle=90, colors=colors)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        ax1.set_title('Function Type Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart with values
        bars = ax2.bar(range(len(type_counts)), type_counts.values, color=colors, 
                      edgecolor='black', linewidth=1)
        ax2.set_xticks(range(len(type_counts)))
        ax2.set_xticklabels(type_counts.index, rotation=45, ha='right')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Function Counts by Type', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig1_function_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig1_function_distribution.pdf', bbox_inches='tight')  # PDF for papers
        plt.close()
        print("✓ Figure 1: Function Distribution")
    
    def plot_complexity_levels(self):
        """Figure 2: Dataset distribution across complexity levels"""
        plt.figure(figsize=(10, 6))
        
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        level_counts = self.df['level'].value_counts().reindex(level_order)
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = plt.bar(range(len(level_counts)), level_counts.values, color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.xticks(range(len(level_counts)), 
                   [l.capitalize() for l in level_counts.index],
                   fontsize=12)
        plt.ylabel('Number of Functions', fontsize=12)
        plt.title('Dataset Distribution Across Curriculum Levels', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage annotations
        total = level_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, level_counts.values)):
            percentage = count / total * 100
            plt.text(bar.get_x() + bar.get_width()/2., count/2,
                    f'{percentage:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig2_complexity_levels.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig2_complexity_levels.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 2: Complexity Levels")
    
    def plot_success_rates(self):
        """Figure 3: Success rates by level and overall"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Overall success rate
        overall_success = self.df['success'].sum()
        overall_total = len(self.df)
        overall_rate = overall_success / overall_total * 100
        
        ax1.bar(['Successful', 'Failed'], 
                [overall_success, overall_total - overall_success],
                color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Overall Success Rate: {overall_rate:.1f}%', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate([overall_success, overall_total - overall_success]):
            ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # By level
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        success_by_level = []
        for level in level_order:
            level_data = self.df[self.df['level'] == level]
            success_rate = level_data['success'].sum() / len(level_data) * 100
            success_by_level.append(success_rate)
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax2.bar(range(len(level_order)), success_by_level, color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        for bar, rate in zip(bars, success_by_level):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_xticks(range(len(level_order)))
        ax2.set_xticklabels([l.capitalize() for l in level_order], fontsize=11)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('Success Rate by Curriculum Level', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig3_success_rates.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig3_success_rates.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 3: Success Rates")
    
    def plot_attempts_distribution(self):
        """Figure 4: Distribution of attempts needed"""
        plt.figure(figsize=(10, 6))
        
        attempts_data = self.df[self.df['success'] == True]['attempts']
        
        plt.hist(attempts_data, bins=range(1, attempts_data.max()+2), 
                 color='#3498db', edgecolor='black', alpha=0.7, linewidth=1.5)
        plt.xlabel('Number of Attempts', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Attempts for Successful Generations', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_attempts = attempts_data.mean()
        median_attempts = attempts_data.median()
        plt.axvline(mean_attempts, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_attempts:.2f}')
        plt.axvline(median_attempts, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_attempts:.0f}')
        
        # Add text box with statistics
        stats_text = f"n = {len(attempts_data)}\nMean = {mean_attempts:.2f}\nMedian = {median_attempts:.0f}\nMode = {attempts_data.mode()[0]}"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig4_attempts_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig4_attempts_distribution.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 4: Attempts Distribution")
    
    def plot_code_length_stats(self):
        """Figure 5: Code length statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter successful generations
        successful = self.df[self.df['success'] == True]
        
        # Histogram
        ax1.hist(successful['code_length'], bins=30, color='#9b59b6', 
                edgecolor='black', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Code Length (characters)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Generated Code Length', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        mean_length = successful['code_length'].mean()
        median_length = successful['code_length'].median()
        ax1.axvline(mean_length, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_length:.0f}')
        ax1.axvline(median_length, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_length:.0f}')
        ax1.legend()
        
        # Box plot by level
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        data_by_level = [successful[successful['level'] == level]['code_length'].values 
                        for level in level_order]
        
        bp = ax2.boxplot(data_by_level, labels=[l.capitalize() for l in level_order],
                        patch_artist=True, showmeans=True)
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        ax2.set_ylabel('Code Length (characters)', fontsize=12)
        ax2.set_title('Code Length by Curriculum Level', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig5_code_length.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig5_code_length.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 5: Code Length Statistics")
    
    def plot_function_type_by_level(self):
        """Figure 6: Function type distribution across levels"""
        plt.figure(figsize=(12, 8))
        
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        
        # Create cross-tabulation
        ct = pd.crosstab(self.df['level'], self.df['function_type'])
        ct = ct.reindex(level_order)
        
        # Stacked bar chart
        ax = ct.plot(kind='bar', stacked=True, colormap='Set3', 
                    edgecolor='black', linewidth=0.5, figsize=(12, 7))
        
        plt.xlabel('Curriculum Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Function Type Distribution Across Curriculum Levels', 
                 fontsize=14, fontweight='bold')
        plt.legend(title='Function Type', bbox_to_anchor=(1.05, 1), 
                  loc='upper left', frameon=True)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig6_type_by_level.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig6_type_by_level.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 6: Function Types by Level")
    
    def plot_success_heatmap(self):
        """Figure 7: Success rate heatmap (level x function type)"""
        plt.figure(figsize=(12, 8))
        
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        
        # Calculate success rates
        pivot_data = []
        for level in level_order:
            level_data = self.df[self.df['level'] == level]
            level_rates = []
            for ftype in sorted(self.df['function_type'].unique()):
                type_data = level_data[level_data['function_type'] == ftype]
                if len(type_data) > 0:
                    success_rate = type_data['success'].sum() / len(type_data) * 100
                else:
                    success_rate = np.nan  # Use NaN for missing data
                level_rates.append(success_rate)
            pivot_data.append(level_rates)
        
        pivot_df = pd.DataFrame(pivot_data, 
                               index=[l.capitalize() for l in level_order],
                               columns=sorted(self.df['function_type'].unique()))
        
        # Create heatmap
        mask = pivot_df.isna()
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Success Rate (%)'}, linewidths=0.5,
                   mask=mask, vmin=0, vmax=100, center=50)
        
        plt.title('Success Rate Heatmap: Level vs Function Type', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Function Type', fontsize=12)
        plt.ylabel('Curriculum Level', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig7_success_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig7_success_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 7: Success Heatmap")
    
    def plot_generation_timeline(self):
        """Figure 8: Generation timeline (if timestamps available)"""
        if 'timestamp' not in self.df.columns:
            print("⚠ Skipping timeline plot (no timestamps)")
            return
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        # Cumulative success over time
        self.df['cumulative_success'] = self.df['success'].cumsum()
        self.df['cumulative_total'] = range(1, len(self.df) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative plot
        ax1.plot(self.df['cumulative_total'], self.df['cumulative_success'], 
                color='#2ecc71', linewidth=2.5, label='Successful', marker='o', 
                markersize=2, markevery=20)
        ax1.plot(self.df['cumulative_total'], self.df['cumulative_total'], 
                color='#3498db', linewidth=2, linestyle='--', label='Total')
        ax1.fill_between(self.df['cumulative_total'], self.df['cumulative_success'], 
                        alpha=0.3, color='#2ecc71')
        
        ax1.set_xlabel('Number of Generations', fontsize=12)
        ax1.set_ylabel('Cumulative Count', fontsize=12)
        ax1.set_title('Cumulative Generation Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, frameon=True)
        ax1.grid(alpha=0.3)
        
        # Success rate over time (rolling average)
        window = 20
        self.df['rolling_success_rate'] = self.df['success'].rolling(window=window, min_periods=1).mean() * 100
        
        ax2.plot(self.df['cumulative_total'], self.df['rolling_success_rate'], 
                color='#e74c3c', linewidth=2.5)
        ax2.fill_between(self.df['cumulative_total'], self.df['rolling_success_rate'], 
                        alpha=0.3, color='#e74c3c')
        ax2.axhline(y=self.df['success'].mean() * 100, color='blue', linestyle='--', 
                   linewidth=2, label=f'Overall Mean: {self.df["success"].mean()*100:.1f}%')
        
        ax2.set_xlabel('Number of Generations', fontsize=12)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title(f'Rolling Success Rate (window={window})', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, frameon=True)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig8_generation_timeline.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig8_generation_timeline.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 8: Generation Timeline")
    
    def plot_complexity_analysis(self):
        """Figure 9: Complexity analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Complexity distribution
        successful = self.df[self.df['success'] == True]
        
        ax1.scatter(successful['complexity_score'], successful['attempts'],
                   c=successful['level'].map({'foundation': 0, 'conceptual': 1, 
                                             'application': 2, 'advanced': 3}),
                   cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Function Complexity Score', fontsize=12)
        ax1.set_ylabel('Attempts Required', fontsize=12)
        ax1.set_title('Complexity vs Generation Attempts', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(successful['complexity_score'], successful['attempts'], 1)
        p = np.poly1d(z)
        ax1.plot(successful['complexity_score'], p(successful['complexity_score']), 
                "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
        ax1.legend()
        
        # Complexity by level
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        complexity_by_level = [self.df[self.df['level'] == level]['complexity_score'].values 
                              for level in level_order]
        
        bp = ax2.boxplot(complexity_by_level, labels=[l.capitalize() for l in level_order],
                        patch_artist=True, showmeans=True)
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        ax2.set_ylabel('Complexity Score', fontsize=12)
        ax2.set_title('Function Complexity by Level', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fig9_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'fig9_complexity_analysis.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 9: Complexity Analysis")
    
    def generate_summary_table(self):
        """Generate LaTeX table for paper - Table 1"""
        level_order = ['foundation', 'conceptual', 'application', 'advanced']
        
        table_data = []
        for level in level_order:
            level_data = self.df[self.df['level'] == level]
            total = len(level_data)
            successful = level_data['success'].sum()
            success_rate = successful / total * 100
            avg_attempts = level_data[level_data['success'] == True]['attempts'].mean()
            avg_length = level_data[level_data['success'] == True]['code_length'].mean()
            
            table_data.append({
                'Level': level.capitalize(),
                'Total': total,
                'Successful': successful,
                'Success Rate (\\%)': f'{success_rate:.1f}',
                'Avg Attempts': f'{avg_attempts:.2f}',
                'Avg Code Length': f'{int(avg_length)}'
            })
        
        # Add overall row
        total = len(self.df)
        successful = self.df['success'].sum()
        success_rate = successful / total * 100
        avg_attempts = self.df[self.df['success'] == True]['attempts'].mean()
        avg_length = self.df[self.df['success'] == True]['code_length'].mean()
        
        table_data.append({
            'Level': '\\textbf{Overall}',
            'Total': f'\\textbf{{{total}}}',
            'Successful': f'\\textbf{{{successful}}}',
            'Success Rate (\\%)': f'\\textbf{{{success_rate:.1f}}}',
            'Avg Attempts': f'\\textbf{{{avg_attempts:.2f}}}',
            'Avg Code Length': f'\\textbf{{{int(avg_length)}}}'
        })
        
        df_table = pd.DataFrame(table_data)
        
        # Save as LaTeX with better formatting
        latex_str = df_table.to_latex(index=False, escape=False, column_format='lrrrrr')
        
        # Add table environment
        latex_full = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Dataset Generation Statistics by Curriculum Level}}
\\label{{tab:dataset_statistics}}
{latex_str}
\\end{{table}}
"""
        
        with open(self.viz_dir / 'table1_summary_statistics.tex', 'w') as f:
            f.write(latex_full)
        
        # Save as CSV
        df_table.to_csv(self.viz_dir / 'table1_summary_statistics.csv', index=False)
        
        print("✓ Table 1: Summary Statistics")
    
    def generate_dataset_statistics(self):
        """Generate additional statistics table - Table 2"""
        successful = self.df[self.df['success'] == True]
        
        stats_data = {
            'Metric': [
                'Total Functions',
                'Successfully Generated',
                'Failed Generations',
                'Overall Success Rate',
                'Mean Attempts (Successful)',
                'Median Attempts (Successful)',
                'Mean Code Length',
                'Median Code Length',
                'Std Dev Code Length',
                'Most Common Function Type',
                'Least Common Function Type',
            ],
            'Value': [
                len(self.df),
                successful.shape[0],
                len(self.df) - successful.shape[0],
                f"{successful.shape[0] / len(self.df) * 100:.2f}\\%",
                f"{successful['attempts'].mean():.2f}",
                f"{successful['attempts'].median():.0f}",
                f"{successful['code_length'].mean():.0f}",
                f"{successful['code_length'].median():.0f}",
                f"{successful['code_length'].std():.0f}",
                self.df['function_type'].value_counts().index[0],
                self.df['function_type'].value_counts().index[-1],
            ]
        }
        
        df_stats = pd.DataFrame(stats_data)
        
        # Save as LaTeX
        latex_str = df_stats.to_latex(index=False, escape=False, column_format='lr')
        latex_full = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Comprehensive Dataset Statistics}}
\\label{{tab:dataset_stats}}
{latex_str}
\\end{{table}}
"""
        
        with open(self.viz_dir / 'table2_dataset_statistics.tex', 'w') as f:
            f.write(latex_full)
        
        df_stats.to_csv(self.viz_dir / 'table2_dataset_statistics.csv', index=False)
        
        print("✓ Table 2: Dataset Statistics")

def main():
    print("\n" + "="*70)
    print("DERIVATIVE DATASET VISUALIZATION TOOL")
    print("="*70)
    
    # Check if dataset exists
    dataset_dir = Path("derivative_dataset_537")
    if not dataset_dir.exists():
        print(f"\n❌ Dataset directory not found: {dataset_dir}")
        print("Please run data_generation_pipeline.py first")
        return
    
    visualizer = DatasetVisualizer(str(dataset_dir))
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\n📁 Output directory: {visualizer.viz_dir}")
    print("\n📊 Generated files:")
    print("  • fig1_function_distribution.png/pdf")
    print("  • fig2_complexity_levels.png/pdf")
    print("  • fig3_success_rates.png/pdf")
    print("  • fig4_attempts_distribution.png/pdf")
    print("  • fig5_code_length.png/pdf")
    print("  • fig6_type_by_level.png/pdf")
    print("  • fig7_success_heatmap.png/pdf")
    print("  • fig8_generation_timeline.png/pdf")
    print("  • fig9_complexity_analysis.png/pdf")
    print("  • table1_summary_statistics.tex/csv")
    print("  • table2_dataset_statistics.tex/csv")
    
    print("\n✅ Ready for paper inclusion!")
    print("\n💡 LaTeX Usage:")
    print("   \\input{visualizations/table1_summary_statistics.tex}")
    print("   \\includegraphics[width=0.8\\textwidth]{visualizations/fig1_function_distribution.pdf}")

if __name__ == "__main__":
    main()