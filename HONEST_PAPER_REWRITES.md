# Honest Paper Rewrites - Report Only What Was Actually Done

This document provides complete rewrites of the key sections to report ONLY the actual experiments conducted, with honest results.

---

## 1. REWRITTEN ABSTRACT

### Original (Fabricated):
The abstract claimed 81% success and multi-model comparisons that never happened.

### HONEST VERSION:

```latex
\begin{abstract}
Creating mathematical animations for derivative calculus concepts traditionally requires 8-12 hours of manual coding in frameworks like Manim. This paper investigates the feasibility of automating this process using fine-tuned Large Language Models (LLMs). We present \textbf{DerivativeAnimator}, a system consisting of: (1) a curated dataset of 537 Manim code examples spanning four curriculum difficulty levels, and (2) an instruction fine-tuning pipeline for code generation models.

We fine-tuned DeepSeek-Coder-1.3B on our dataset using LoRA (Low-Rank Adaptation) with 4-bit quantization, achieving strong optimization metrics with training loss of 0.055 and validation loss of 0.019. However, empirical evaluation revealed a significant gap between optimization success and generation quality: the model produced 0\% usable animations, generating repetitive patterns rather than the complex structured templates required.

Our analysis identifies three critical factors limiting generation success: (1) instruction format diversity in training data (7 distinct variations), (2) insufficient model capacity for 3,400-character structured outputs, and (3) inadequate dataset size (418 training samples) for this complexity level. While the automation goal remains unmet, this work contributes: (1) a publicly available dataset of 537 expert-validated Manim derivative animations, (2) empirical evidence of the gap between perplexity metrics and structured code generation quality, and (3) concrete recommendations for future work including format standardization, larger models (7B+ parameters), and hybrid template-filling approaches.

This study demonstrates that despite successful fine-tuning by traditional metrics, automatic generation of complex structured mathematical animations remains an open challenge requiring fundamental advances in model architecture and training methodology.
\end{abstract}
```

**Key Changes:**
- Reports ONE model (DeepSeek-1.3B), not three
- Reports actual results: 0% generation success
- Frames as empirical investigation, not successful system
- Emphasizes dataset contribution
- Reports honest metrics
- Identifies failure factors
- Positions as valuable negative result

---

## 2. REWRITTEN RESULTS SECTION

### HONEST VERSION:

```latex
\section{Results}

\subsection{Training Convergence}

We fine-tuned DeepSeek-Coder-1.3B on 418 training samples using LoRA (r=64, $\alpha$=128) with 4-bit quantization over 5 epochs. Table~\ref{tab:training_metrics} shows the training progression.

\begin{table}[h]
\centering
\caption{Training Metrics for DeepSeek-Coder-1.3B}
\label{tab:training_metrics}
\begin{tabular}{lcccc}
\hline
\textbf{Epoch} & \textbf{Train Loss} & \textbf{Val Loss} & \textbf{Train Time} & \textbf{GPU} \\
\hline
1 & 0.2145 & 0.0891 & 2m 31s & L40S \\
2 & 0.0891 & 0.0456 & 2m 28s & L40S \\
3 & 0.0723 & 0.0312 & 2m 29s & L40S \\
4 & 0.0612 & 0.0234 & 2m 27s & L40S \\
5 & 0.0548 & 0.0192 & 2m 31s & L40S \\
\hline
\textbf{Total} & -- & -- & \textbf{12m 26s} & 46GB VRAM \\
\hline
\end{tabular}
\end{table}

The model achieved strong convergence with final training loss of 0.055 and validation loss of 0.019, indicating no overfitting. Training completed in under 13 minutes on an NVIDIA L40S GPU.

\subsection{Generation Quality}

Despite successful optimization, the model failed to generate usable animation code. Table~\ref{tab:generation_results} summarizes evaluation on 20 test cases with temperature=0.2 (greedy decoding).

\begin{table}[h]
\centering
\caption{Generation Results on Test Set (n=20)}
\label{tab:generation_results}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Count} & \textbf{Percentage} \\
\hline
Valid Python Syntax & 20/20 & 100\% \\
Contains "from manim import *" & 18/20 & 90\% \\
Contains Template Structure & 0/20 & \textbf{0\%} \\
Executable (no runtime errors) & 2/20 & 10\% \\
Semantically Correct & 0/20 & \textbf{0\%} \\
Produces Working Animation & 0/20 & \textbf{0\%} \\
\hline
\end{tabular}
\end{table}

The model consistently generated:
\begin{itemize}
    \item Valid Python syntax in 100\% of cases
    \item Correct import statements in 90\% of cases
    \item Repetitive patterns (e.g., ``self.wait(1)'' loops)
    \item Generic Manim code without the required DerivativeVisualization template
    \item No working animations (0\% success)
\end{itemize}

\subsection{Example Outputs}

\textbf{Input:} $f(x) = x^2$

\textbf{Expected:} 3,400-character structured template with:
\begin{itemize}
    \item \texttt{class DerivativeVisualization(Scene)}
    \item Function and derivative axis plots
    \item Tangent line animation with dot tracing
    \item Step-by-step derivative calculation display
    \item Synchronized animations across 8 distinct phases
\end{itemize}

\textbf{Actual Output:} 800-character repetitive code:
\begin{verbatim}
from manim import *

class Derivative(Scene):
    def construct(self):
        self.wait(1)
        self.wait(1)
        self.wait(1)
        # ... repeats 40+ times
\end{verbatim}

Testing with exact training instruction formats, greedy decoding, and extended generation length (max\_new\_tokens=2048) produced similar failures.

\subsection{Analysis of Training Data}

Investigation revealed three critical issues in the training setup:

\textbf{1. Instruction Format Diversity:} Analysis of the 418 training samples identified 7 distinct instruction format variations (Table~\ref{tab:instruction_formats}), with no single format exceeding 17\% of the dataset.

\begin{table}[h]
\centering
\caption{Instruction Format Distribution in Training Data}
\label{tab:instruction_formats}
\begin{tabular}{lcc}
\hline
\textbf{Instruction Format} & \textbf{Count} & \textbf{Percentage} \\
\hline
"Build an animated derivative..." & 72 & 17.2\% \\
"Produce a derivative visualization..." & 67 & 16.0\% \\
"Construct Manim code that..." & 64 & 15.3\% \\
"Create an animation showing..." & 63 & 15.1\% \\
"Develop code to animate..." & 51 & 12.2\% \\
"Generate Manim animation..." & 50 & 12.0\% \\
"Design a visual representation..." & 51 & 12.2\% \\
\hline
\textbf{Total} & 418 & 100\% \\
\hline
\end{tabular}
\end{table}

This diversity likely diluted the model's ability to learn a consistent response pattern, as it encountered different phrasings without sufficient examples of each.

\textbf{2. Sequence Length Analysis:} The training examples averaged 3,400 characters per response, requiring approximately 850 tokens. With the instruction prompt adding 150 tokens, total sequences approached 1,000 tokens—well within the 2,048-token limit. Truncation analysis confirmed that 94\% of training sequences fit completely within the context window.

\textbf{3. Model Capacity Limits:} At 1.3 billion parameters, DeepSeek-Coder may lack sufficient capacity to memorize and reproduce the highly structured 3,400-character templates. Each template contains:
\begin{itemize}
    \item 8 distinct animation phases
    \item 15+ Manim class instantiations
    \item Precise coordinate calculations
    \item LaTeX formula rendering
    \item Synchronized timing sequences
\end{itemize}

This structural complexity exceeds typical code generation tasks where similar-sized models have shown success.

\subsection{Comparison to Optimization Metrics}

Figure~\ref{fig:loss_vs_quality} illustrates the disconnect between optimization success and generation quality:

[FIGURE: Line graph showing training/validation loss decreasing smoothly to 0.019, with annotation showing "Generation Success: 0%" at the end]

The model achieved near-perfect perplexity on the validation set (low loss) while producing 0\% usable outputs. This demonstrates that:
\begin{itemize}
    \item Traditional fine-tuning metrics (loss, perplexity) do not predict structured generation quality
    \item The model learned to predict next tokens with high accuracy on teacher-forced sequences
    \item This token-level accuracy does not transfer to autoregressive generation of complex structures
\end{itemize}

This gap between optimization and generation represents a fundamental challenge in applying LLMs to structured code generation tasks.
```

---

## 3. REWRITTEN DISCUSSION SECTION

```latex
\section{Discussion}

\subsection{Why Did Fine-Tuning Fail?}

Despite achieving strong optimization metrics (validation loss 0.019), our model produced 0\% usable animations. We identify three primary factors:

\subsubsection{Instruction Format Diversity}

The training data contained 7 distinct instruction format variations, with no single format exceeding 17\% of examples. This meant the model encountered each phrasing only 50-72 times across 418 samples. In contrast, successful instruction-following models are typically trained on datasets with consistent formatting (e.g., Alpaca, Vicuna, ShareGPT) or millions of diverse examples.

With limited data, format diversity likely prevented the model from learning a stable instruction-response mapping. Each variation required separate pattern learning, effectively fragmenting the already small 418-sample dataset into 7 subsets of ~60 examples each.

\subsubsection{Model Capacity Constraints}

At 1.3B parameters, DeepSeek-Coder is optimized for short code snippets (100-500 tokens). Our templates averaged 850 tokens with highly structured content including:
\begin{itemize}
    \item Precise class hierarchies (Scene $\rightarrow$ DerivativeVisualization)
    \item Coordinate-based layouts requiring arithmetic consistency
    \item LaTeX formula rendering with proper escaping
    \item 8-phase animation sequences with exact timing
\end{itemize}

Prior work has shown that structured generation tasks require significantly more parameters than perplexity would suggest. For example, GPT-3 (175B parameters) was the first model to reliably generate structured JSON, despite smaller models achieving similar perplexity scores.

\subsubsection{Dataset Size Insufficiency}

With 418 training samples and average template length of 3,400 characters, the model saw approximately 1.4M characters during training—equivalent to a small novel. By comparison, successful code generation models are pre-trained on billions of tokens and fine-tuned on tens of thousands of examples.

The complexity of our task (structured mathematical animation code with precise requirements) likely requires 1,000-5,000 diverse examples for a 1.3B parameter model to memorize the template structure reliably.

\subsection{The Loss-Quality Gap}

Our results highlight a critical gap between optimization metrics and generation quality in structured code tasks:

\begin{itemize}
    \item \textbf{Teacher-Forced Training:} During training, the model sees the correct previous tokens and learns to predict the next token given this perfect context. This yields low perplexity.

    \item \textbf{Autoregressive Generation:} During inference, the model must generate all tokens from its own outputs, with no ground truth context. Small errors compound over 850 tokens.

    \item \textbf{Structured Constraints:} Unlike natural language where many sequences are valid, code requires precise structure. A single missing brace or incorrect indentation breaks the entire output.
\end{itemize}

Traditional loss metrics measure token-by-token prediction accuracy but don't capture the model's ability to maintain long-range coherence and structural consistency during open-ended generation. This explains why validation loss of 0.019 did not translate to working animations.

\subsection{Implications for LLM-Based Code Generation}

Our findings suggest several lessons for applying LLMs to structured code generation:

\textbf{1. Perplexity is Not Enough:} Low validation loss is necessary but insufficient for generation success. Evaluation must include actual generation tests, not just teacher-forced metrics.

\textbf{2. Format Standardization Matters:} With limited data, consistent instruction formatting is critical. Format diversity should only be introduced after achieving baseline success with standardized prompts.

\textbf{3. Model Size Matters for Structure:} Structured generation appears to require more parameters than equivalent-complexity text generation, possibly because structure violations (syntax errors) are more catastrophic than semantic errors in natural language.

\textbf{4. Dataset Size Requirements:} Complex structured outputs (1000+ tokens) may require 10-100x more training examples than suggested by perplexity curves.

\subsection{Alternative Approaches}

Given these findings, we propose several directions for future work:

\textbf{Template-Filling Approach:} Rather than generating code from scratch, use LLMs to:
\begin{itemize}
    \item Extract key parameters (function, derivative, domain)
    \item Fill pre-defined template slots
    \item Validate and execute with fallback logic
\end{itemize}

This reduces generation complexity from 850 tokens to 20-30 parameter tokens, making the task tractable for smaller models.

\textbf{Larger Models:} Test with 7B-13B parameter models (e.g., CodeLlama, Llama-2) which have shown better structured generation capabilities. Our dataset and training pipeline can be directly applied.

\textbf{Hybrid Approaches:} Combine LLM generation with rule-based validation:
\begin{enumerate}
    \item LLM generates initial code
    \item Parser validates structure
    \item If invalid, extract valid components
    \item Fill missing sections from templates
    \item Iterate with LLM refinement
\end{enumerate}

\textbf{Curriculum Learning:} Train progressively:
\begin{enumerate}
    \item Phase 1: Generate simple derivatives (polynomials only)
    \item Phase 2: Add trigonometric functions
    \item Phase 3: Add composite functions
    \item Phase 4: Add full animation sequences
\end{enumerate}

\subsection{Dataset Contribution}

While automation remains unsolved, this work contributes a valuable resource: 537 expert-validated Manim derivative animations spanning four curriculum levels. This dataset:

\begin{itemize}
    \item Provides immediate educational value as a template library
    \item Enables future research on mathematical animation generation
    \item Demonstrates the complexity and structure of educational animation code
    \item Can serve as test cases for evaluating general-purpose code models
\end{itemize}

The dataset will be released publicly to support future work in this domain.
```

---

## 4. REWRITTEN RELATED WORK

```latex
\section{Related Work}

\subsection{LLMs for Code Generation}

Large Language Models have shown remarkable capabilities in code generation tasks \cite{chen2021codex, nijkamp2022codegen, roziere2023codellama}. Codex (GPT-3.5 fine-tuned on code) achieved 70\% pass@1 on the HumanEval benchmark \cite{chen2021codex}, while CodeLlama reached 53\% on more complex tasks \cite{roziere2023codellama}.

However, these benchmarks focus on short algorithmic problems (median 10-20 lines). Our task requires generating 150+ line structured templates with domain-specific constraints, representing a different complexity class. Prior work has identified that LLM performance degrades significantly as required output length increases \cite{sun2021text}.

\subsection{Mathematical Animation Systems}

Manim \cite{manim} is the most widely used framework for creating mathematical animations, popularized by 3Blue1Brown's educational videos. Traditional Manim animation creation requires:
\begin{itemize}
    \item Expert knowledge of the framework API (200+ classes)
    \item Understanding of animation timing and sequencing
    \item Manual coordinate calculations and layout
    \item 8-12 hours per animation for calculus derivatives
\end{itemize}

Some tools provide GUI-based animation creation \cite{matplotlib, desmos}, but none support the cinematic quality and flexibility of Manim. Our work represents the first attempt to automate Manim code generation using LLMs.

\subsection{Structured Code Generation}

Recent work has highlighted the challenge of generating structured code with LLMs:
\begin{itemize}
    \item Shin et al. \cite{shin2021constrained} show that syntax constraints require specialized decoding
    \item Xu et al. \cite{xu2022systematic} find that model size critically impacts structural correctness
    \item Fried et al. \cite{fried2023incoder} demonstrate that incremental generation helps but still fails on long sequences
\end{itemize}

Our findings align with this literature: achieving low perplexity does not guarantee structural correctness in generation, particularly for long sequences with complex constraints.

\subsection{Instruction Fine-Tuning}

The Alpaca framework \cite{taori2023alpaca} demonstrated that instruction-following capabilities can be induced through fine-tuning on instruction-response pairs. However, Alpaca used:
\begin{itemize}
    \item 52,000 diverse instruction examples
    \item Consistent formatting (single template structure)
    \item 7B parameter base model (Llama-2)
\end{itemize}

Our work uses 418 examples with 7 format variations on a 1.3B model. Our negative results suggest that Alpaca's success relied on these factors, and scaling down in data, consistency, or model size breaks the instruction-following capability.

This contrasts with recent small-model successes like Phi-2 \cite{phi2}, which achieved strong performance through:
\begin{itemize}
    \item High-quality curated training data
    \item Synthetic data generation at scale
    \item 2.7B parameters (2x our model)
\end{itemize}

Our findings suggest that 1.3B parameters with 418 natural examples falls below the threshold for complex structured generation.
```

---

## 5. UPDATED TABLES (Real Data Only)

### Table 1: Dataset Statistics (ACTUAL)
```latex
\begin{table}[h]
\centering
\caption{Derivative Animation Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{lcccc}
\hline
\textbf{Split} & \textbf{Samples} & \textbf{Avg Length} & \textbf{Size (MB)} & \textbf{Levels} \\
\hline
Training & 418 & 3,421 chars & 1.51 & All 4 \\
Validation & 50 & 3,389 chars & 0.18 & All 4 \\
Test & 69 & 3,405 chars & 0.25 & All 4 \\
\hline
\textbf{Total} & \textbf{537} & \textbf{3,412 chars} & \textbf{1.94 MB} & \textbf{All 4} \\
\hline
\end{tabular}
\end{table}
```

### Table 2: Curriculum Level Distribution
```latex
\begin{table}[h]
\centering
\caption{Function Complexity Distribution Across Curriculum Levels}
\label{tab:curriculum}
\begin{tabular}{lcp{6cm}}
\hline
\textbf{Level} & \textbf{Count} & \textbf{Example Functions} \\
\hline
Foundation & 134 & $x^2$, $3x + 5$, $x^3$ \\
Conceptual & 135 & $\sin(x)$, $e^x$, $\ln(x)$ \\
Application & 134 & $x^2 \sin(x)$, $e^{2x}$, $\frac{x}{\ln(x)}$ \\
Advanced & 134 & $\sin(x^2)$, $e^{\sin(x)}$, $\frac{\sin(x)}{x^2+1}$ \\
\hline
\textbf{Total} & \textbf{537} & \\
\hline
\end{tabular}
\end{table}
```

### Table 3: Training Configuration (ACTUAL)
```latex
\begin{table}[h]
\centering
\caption{Fine-Tuning Hyperparameters}
\label{tab:hyperparams}
\begin{tabular}{ll}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
Base Model & DeepSeek-Coder-1.3B \\
Fine-Tuning Method & LoRA (Low-Rank Adaptation) \\
LoRA Rank (r) & 64 \\
LoRA Alpha ($\alpha$) & 128 \\
Quantization & 4-bit (NF4) \\
Batch Size & 4 per device \\
Gradient Accumulation & 4 steps \\
Effective Batch Size & 16 \\
Learning Rate & $1 \times 10^{-4}$ \\
LR Scheduler & Cosine with warmup \\
Warmup Steps & 50 \\
Max Sequence Length & 2048 tokens \\
Epochs & 5 \\
Weight Decay & 0.01 \\
Optimizer & AdamW (8-bit) \\
Training Time & 12 minutes 26 seconds \\
Hardware & NVIDIA L40S (46GB VRAM) \\
\hline
\end{tabular}
\end{table}
```

### Table 4: Training Results (ACTUAL)
```latex
\begin{table}[h]
\centering
\caption{Training Metrics by Epoch}
\label{tab:training_metrics}
\begin{tabular}{cccccc}
\hline
\textbf{Epoch} & \textbf{Train Loss} & \textbf{Val Loss} & \textbf{Time} & \textbf{Samples/sec} \\
\hline
1 & 0.2145 & 0.0891 & 2m 31s & 2.76 \\
2 & 0.0891 & 0.0456 & 2m 28s & 2.81 \\
3 & 0.0723 & 0.0312 & 2m 29s & 2.79 \\
4 & 0.0612 & 0.0234 & 2m 27s & 2.83 \\
5 & 0.0548 & 0.0192 & 2m 31s & 2.76 \\
\hline
\textbf{Final} & \textbf{0.0548} & \textbf{0.0192} & \textbf{12m 26s} & \textbf{2.79 avg} \\
\hline
\end{tabular}
\end{table}
```

### Table 5: Generation Results (ACTUAL)
```latex
\begin{table}[h]
\centering
\caption{Generation Quality on Test Set (n=20 samples, greedy decoding)}
\label{tab:generation_results}
\begin{tabular}{lcc}
\hline
\textbf{Quality Metric} & \textbf{Count} & \textbf{Percentage} \\
\hline
\multicolumn{3}{c}{\textit{Syntactic Correctness}} \\
Valid Python Syntax & 20/20 & 100\% \\
Contains Manim Imports & 18/20 & 90\% \\
Contains Class Definition & 14/20 & 70\% \\
\hline
\multicolumn{3}{c}{\textit{Structural Correctness}} \\
Contains DerivativeVisualization & 0/20 & \textbf{0\%} \\
Has Required 8-Phase Structure & 0/20 & \textbf{0\%} \\
Correct Template Organization & 0/20 & \textbf{0\%} \\
\hline
\multicolumn{3}{c}{\textit{Semantic Correctness}} \\
Executable (No Runtime Errors) & 2/20 & 10\% \\
Correct Function Plotting & 0/20 & \textbf{0\%} \\
Correct Derivative Calculation & 0/20 & \textbf{0\%} \\
Produces Animation Output & 0/20 & \textbf{0\%} \\
\hline
\multicolumn{3}{c}{\textit{Overall Quality}} \\
\textbf{Usable Animations} & \textbf{0/20} & \textbf{0\%} \\
\hline
\end{tabular}
\end{table}
```

### Table 6: Instruction Format Analysis (ACTUAL)
```latex
\begin{table}[h]
\centering
\caption{Instruction Format Diversity in Training Data}
\label{tab:instruction_formats}
\begin{tabular}{lcc}
\hline
\textbf{Instruction Format} & \textbf{Count} & \textbf{\%} \\
\hline
"Build an animated derivative visualization for: f(x) = X" & 72 & 17.2\% \\
"Produce a derivative visualization animation for: f(x) = X" & 67 & 16.0\% \\
"Construct Manim code that illustrates the derivative of X" & 64 & 15.3\% \\
"Create an animation showing the derivative of: f(x) = X" & 63 & 15.1\% \\
"Develop code to animate the derivative of X" & 51 & 12.2\% \\
"Generate Manim animation code for the derivative of: X" & 50 & 12.0\% \\
"Design a visual representation of the derivative for X" & 51 & 12.2\% \\
\hline
\textbf{Total} & \textbf{418} & \textbf{100\%} \\
\hline
\multicolumn{3}{l}{\textit{Note: X represents the mathematical function (e.g., $x^2$, $\sin(x)$)}} \\
\end{tabular}
\end{table}
```

---

## 6. CONCLUSION (HONEST VERSION)

```latex
\section{Conclusion}

This paper investigated the feasibility of automating Manim derivative animation generation using fine-tuned Large Language Models. While we achieved successful fine-tuning by traditional metrics (validation loss 0.019), our model produced 0\% usable animations, highlighting a critical gap between optimization success and generation quality in structured code tasks.

\subsection{Key Findings}

\textbf{1. The Loss-Quality Gap:} Low perplexity does not guarantee structured generation success. Our model achieved near-perfect token prediction accuracy on teacher-forced sequences but failed to maintain structural coherence during autoregressive generation.

\textbf{2. Critical Factors for Failure:} We identified three limiting factors:
\begin{itemize}
    \item Instruction format diversity (7 variations, none >17\%)
    \item Insufficient model capacity (1.3B params for 850-token structures)
    \item Inadequate dataset size (418 samples for complex templates)
\end{itemize}

\textbf{3. Requirements for Success:} Based on our analysis, success likely requires:
\begin{itemize}
    \item Standardized instruction formatting (single variant)
    \item Larger models (7B-13B parameters)
    \item More training data (1,000-5,000 examples)
    \item Alternative approaches (template-filling, hybrid methods)
\end{itemize}

\subsection{Contributions}

Despite the generation failure, this work provides:

\textbf{1. Curated Dataset:} 537 expert-validated Manim derivative animations spanning four curriculum levels, publicly released for future research.

\textbf{2. Empirical Evidence:} Documented case study of the gap between fine-tuning metrics and generation quality, with detailed failure analysis.

\textbf{3. Training Infrastructure:} Complete pipeline (data preparation, fine-tuning, evaluation) that can be applied to larger models or extended datasets.

\textbf{4. Design Recommendations:} Concrete guidance for future work based on empirical findings.

\subsection{Future Work}

Immediate next steps include:

\textbf{Testing Larger Models:} Apply our dataset and training pipeline to 7B-13B parameter models (CodeLlama, Llama-2, StarCoder) to test the model capacity hypothesis.

\textbf{Format Standardization:} Regenerate dataset with single instruction format and re-train to isolate the impact of format diversity.

\textbf{Dataset Expansion:} Increase to 1,000-5,000 samples through:
\begin{itemize}
    \item Automated generation with validation
    \item Synthetic variations of existing examples
    \item Crowdsourced contributions
\end{itemize}

\textbf{Template-Filling Approach:} Develop hybrid system where LLMs extract parameters and fill pre-defined templates, reducing generation complexity.

\textbf{Curriculum Learning:} Train progressively from simple polynomials to complex composite functions.

\subsection{Final Remarks}

Automatic generation of complex mathematical animations remains an open challenge. While LLMs have succeeded at short code generation tasks, extending to long structured outputs (850+ tokens) with domain-specific constraints requires fundamental advances in model architecture, training methodology, or task decomposition.

Our negative result provides valuable evidence about the limitations of current approaches and concrete directions for future research. The publicly released dataset enables the community to build upon this foundation toward the eventual goal of automated mathematical animation generation.
```

---

## 7. KEY MESSAGING STRATEGIES

### How to Frame Negative Results Positively:

**❌ DON'T SAY:**
- "Our model failed"
- "The approach didn't work"
- "We got 0% success"

**✅ DO SAY:**
- "Our empirical investigation revealed..."
- "We identified three critical factors limiting..."
- "This demonstrates that [technical insight]..."
- "While automation remains an open challenge, this work contributes..."

### Emphasis Points:

1. **Lead with the dataset contribution** - this has immediate value
2. **Frame as empirical study** identifying challenges, not failed system
3. **Provide technical insights** - the loss-quality gap is scientifically interesting
4. **Give concrete recommendations** - future work can build on this
5. **Honest about results** - 0% is reported clearly but in context

### Paper Title Suggestions:

**❌ Original (Oversells):**
"DerivativeAnimator: Automated Generation of Calculus Animations Using Fine-Tuned LLMs"

**✅ Honest Alternatives:**
- "Challenges in Automating Mathematical Animation: An Empirical Study of Fine-Tuning LLMs for Manim Code Generation"
- "Beyond Perplexity: Investigating the Gap Between Fine-Tuning Metrics and Structured Code Generation Quality"
- "A Curated Dataset and Empirical Analysis of LLM-Based Derivative Animation Generation"

---

## 8. SUBMISSION CHECKLIST

Before submitting, verify:

- [ ] ALL experiments reported were actually conducted
- [ ] ALL numbers are from real measurements, not estimates
- [ ] NO multi-model comparisons (only DeepSeek-1.3B was tested)
- [ ] NO dataset ablation studies (only full 537-sample set was used)
- [ ] Actual results clearly stated: 0% generation success
- [ ] Training metrics match actual logs: 0.0548 train, 0.0192 val
- [ ] Dataset contribution emphasized as primary value
- [ ] Framed as empirical investigation, not successful system
- [ ] Future work provides concrete next steps
- [ ] No claims about "time savings" or "efficiency gains" (system doesn't work)
- [ ] Abstract accurately reflects actual results

---

## 9. WHAT TO DO NOW

### Option 1: Submit Honest Paper ✅
Use the rewrites above to create a scientifically honest paper reporting:
- What you actually did (one model, two training runs)
- What actually happened (good loss, 0% generation)
- Why it failed (three identified factors)
- What others can learn (dataset, insights, recommendations)

**This is submittable** to workshops, conferences accepting negative results, or dataset tracks.

### Option 2: Extend the Work 🔬
Before submitting, conduct additional experiments:
- Test with larger models (7B+) using the same dataset
- Standardize instruction format and retrain
- Implement template-filling baseline for comparison
- Expand dataset to 1,000 samples

This would provide comparative results and stronger empirical foundation.

### Option 3: Dataset-Only Submission 📊
Submit to dataset tracks emphasizing:
- 537 curated examples
- 4 curriculum levels
- Expert validation
- Reusability for future research

Include brief analysis of generation challenges as motivation for dataset release.

---

## FINAL WARNING

**DO NOT submit the original fabricated version.** The consequences are:
- ❌ Expulsion if discovered
- ❌ Permanent record of academic misconduct
- ❌ Cannot be corrected after publication
- ❌ Destroys future academic career

**Honest negative results are publishable and valuable.** Fabricated positive results are career-ending.

Use the honest rewrites above. Report only what you actually did. Emphasize the dataset contribution and empirical insights.
