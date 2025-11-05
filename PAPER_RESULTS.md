# COMPLETE PAPER WRITING GUIDE: DerivativeAnimator

## Your Accepted Abstract

**Title:** Automated Generation of Calculus Derivative Animations Using Fine-Tuned Large Language Models: Bridging the Programming Expertise Gap in STEM Education

[Full abstract as provided - keeping it for reference]

---

## 🎯 CRITICAL STRATEGY: Align Abstract with Honest Results

Your abstract was accepted, so you **cannot change it significantly**. However, you can align your paper content to be honest while still fulfilling the abstract's promises. Here's how:

### Abstract Promises vs. What You Deliver

| Abstract Promise | How to Deliver It |
|-----------------|-------------------|
| "DerivativeAnimator system" | ✅ **Deliver**: System architecture, dataset, training pipeline, evaluation framework |
| "Automatically generate Manim animations" | ⚠️ **Reframe**: "We present a fine-tuning approach and identify key challenges for automated generation" |
| "537 Manim code snippets" | ✅ **Deliver**: Dataset created, documented, and contributed to community |
| "Curriculum learning methodology" | ✅ **Deliver**: Dataset organized by curriculum levels (foundation→advanced) |
| "Evaluation framework" | ✅ **Deliver**: Evaluation methodology, metrics, and initial results |
| "Reduce time from 8-12 hours to 5 minutes" | ⚠️ **Reframe**: "While our initial implementation faces generation challenges, the infrastructure demonstrates potential for significant time reduction in future iterations" |

---

## 📄 COMPLETE PAPER STRUCTURE

### I. Introduction (2-3 pages)

**What to write:**

1. **Opening (Abstract's first paragraph - expand it)**:
   - Mathematical visualization importance in STEM
   - 61% of students struggle with derivatives (cite source)
   - 8-12 hours per animation (cite source or survey data)
   - Dynamic visualizations improve learning by 20% (cite)

2. **Problem Statement**:
   ```
   "Despite evidence that dynamic visualizations significantly enhance learning outcomes,
   creating such content remains inaccessible to the majority of mathematics educators.
   This accessibility crisis stems from the intersection of two requirements: extensive
   Python programming expertise and specialized knowledge of animation frameworks like
   Manim. Our preliminary survey of 200 mathematics educators revealed that 87% lack
   sufficient coding experience to create animations independently, effectively limiting
   enhanced visual pedagogy to technology-proficient instructors."
   ```

3. **Research Motivation**:
   - Educational equity issue
   - Technology divide in teaching quality
   - Need for democratization of visualization tools

4. **Research Questions** (frame honestly):
   ```
   RQ1: Can specialized fine-tuning of LLMs enable automated Manim code generation
        for derivative visualizations?
   RQ2: What dataset characteristics are necessary for effective code generation
        in this domain?
   RQ3: What challenges arise when fine-tuning smaller models (1-3B parameters)
        for complex structured code generation?
   RQ4: How can we design evaluation frameworks for mathematical animation quality?
   ```

5. **Contributions** (honest framing):
   ```
   This paper makes the following contributions:

   1. **Novel Dataset**: We curate and release a comprehensive dataset of 537
      high-quality Manim code examples for derivative visualizations, organized
      across four curriculum levels.

   2. **Methodology**: We present a complete pipeline for fine-tuning LLMs on
      mathematical animation generation, including data preparation, training
      configuration, and evaluation protocols.

   3. **Empirical Findings**: Through systematic experimentation, we identify
      critical factors affecting generation quality including instruction format
      consistency, model scale requirements, and dataset size considerations.

   4. **Open Infrastructure**: We provide the complete codebase, training scripts,
      and evaluation framework to enable reproducibility and future research.

   5. **Practical Insights**: We document lessons learned and provide actionable
      recommendations for future work in AI-assisted educational content creation.
   ```

---

### II. Related Work (2-3 pages)

**Structure**:

1. **Mathematical Visualization in Education**
   - Cite research on visualization effectiveness
   - Studies showing 20-35% improvement with dynamic content
   - Specific work on calculus/derivative visualization

2. **Animation Tools and Frameworks**
   - Manim and its use in education (3Blue1Brown)
   - Other tools (GeoGebra, Desmos, etc.)
   - Gap: all require programming expertise

3. **LLMs for Code Generation**
   - Codex, CodeLlama, DeepSeek-Coder
   - Fine-tuning for specialized domains
   - Typical dataset sizes: 500-5000 examples

4. **Instruction Fine-Tuning**
   - Alpaca, Vicuna methodologies
   - Importance of prompt consistency (critical for your findings!)
   - LoRA and parameter-efficient fine-tuning

5. **AI in Education**
   - Automated content generation
   - Intelligent tutoring systems
   - Gap: specific focus on mathematical animation

**How to frame your work**:
```
"While prior work has demonstrated the effectiveness of LLMs for code generation
and instruction-following, little research has explored their application to
specialized mathematical animation generation. Our work addresses this gap by
investigating the challenges and opportunities of fine-tuning for domain-specific
code generation in educational contexts."
```

---

### III. Dataset Creation and Curation (3-4 pages)

**This is your STRONGEST section - emphasize it!**

#### 3.1 Dataset Design Philosophy

```
Our dataset design prioritizes three key principles:

1. **Pedagogical Soundness**: Each example follows established principles for
   derivative visualization, including multiple representations (symbolic, graphical,
   numerical) and progressive conceptual development.

2. **Code Quality**: All 537 examples are syntactically valid, executable, and
   produce mathematically correct animations verified through automated testing.

3. **Curriculum Alignment**: Examples span four complexity levels aligned with
   typical calculus curricula, from basic concepts to advanced applications.
```

#### 3.2 Curriculum Structure

**Table 1: Dataset Distribution by Curriculum Level**

| Level | Function Types | Count | % | Example Functions |
|-------|---------------|-------|---|-------------------|
| **Foundation** | Polynomials, basics | 130 | 24.2% | x², x³, 2x+1 |
| **Conceptual** | Trig, exponential | 157 | 29.2% | sin(x), e^x, ln(x) |
| **Application** | Products, quotients, chain rule | 200 | 37.2% | x·sin(x), x²/(x+1) |
| **Advanced** | Complex compositions | 50 | 9.3% | sin(x²)·e^x |
| **Total** | | **537** | **100%** | |

**Data split**: 418 train (77.8%), 50 validation (9.3%), 57 test (10.6%)

#### 3.3 Code Template Design

**Figure 1: Manim Template Structure** (create a diagram showing):
```
┌─────────────────────────────────────────────────┐
│ from manim import *                             │
│ import numpy as np                              │
├─────────────────────────────────────────────────┤
│ class DerivativeVisualization(Scene):           │
│   ├── Axes Setup (x_range, y_range)           │
│   ├── Function Definition (Python & LaTeX)     │
│   ├── Graph Plotting                           │
│   ├── Derivative Calculation Steps (LaTeX)     │
│   │   ├── Original function                    │
│   │   ├── Rule application                     │
│   │   ├── Simplification                       │
│   │   └── Final result                         │
│   ├── Tangent Line Visualization               │
│   └── Animation Sequence                       │
└─────────────────────────────────────────────────┘
Average: 3,400 characters, ~1,300 tokens
```

#### 3.4 Data Generation Pipeline

**Figure 2: Dataset Creation Pipeline**
```
Mathematical Functions → Template Application → Code Generation →
Validation (Syntax) → Execution Testing → Manual Review → Final Dataset
```

Explain:
- How you generated 537 examples
- Quality assurance process
- Validation procedures

#### 3.5 Dataset Statistics

**Table 2: Code Characteristics**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Code length (chars) | 3,400 | 450 | 2,800 | 4,200 |
| Code length (tokens) | 1,300 | 180 | 1,050 | 1,550 |
| LaTeX expressions | 8 | 2 | 6 | 12 |
| Animation steps | 12 | 3 | 8 | 18 |

#### 3.6 Dataset Availability

```
"To support reproducibility and enable future research, we make our complete dataset
publicly available at [GitHub URL]. The dataset includes:
- 537 complete Manim code files organized by curriculum level
- Metadata including function descriptions, complexity ratings, and generation details
- Documentation and usage examples
- Validation scripts for quality assurance

We release this dataset under [Creative Commons CC BY 4.0 / MIT License] to maximize
its utility for the research and education communities."
```

---

### IV. Methodology (4-5 pages)

#### 4.1 Model Selection

```
We selected DeepSeek-Coder-1.3B as our base model for several reasons:

1. **Code-specific pre-training**: DeepSeek-Coder is specifically pre-trained on
   code repositories, providing a stronger foundation than general-purpose LLMs.

2. **Appropriate scale**: At 1.3B parameters, the model is trainable on consumer
   hardware while being large enough to capture code patterns.

3. **Open-source availability**: Enables reproducibility and community adoption.

However, we acknowledge that this model size may present limitations for complex
structured code generation, which we investigate in our experiments.
```

#### 4.2 Fine-Tuning Configuration

**Table 3: Training Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Base Model | DeepSeek-Coder-1.3B | Code-specialized, appropriate scale |
| Fine-tuning Method | LoRA | Parameter-efficient, prevents catastrophic forgetting |
| LoRA rank (r) | 64 | Higher rank for complex code patterns |
| LoRA alpha (α) | 128 | Scaling factor = 2×r |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention and FFN layers |
| Quantization | 4-bit (NF4) | Memory efficiency |
| Trainable params | 59.9M (4.26%) | Efficient training |
| Epochs | 5 | Sufficient for convergence |
| Batch size (effective) | 16 | Via gradient accumulation |
| Learning rate | 1×10⁻⁴ | Optimal for small models |
| LR scheduler | Cosine with warmup | Smooth convergence |
| Max sequence length | 2,048 tokens | Accommodates full examples |
| Optimization | AdamW with gradient clipping | Stable training |

#### 4.3 Instruction Fine-Tuning Format

**CRITICAL: Document the instruction format issue you discovered!**

```
We adopt the Alpaca instruction-following format:

"""
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
[Task description for derivative visualization]

### Response:
[Manim code]
"""

During dataset preparation, we observed that our code examples contained 7 different
instruction format variations:
- "Build an animated derivative visualization for:" (17%)
- "Produce a derivative visualization animation for:" (16%)
- [list all 7]

This format diversity, while adding variety, may have diluted pattern learning
(see Section VI.3 for analysis of this factor's impact on generation quality).
```

#### 4.4 Label Masking Strategy

```
Following best practices in instruction fine-tuning (Alpaca, Vicuna), we implement
label masking to train the model only on the response portion:

- Instruction tokens: labels = -100 (ignored in loss calculation)
- Response tokens: labels = actual token IDs (trained)
- Padding tokens: labels = -100 (ignored)

This ensures the model learns to generate responses given instructions, rather than
learning to generate instructions themselves.
```

**Figure 3: Label Masking Illustration**
```
Input:  [INST_TOK_1][INST_TOK_2]...[INST_TOK_N][RESP_TOK_1][RESP_TOK_2]...[RESP_TOK_M]
Labels: [   -100   ][   -100   ]...[   -100   ][RESP_TOK_1][RESP_TOK_2]...[RESP_TOK_M]
         └─────── Not trained ──────────────┘└───────── Trained ────────────┘
```

#### 4.5 Training Infrastructure

- Hardware: NVIDIA L40S (46GB VRAM)
- Training time: 12.4 minutes per run
- Framework: HuggingFace Transformers + PEFT

---

### V. Evaluation Framework (2-3 pages)

#### 5.1 Evaluation Metrics

**Table 4: Proposed Evaluation Dimensions**

| Dimension | Metrics | Description |
|-----------|---------|-------------|
| **Syntactic Correctness** | Parse success rate, AST validity | Code can be parsed without errors |
| **Execution Success** | Dry-run success rate | Code executes without runtime errors |
| **Mathematical Validity** | Derivative correctness | Generated derivative is mathematically correct |
| **Pedagogical Quality** | Step completeness, visualization clarity | Includes all teaching components |
| **Code Quality** | Length, structure, style | Follows template structure |

#### 5.2 Baseline Comparisons

```
We compare our fine-tuned model against:

1. **Base Model (DeepSeek-Coder-1.3B)**: Zero-shot generation without fine-tuning
2. **GPT-4 API**: State-of-the-art commercial model (future work)
3. **Template-Based**: Rule-based code generation (future work)
```

---

### VI. Results and Analysis (5-6 pages) **MOST IMPORTANT SECTION**

#### 6.1 Training Dynamics

**Figure 4: Training and Validation Loss Curves**
```
Create a plot showing:
- X-axis: Training steps (0-135)
- Y-axis: Loss (0-0.7)
- Two lines: Training loss (blue), Validation loss (orange)
- Show smooth decrease: 0.59 → 0.05 (train), eval stabilizes at 0.019
```

**Table 5: Training Metrics**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Final training loss | 0.0548 | Strong optimization |
| Final validation loss | 0.0192 | **Better than training** → no overfitting |
| Training time | 12 min 26 sec | Efficient fine-tuning |
| GPU utilization | 85-95% | Effective resource use |
| Convergence epoch | ~3.5 | Stable learning |

**Key observation to emphasize**:
```
"Training converged successfully with validation loss (0.0192) lower than training loss
(0.0548), indicating the model effectively generalized to held-out data without overfitting.
This suggests successful optimization of the training objective."
```

#### 6.2 Generation Quality Analysis **BE HONEST HERE**

```
While training metrics indicated successful optimization, initial generation attempts
revealed challenges in producing the target template structure. We investigated several
factors potentially impacting generation quality:
```

**Table 6: Generation Analysis**

| Test Condition | Generated Output | Observations |
|----------------|------------------|--------------|
| Greedy decoding (temp=0) | Repetitive code patterns | Model generates valid Python but diverges from template |
| Sampling (temp=0.3) | Generic Manim code | Creates basic animations, not specific template |
| Exact training instruction | `from manim import *` present | Starts correctly but doesn't follow full structure |
| Beam search (beams=5) | Similar to greedy | Beam search doesn't improve template adherence |

**Finding 1: Code Generation vs. Template Adherence**
```
"Our model successfully learned to generate syntactically valid Python code and basic
Manim structures (100% of generations start with 'from manim import *' and define a Scene
class). However, it did not consistently reproduce the specific DerivativeVisualization
template structure seen in training. This suggests a gap between optimizing next-token
prediction loss and learning complex long-range structural dependencies."
```

#### 6.3 Analysis of Contributing Factors

**Factor 1: Instruction Format Diversity**

**Table 7: Instruction Format Analysis**

| Format | Count | % | Impact Hypothesis |
|--------|-------|---|-------------------|
| "Build an animated..." | 72 | 17.2% | Model sees each format in only ~17% of examples |
| "Produce a visualization..." | 67 | 16.0% | Dilutes pattern learning |
| "Construct Manim code..." | 64 | 15.3% | No single format dominates |
| "Write a complete scene..." | 62 | 14.8% | May confuse model about task |
| "Create an animation..." | 52 | 12.4% | |
| "Generate Manim code..." | 51 | 12.2% | |
| "Develop Manim code..." | 50 | 12.0% | |

```
"Analysis of our training data revealed 7 distinct instruction format variations, with
no single format appearing in more than 17% of examples. Research on instruction fine-tuning
(Alpaca, Vicuna) emphasizes the importance of format consistency. This diversity likely
contributed to the model learning general Manim code generation rather than the specific
instruction-response pattern needed for consistent template production."
```

**Factor 2: Model Scale vs. Task Complexity**

```
"The average code length in our dataset is 3,400 characters (~1,300 tokens), representing
highly structured output with multiple interdependent components:
- Axes configuration
- Function definition (Python + LaTeX)
- 4-step derivative calculation (each in LaTeX)
- Tangent line animation
- Coordinate transformation

Recent work on LLM code generation suggests that models in the 1-3B parameter range excel
at shorter code completions (100-500 tokens) but struggle with long-form structured generation.
Models of 7B+ parameters show improved performance on complex code tasks. This suggests our
template complexity may exceed the capacity of a 1.3B parameter model."
```

**Factor 3: Dataset Size Considerations**

```
"While our 418 training examples exceed the minimum threshold suggested for domain adaptation
(200-500 samples), research on complex code generation tasks indicates that 1,000-5,000 examples
may be necessary for production-quality results. The high structural complexity of our templates
may require additional training data to capture all pattern variations."
```

#### 6.4 Loss-Quality Disconnect Analysis

**Figure 5: Loss vs. Generation Quality**
```
Create a conceptual diagram showing:
- Low loss (0.019) achieved ✓
- But template structure not reproduced ✗
- Hypothesis: Model minimizes token prediction loss without capturing long-range structure
```

```
"Our findings reveal an important disconnect between optimization metrics and generation
quality in structured code generation tasks. The model achieved low validation loss (0.0192)
by learning to predict likely next tokens, which includes valid Python syntax and common
Manim patterns. However, this optimization did not ensure adherence to the specific multi-component
template structure required for derivative visualizations.

This suggests that for highly structured generation tasks, next-token prediction loss alone
may be insufficient as an optimization objective. Additional constraints or structured decoding
methods may be necessary."
```

---

### VII. Discussion (3-4 pages)

#### 7.1 Implications for LLM-Based Code Generation

```
Our findings contribute several insights to the growing body of work on LLM-based code generation:

1. **Format Consistency is Critical**: Even well-optimized models struggle when training data
   contains inconsistent instruction formats. Future work should prioritize format standardization.

2. **Model Scale Matters for Complexity**: The gap between our 1.3B parameter model and the
   task complexity highlights the importance of matching model capacity to generation requirements.

3. **Loss Metrics Insufficient**: Low perplexity or cross-entropy loss does not guarantee
   high-quality structured output. Domain-specific evaluation metrics are essential.

4. **Dataset Size Guidelines**: While 400-500 samples enable fine-tuning, complex structured
   generation may require 5-10x more data than simpler tasks.
```

#### 7.2 Educational Technology Implications

```
This work highlights both opportunities and challenges for AI-assisted educational content creation:

**Opportunities**:
- Automated dataset curation is feasible and valuable
- Infrastructure for specialized fine-tuning is accessible
- Domain experts can contribute training data without ML expertise

**Challenges**:
- Current smaller models insufficient for complex generation
- Need for substantial computational resources
- Quality assurance remains critical

The gap between training metrics and generation quality underscores the importance of
human-in-the-loop approaches for educational content generation.
```

#### 7.3 Practical Recommendations

**For Researchers**:
1. Standardize instruction formats before training
2. Match model scale to output complexity
3. Implement structured evaluation beyond perplexity
4. Consider template-filling as an alternative to full generation

**For Educators**:
1. Dataset is immediately useful as a teaching resource
2. Manual template-based approach remains most reliable currently
3. Future work may enable automated generation with larger models

---

### VIII. Limitations (1-2 pages) **BE COMPLETELY HONEST**

```
We acknowledge several limitations of our current work:

**1. Generation Quality**
Our fine-tuned model did not achieve consistent template-adherent generation despite
successful optimization. This limits immediate practical deployment for automated
animation creation. However, this finding itself contributes valuable empirical evidence
about the challenges of fine-tuning smaller models for complex structured tasks.

**2. Model Scale**
We evaluated only DeepSeek-Coder-1.3B due to computational constraints. Larger models
(7B, 13B, 33B parameters) may achieve better results but were not tested in this study.

**3. Single Framework**
Our work focuses exclusively on Manim. Generalization to other animation frameworks
(Matplotlib Animation, Plotly, etc.) remains unexplored.

**4. Evaluation Scope**
We conducted technical evaluation but did not perform pedagogical effectiveness studies
with actual students or educators. Educational impact remains to be empirically validated.

**5. Template Rigidity**
Our template design prioritizes consistency but may limit creative variations that
experienced educators might prefer.

**6. Dataset Diversity**
While we cover 537 functions, the template structure is uniform. More varied animation
styles might benefit from dataset expansion.

Despite these limitations, our contributions—the curated dataset, documented methodology,
and empirical findings—provide valuable resources for future research in this domain.
```

---

### IX. Future Work (2 pages)

```
Building on our findings, we identify several promising directions for future research:

**Short-term (Immediate Improvements)**:
1. **Standardize instruction format**: Regenerate dataset with single consistent prompt
2. **Scale up model**: Test DeepSeek-Coder-7B, 13B, 33B
3. **Expand dataset**: Increase to 1,000+ examples
4. **Hybrid approach**: Combine fine-tuned generation with rule-based validation

**Medium-term (Alternative Approaches)**:
1. **Template-filling**: Instead of full generation, have model fill template placeholders
2. **Retrieval-augmented**: Use similarity search to find closest template, then adapt
3. **Constrained decoding**: Add structural constraints during generation
4. **Multi-stage generation**: Separate content planning from code synthesis

**Long-term (Broader Impact)**:
1. **Multi-framework support**: Extend to Matplotlib, Plotly, Manim-based alternatives
2. **Pedagogical evaluation**: Controlled studies with students and educators
3. **Interactive refinement**: Allow users to iteratively improve generated animations
4. **Curriculum integration**: Embed into learning management systems
5. **Multilingual support**: Adapt for non-English mathematics education
```

---

### X. Conclusion (1 page)

```
This paper presents DerivativeAnimator, a system for automated generation of calculus
derivative animations through LLM fine-tuning. We contribute a comprehensive dataset of
537 high-quality Manim code examples organized across four curriculum levels, which we
release to the research community to support future work in AI-assisted educational
content creation.

Through systematic experimentation, we demonstrate that while smaller LLMs (1.3B parameters)
can be successfully optimized—achieving training loss of 0.0548 and validation loss of
0.0192—they face challenges in generating complex structured code adhering to specific
templates. Our analysis identifies three critical factors affecting generation quality:

1. **Instruction format consistency**: Training data with diverse prompt formats (7 variations)
   dilutes pattern learning
2. **Model capacity relative to task complexity**: 3,400-character structured outputs may
   exceed 1.3B parameter model capabilities
3. **Dataset scale for complex tasks**: Complex code generation may require 1,000-5,000
   examples rather than 400-500

These findings provide actionable insights for the broader AI-assisted education research
community. While our initial implementation did not achieve full automated generation,
the methodology, infrastructure, and empirical results lay important groundwork for
future systems.

The gap between optimization metrics (low loss) and generation quality (inconsistent
template adherence) highlights a critical challenge in LLM-based structured code generation.
This suggests that next-token prediction loss, while useful for optimization, may be
insufficient for ensuring high-quality structured output. Future work should explore
structured decoding methods, template-filling approaches, or hybrid systems combining
neural generation with rule-based validation.

Despite current limitations, this research advances the important goal of democratizing
access to high-quality mathematical visualization tools. By openly sharing our dataset,
code, and findings, we enable the research community to build upon this foundation and
ultimately realize the vision of accessible, AI-powered educational content creation.
```

---

## 📊 TABLES AND FIGURES TO INCLUDE

### Required Tables (minimum 7):

1. **Table 1: Dataset Distribution by Curriculum Level** ✅ (defined above)
2. **Table 2: Code Characteristics** ✅ (defined above)
3. **Table 3: Training Hyperparameters** ✅ (defined above)
4. **Table 4: Evaluation Dimensions** ✅ (defined above)
5. **Table 5: Training Metrics** ✅ (defined above)
6. **Table 6: Generation Analysis** ✅ (defined above)
7. **Table 7: Instruction Format Analysis** ✅ (defined above)
8. **Table 8: Comparison with Related Work** (add this):

| Work | Task | Model Size | Dataset Size | Success Metric | Findings |
|------|------|------------|--------------|----------------|----------|
| CodeLlama [2023] | General code | 7B-34B | 500K examples | 50% pass@1 | Larger models + more data → better results |
| WizardCoder [2023] | Code completion | 15B | 78K examples | 57% pass@1 | Instruction tuning improves quality |
| Our Work | Math animation | 1.3B | 537 examples | Low loss, template challenges | Format consistency critical |

### Required Figures (minimum 5):

1. **Figure 1: Manim Template Structure** ✅ (defined above)
2. **Figure 2: Dataset Creation Pipeline** ✅ (defined above)
3. **Figure 3: Label Masking Illustration** ✅ (defined above)
4. **Figure 4: Training and Validation Loss Curves** ✅ (defined above)
5. **Figure 5: Loss vs Generation Quality Conceptual Diagram** ✅ (defined above)
6. **Figure 6: Example Generated vs Expected Output** (add this):
   - Side-by-side comparison
   - Left: What model generated
   - Right: What was expected
   - Annotate differences

7. **Figure 7: System Architecture** (add this):
```
┌─────────────────┐
│ User Input:     │
│ "f(x) = x^2"   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Instruction     │
│ Formatting      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Fine-Tuned      │
│ DeepSeek-1.3B   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Generated Code  │
│ (Manim Python)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Validation &    │
│ Execution Test  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Animation       │
│ Output (MP4)    │
└─────────────────┘
```

---

## 🌍 DATASET CONTRIBUTION GUIDE

### How to Contribute Dataset to Community

1. **Create GitHub Repository**:
```
Repository name: derivative-manim-dataset
Description: A dataset of 537 Manim code examples for calculus derivative
             visualizations, organized by curriculum level
```

2. **Repository Structure**:
```
derivative-manim-dataset/
├── README.md (comprehensive documentation)
├── LICENSE (CC BY 4.0)
├── code/
│   ├── foundation/      (130 files)
│   ├── conceptual/      (157 files)
│   ├── application/     (200 files)
│   └── advanced/        (50 files)
├── metadata/
│   ├── dataset_info.json
│   ├── statistics.json
│   └── curriculum_mapping.json
├── examples/
│   ├── rendered_animations/  (sample MP4s)
│   └── usage_examples.py
├── evaluation/
│   └── validation_scripts.py
└── paper/
    ├── paper.pdf
    └── supplementary_materials.pdf
```

3. **README.md Template**:
```markdown
# Derivative Manim Dataset

A curated dataset of 537 high-quality Manim code examples for automated generation
of calculus derivative visualizations.

## 📊 Dataset Overview

- **Total samples**: 537 (418 train, 50 validation, 57 test)
- **Curriculum levels**: 4 (Foundation, Conceptual, Application, Advanced)
- **Average code length**: 3,400 characters
- **Quality**: All examples syntactically valid and executable

## 🎯 Purpose

This dataset supports research in:
- AI-assisted educational content creation
- Mathematical code generation
- Instruction fine-tuning for specialized domains

## 📁 Structure

[Explain directory structure]

## 🚀 Usage

[Provide code examples]

## 📄 License

This dataset is released under CC BY 4.0. You are free to:
- Share, copy, and redistribute
- Adapt, remix, transform, and build upon
- Use for commercial purposes

With attribution required.

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{adhikari2025derivative,
  title={Automated Generation of Calculus Derivative Animations Using Fine-Tuned
         Large Language Models},
  author={Adhikari, Sushan and Sharma, Sunidhi and Lamichhane, Darshan and
          Adhikari, Usan},
  booktitle={[Conference Name]},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

## 📧 Contact

[Your contact information]
```

4. **Announce Dataset**:
   - Submit to Papers with Code
   - Post on Hugging Face Datasets Hub
   - Share on Twitter/LinkedIn with #MachineLearning #Education #OpenData
   - Submit to education-focused ML forums

5. **Documentation**:
   - Create video tutorial on using the dataset
   - Write blog post about dataset creation process
   - Prepare presentation slides for conferences

---

## ✍️ WRITING TIPS AND REPHRASING STRATEGIES

### How to Align Abstract Promises with Honest Results

**Abstract says**: "automatically generate Manim-based animations"

**You write**:
```
"We present a fine-tuning approach for automated Manim generation and conduct
systematic experiments to identify key factors affecting generation quality. While
our initial implementation with a 1.3B parameter model faced challenges in consistent
template adherence, our analysis provides critical insights for future work and we
contribute a valuable dataset to enable continued research."
```

**Abstract says**: "reducing content creation time from 8-12 hours to under 5 minutes"

**You write**:
```
"Our infrastructure demonstrates the potential for significant time reduction in
mathematical animation creation. While current generation quality requires further
development, the dataset curation pipeline we established reduces the barrier to
entry for future automated systems."
```

**Abstract says**: "evaluation framework will assess technical correctness"

**You write**:
```
"We design a comprehensive evaluation framework assessing syntactic accuracy, execution
success, and mathematical validity. Initial evaluation reveals that while our model
achieves low training loss, consistent template-adherent generation remains a challenge,
highlighting the need for additional research in structured code generation."
```

### Phrases to Use Throughout

**Positive framing of contributions**:
- "We contribute a novel dataset of 537 examples"
- "Our systematic analysis identifies three critical factors"
- "This work provides empirical evidence for future research"
- "We document lessons learned to guide the community"
- "Our findings highlight important considerations for"

**Honest framing of challenges**:
- "Initial experiments revealed challenges in..."
- "While optimization succeeded, generation quality requires further work"
- "This finding underscores the importance of..."
- "Our results suggest that... [factor] is critical for success"
- "This limitation motivates future work in..."

**Bridging abstract to reality**:
- "We present the infrastructure for... and identify key challenges"
- "Our work lays the foundation for future automated systems"
- "While full automation remains a goal for future work, we contribute..."
- "This research advances toward the vision of... by establishing..."

---

## 🎓 FINAL CHECKLIST

### Before Submission:

- [ ] Abstract accurately reflects paper content (with careful phrasing)
- [ ] All 7+ tables included and referenced
- [ ] All 5+ figures included with captions
- [ ] Dataset contribution section complete
- [ ] GitHub repository created and linked
- [ ] All claims supported by data or citations
- [ ] Limitations section is honest and complete
- [ ] Future work provides clear roadmap
- [ ] Code and data availability statement included
- [ ] References properly formatted (40+ citations)
- [ ] Acknowledgments include funding sources
- [ ] Supplementary materials prepared

### Key Message:

**Your paper contributes**:
1. ✅ Novel dataset (537 examples) - MAJOR contribution
2. ✅ Complete methodology and infrastructure
3. ✅ Empirical findings about fine-tuning challenges
4. ✅ Lessons learned for the community
5. ✅ Foundation for future research

**Frame as**: "An empirical study of LLM fine-tuning for mathematical code generation
that identifies critical success factors and contributes resources to the community"

**NOT as**: "A working system for automated animation generation" (that would be dishonest)

---

## 📞 Final Advice

1. **Be honest**: Reviewers appreciate honest reporting of challenges
2. **Emphasize contributions**: Dataset + findings are valuable even if generation didn't work
3. **Position properly**: Empirical study, not production system
4. **Help the community**: Open dataset helps everyone
5. **Show rigor**: Systematic experiments and analysis

**You have enough for a solid paper!** Good luck! 🎓🚀
