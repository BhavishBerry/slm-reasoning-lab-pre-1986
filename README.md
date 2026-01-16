# From-Scratch Long-Context SLM (Small Language Model)
> **A Research Initiative in Causal Reasoning & Architectural Efficiency**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-black.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Parameters](https://img.shields.io/badge/Parameters-~300M-success.svg)
![Context Window](https://img.shields.io/badge/Context-5000+-purple.svg)
![Training Data](https://img.shields.io/badge/Data-Pre--1986-yellow.svg)

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Research Question: The RBMK Reactivity Test](#2-core-research-question-the-rbmk-reactivity-test)
3. [Project Philosophy & Constraints](#3-project-philosophy--constraints)
    - [The Pre-1986 Knowledge Cutoff](#the-pre-1986-knowledge-cutoff)
    - [Reasoning Over Knowledge Retrieval](#reasoning-over-knowledge-retrieval)
    - [The Case for Small Models](#the-case-for-small-models)
4. [Repository Manifest](#4-repository-manifest)
5. [System Architecture: Deep Dive](#5-system-architecture-deep-dive)
    - [Model Configuration](#model-configuration)
    - [Component 1: Rotary Position Embeddings (RoPE)](#component-1-rotary-position-embeddings-rope)
    - [Component 2: Multi-Head Self-Attention](#component-2-multi-head-self-attention)
    - [Component 3: Block-Local Sparse Attention](#component-3-block-local-sparse-attention-the-key-innovation)
    - [Component 4: Feed-Forward Networks](#component-4-feed-forward-networks)
    - [Component 5: The Transformer Block](#component-5-the-transformer-block)
6. [Data Engineering Pipeline](#6-data-engineering-pipeline)
    - [Data Sources & Philosophy](#data-sources--philosophy)
    - [Tokenization Strategy (BPE)](#tokenization-strategy-bpe)
    - [Streaming Dataset Implementation](#streaming-dataset-implementation)
7. [Training Protocols & Curriculum](#7-training-protocols--curriculum)
    - [Phase A: Base Pretraining (Foundation)](#phase-a-base-pretraining-foundation)
    - [Phase B: Context Extension (Adaptation)](#phase-b-context-extension-adaptation)
    - [Phase C: Domain Fine-Tuning (Specialization)](#phase-c-domain-fine-tuning-specialization)
    - [Optimization & Hyperparameters](#optimization--hyperparameters)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Usage & Operations Manual](#9-usage--operations-manual)
    - [Environment Setup](#environment-setup)
    - [Hardware Requirements](#hardware-requirements)
    - [Step-by-Step Execution Guide](#step-by-step-execution-guide)
10. [Theoretical Appendices](#10-theoretical-appendices)
    - [Appendix A: The Mathematics of RoPE](#appendix-a-the-mathematics-of-rope)
    - [Appendix B: Attention Complexity Analysis](#appendix-b-attention-complexity-analysis)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)
12. [References & Citations](#12-references--citations)

---

## 1. Executive Summary

This project represents a rigorous, first-principles implementation of a **decoder-only Transformer** trained entirely from scratch. Unlike contemporary LLM projects that rely on finetuning massive pretrained checkpoints (e.g., Llama, Mistral), this initiative builds every component—from the tokenizer to the attention mechanism—to explore the fundamental mechanics of **long-context reasoning** in resource-constrained environments.

The resulting model is a **300 Million Parameter** Small Language Model (SLM) capable of handling context windows up to **5,000 tokens**. It introduces and validates advanced architectural techniques such as **Rotary Position Embeddings (RoPE)** for extrapolation and **Block-Local Sparse Attention** for quadratic complexity reduction.

**Key Achievements:**
- **Zero-Dependency Architecture**: Pure PyTorch implementation of the Transformer block.
- **Efficient Long Context**: Scaling from 2k to 5k tokens using block-sparsity.
- **Controlled Data Environment**: Trained exclusively on pre-1986 public domain scientific literature.
- **Three-Stage Training Pipeline**: Curriculum learning from language modeling to domain adaptation.

The primary goal is not to beat benchmarks on MMLU, but to produce a "glass box" model where every neuron and attention head's purpose is understood, derived, and explainable.

---

## 2. Core Research Question: The RBMK Reactivity Test

**Can a Model Infer the Positive Reactivity Coefficient Without Hindsight?**

A central motivation of this project is to investigate whether a language model, trained *exclusively* on foundational pre-1986 physics and engineering knowledge, can infer the presence and implications of a positive reactivity coefficient in an RBMK-type nuclear reactor without being told that such a failure ever occurred.

### The Physics of the Problem
The RBMK reactor design is historically associated with a dangerous positive feedback mechanism: under certain operating conditions, a reduction in coolant density (void formation) leads to an increase in reactor power rather than a decrease. This behavior violates the intuitive expectation that engineered safety systems should exhibit negative feedback. Importantly, the physics underlying this phenomenon—neutron moderation, void formation, reactivity balance, and control rod dynamics—were all known in principle prior to 1986.

### The "Clean Room" Experiment
This project asks a narrowly defined but deep question: **Given only foundational nuclear physics, reactor kinetics, and control theory—can a language model reason its way to identifying a dangerous positive feedback loop?**

The model is **never** trained on:
- Accident reports of Chernobyl (post-1986).
- Design critiques of RBMK reactors (post-1986).
- Safety verdicts or post-event analyses.

It is exposed **only** to:
- Neutron diffusion theory.
- Reactor kinetics equations.
- Moderator and coolant behavior.
- Classical control-system stability concepts.

### The Reasoning Test
During evaluation, the model is presented with abstracted system descriptions. The key test is whether the model can internally integrate distinct domains of knowledge to:
1.  **Identify** conditions where reduced coolant density increases neutron moderation.
2.  **Recognize** that this introduces a positive feedback loop.
3.  **Conclude** that positive reactivity coefficients undermine passive safety.

Success in this context does not mean reproducing historical conclusions. Instead, it would indicate that the model can perform **counterfactual reasoning**: predicting system instability based solely on first principles. This serves as a stress test for **predictive safety analysis**, probing whether AI systems can detect latent risks in complex designs *before* they manifest as catastrophic failures.

---

## 3. Project Philosophy & Constraints

### The Pre-1986 Knowledge Cutoff
This project adheres to a strict "Pre-1986" data constraint. All training data consists of public domain books, textbooks, and technical papers published before 1986.

**Why this constraint?**
1.  **Copyright Purity**: Ensuring all data is indisputably in the public domain.
2.  **Reasoning vs. Memorization**: Modern internet data is cluttered with "answers". By using older, foundational texts, we force the model to learn the *principles* of physics and engineering rather than memorizing StackOverflow solutions.
3.  **Epistemic Isolation**: It simulates a "clean room" environment to study reasoning emergence without modern bias.

### Reasoning Over Knowledge Retrieval
Typical RAG (Retrieval-Augmented Generation) systems mask model deficiencies by fetching external data. This project explicitly rejects RAG. The goal is not to build a search engine, but a **reasoning engine**.

### The Case for Small Models
At 300 million parameters, this model is orders of magnitude smaller than GPT-4. However, "small" models allow for:
-  **Interpretability**: Easier tracing of attention patterns.
-  **Iteration Speed**: Rapid architectural experimentation.
-  **Democratization**: Feasible research on single-GPU workstations.

---

## 4. Repository Manifest

Below is a comprehensive breakdown of the project structure and the purpose of every file.

```text
train_slm/
├── context/
│   ├── from_scratch_long_context_slm_300_m_5_k_implementation_canvas.md
│   └── project_introduction_from_scratch_long_context_slm.md
├── data/
│   └── pre1986_training_streams_v1_FINAL/
│       ├── base_stream.txt         # General Knowledge
│       ├── finetune_control.txt    # Control Theory
│       ├── finetune_nuclear.txt    # Nuclear Engineering
│       └── finetune_reliability.txt# Reliability Engineering
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Statistics & Validation
│   ├── 02_tokenizer_training.ipynb # BPE Training
│   ├── 03_model_architecture.ipynb # THE CORE: Model Classes
│   ├── 04_training_pipeline.ipynb  # THE ENGINE: Training Loop
│   └── 05_evaluation.ipynb         # Testing & Metrics
├── tokenizer/
│   └── tokenizer.json              # Trained Vocabulary
├── configs/
│   └── model_config.json           # Hyperparameters
├── checkpoints/                    # Model Weights
├── .gitignore                      # Git config
└── README.md                       # This Documentation
```

---

## 5. System Architecture: Deep Dive

The model follows the GPT-2/Llama lineage of **decoder-only Transformers**, but with significant modernization for 2025-era efficiency standards.

### Model Configuration

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1024        # Hidden size
    n_layers: int = 24         # Number of transformer blocks
    n_heads: int = 16          # Attention heads
    d_ff: int = 4096           # 4x expansion
    max_seq_len: int = 2048    # Base context
    block_size: int = 512      # For local attention
```

### Component 1: Rotary Position Embeddings (RoPE)

Instead of absolute position embeddings, we implement **RoPE**. RoPE encodes relative positions by rotating the Query and Key vectors in the complex plane, allowing for better extrapolation to undefined sequence lengths.

### Component 2: Multi-Head Self-Attention

Standard Causal Self-Attention mechanism used in Phase A. $O(N^2)$ complexity.

### Component 3: Block-Local Sparse Attention (The Key Innovation)

For long contexts (4k-5k tokens), full attention becomes expensive. We introduce **Block-Local Attention** for Phase B and C.

**Mechanism:**
1.  Split sequence into blocks (e.g., 512 tokens).
2.  Each token attends to:
    -   Its own block (Intra-block).
    -   The previous block (Recurrent neighbor).

**Why it works**:
It reduces complexity from quadratic to linear while maintaining local coherence. Global coherence is maintained via layer depth.

### Component 4: Feed-Forward Networks & Normalization

- **Activation**: GeLU (Gaussian Error Linear Unit).
- **Normalization**: Pre-Norm architecture for stability.
- **Weight Tying**: Input embedding and output projection share weights.

---

## 6. Data Engineering Pipeline

### Data Sources & Philosophy
We curated three primary text streams:
1.  **Base Stream**: General scientific literature, classic fiction, and math textbooks (< 1986).
2.  **Control Stream**: Control theory, feedback loops, system dynamics.
3.  **Reliability & Nuclear Streams**: Failure mode analysis, safety reports, reliability engineering.

### Tokenization Strategy (BPE)
- **Vocab Size**: 32,000.
- **Algorithm**: Byte-Pair Encoding (BPE) trained on base stream shards.
- **Special Tokens**: `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`.

### Streaming Dataset Implementation
We implement a **Streaming `IterableDataset`** to process large files without RAM saturation. The dataset yields a continuous stream of tokens, and the DataLoader slices them into `seq_len` windows on the fly.

---

## 7. Training Protocols & Curriculum

We employ a **Curriculum Learning** strategy divided into three distinct phases.

### Phase A: Base Pretraining (Foundation)
-   **Objective**: Linguistic competence and general knowledge.
-   **Context**: 2048 tokens.
-   **Attention**: Full $O(N^2)$ Global Attention.
-   **Optimization**: AdamW, Peak LR `3e-4`.

### Phase B: Context Extension (Adaptation)
-   **Objective**: Adapt to long-range dependencies.
-   **Context**: **Extended to 4096-5120 tokens**.
-   **Attention**: Block-Local Sparse Attention.
-   **RoPE Scaling**: Linear frequency scaling ($factor=2.0$).
-   **Optimization**: Lower Peak LR `1e-4`.

### Phase C: Domain Fine-Tuning (Specialization)
-   **Objective**: Specialize in analytical, safety-critical thinking.
-   **Context**: 4096 tokens.
-   **Data**: Mixed stream (Control + Nuclear + Reliability).
-   **Optimization**: Lowest Peak LR `5e-5`.

---

## 8. Evaluation Framework

We use **Qualitative Zero-Shot Analysis** with open-ended engineering scenarios.

**Example Prompts**:
1.  "Describe the failure mode of a steam governor if the flyball linkage snaps."
2.  "A PID controller has a high integral gain. Explain in terms of phase margin what happens to the system response."
3.  "Analyze the stability of a reactor core where the coolant has a significantly lower neutron absorption cross-section than the moderator."

**Success Metrics**:
-   **Causal Coherence**
-   **Terminological Precision**
-   **Long-Term Memory**

---

## 9. Usage & Operations Manual

### Environment Setup

**1. Clone the Repository**
```bash
git clone https://github.com/BhavishBerry/slm-reasoning-lab-pre-1986.git
cd slm-reasoning-lab-pre-1986
```

**2. Virtual Environment**
```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install torch numpy tqdm matplotlib jupyterlab tokenizers pandas
```

### Hardware Requirements
-   **Minimum**: NVIDIA GPU with 8GB VRAM.
-   **Recommended**: NVIDIA GPU with 24GB VRAM (RTX 3090/4090).

### Step-by-Step Execution Guide

1.  **Data Audit**: Run `notebooks/01_data_exploration.ipynb`.
2.  **Tokenizer**: Run `notebooks/02_tokenizer_training.ipynb`.
3.  **Architecture**: Run `notebooks/03_model_architecture.ipynb`.
4.  **Training**: Run `notebooks/04_training_pipeline.ipynb` (approx. 48 hours).
5.  **Evaluation**: Run `notebooks/05_evaluation.ipynb`.

---

## 10. Theoretical Appendices

### Appendix A: The Mathematics of RoPE
$$
f(x, m) = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \cdot e^{im\theta}
$$
This formulation preserves translation invariance: $\langle f(q, m), f(k, n) \rangle = g(q, k, m-n)$.

### Appendix B: Attention Complexity Analysis
| Mechanism          | Complexity     | Max viable context (16GB VRAM) |
| :----------------- | :------------- | :----------------------------- |
| **Full Attention** | $O(N^2)$       | ~2048                          |
| **Block-Local**    | $O(N \cdot B)$ | 8192+                          |

---

## 11. Troubleshooting & FAQ

**Q: My loss is not decreasing.**
*A: Check tokenizer consistency and learning rate warmup.*

**Q: CUDA OOM.**
*A: Reduce batch info and increase gradient accumulation.*

**Q: Can I use this for coding Python?**
*A: No. The training data is pre-1986.*

---

## 12. References & Citations

1.  **Attention Is All You Need**, Vaswani et al.
2.  **RoFormer**, Su et al.
3.  **Llama 2**, Touvron et al.
4.  **Long LoRA**, Yan et al.

---

> **Author Note**: This project serves as a foundational educational resource for understanding the mechanics of Large Language Models. By building "small", we understand "large".

*Built with ❤️ in Python/PyTorch. Validated on NVIDIA RTX Architecture.*
