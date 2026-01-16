# From-Scratch Long-Context SLM (300M Params, ~5k Context)

## Project Context (Read First)
This canvas is written after reviewing the project README and constraints. The model is:
- Trained **from scratch** (no pretrained weights)
- Uses **only pre-1986 knowledge**
- Avoids RAG, instruction tuning, and hindsight leakage
- Focused on **counterfactual safety & system-level reasoning**
- Evaluated **zero-shot only**

The goal is *not* benchmark performance, but to demonstrate:
- Reasoning emergence
- Long-context architectural techniques
- Careful experimental discipline

---

## 0. High-Level Objective

Build a decoder-only Small Language Model that:
- Has ~250–300M parameters
- Trains at 2k context and extends to ~5k context
- Uses **RoPE scaling + block-local attention**
- Demonstrates modern long-context ideas in an explainable way

Non-goals:
- No retrieval systems
- No instruction-following datasets
- No chain-of-thought forcing
- No mixture-of-experts

---

## 1. Data Pipeline

### 1.1 Data Sources

**Base pretraining data (≤1985 only)**
- Public-domain English books & essays
- Mathematics and physics textbooks
- Classical engineering & control systems material
- Scientific and technical writing

**Fine-tuning data**
- Synthetic system descriptions
- Counterfactual engineering scenarios
- Stability and failure-mode analyses
- No labels, no answers, no verdicts

---

### 1.2 Cleaning & Normalization

Steps:
1. Convert all sources to UTF-8 plain text
2. Remove front/back matter (TOC, references, metadata)
3. Normalize whitespace and symbols
4. Preserve equations and technical notation
5. Split into **documents**, not sentences

Rationale:
- Long-context reasoning depends on document structure
- Sentence-level chunking destroys causal chains

---

### 1.3 Chunking Strategy

- Do **not** chunk during preprocessing
- Preserve full documents
- Chunk dynamically at batch construction time

This avoids teaching the model artificial stopping points.

---

## 2. Tokenization

### 2.1 Tokenizer Design

- Type: BPE or Unigram LM
- Vocabulary size: ~32,000 tokens

Reasons:
- Handles technical vocabulary cleanly
- Prevents excessive token fragmentation
- Keeps embedding size manageable

---

### 2.2 Tokenizer Training

Procedure:
1. Train tokenizer on **base pretraining corpus only**
2. Validate tokenization of equations & technical terms
3. Freeze tokenizer permanently

Rule:
> Tokenization defines thought granularity. Poor tokens → poor reasoning.

---

## 3. Embeddings & Positional Encoding

### 3.1 Token Embeddings

- Standard learned embedding table
- Shape: `[vocab_size, d_model]`

---

### 3.2 Positional Encoding (Critical)

- Use **Rotary Position Embeddings (RoPE)**
- Train at context = 2048
- Extend to 4096–5120 using RoPE scaling (linear or NTK-aware)

Benefits:
- No learned position cap
- Extrapolates to longer contexts
- No additional parameters

---

## 4. Model Architecture

### 4.1 Core Architecture

- Decoder-only Transformer
- Dense (no MoE)

Target configuration:
- Layers: 24
- d_model: 1024
- Attention heads: 16
- Head dim: 64
- FFN size: 4096 (4× expansion)
- Parameters: ~250–300M

---

### 4.2 Attention Mechanism

#### Phase 1: Full Attention
- Used only during early pretraining
- Context length: 2048

#### Phase 2: Block-Local Attention (Key Contribution)

Mechanism:
- Split sequence into fixed blocks (e.g., 512 tokens)
- Each token attends to:
  - Tokens in its own block
  - Tokens in the immediately previous block (causal)

Effect:
- Reduces O(N²) attention to O(N × block_size)
- Makes ~5k context feasible

---

### 4.3 Optional Landmark Tokens

Special tokens such as:
- <SUMMARY>
- <STATE>
- <ASSUMPTION>

Properties:
- Attend globally
- Act as memory anchors
- Very small compute cost

Only introduce after base training is stable.

---

## 5. Training Plan

### Phase A: Base Pretraining

Purpose:
- Learn language, math, and reasoning structure

Config:
- Context: 2048
- Attention: full
- Learning rate: ~3e-4 (with warmup)
- Small batch size + gradient accumulation

Success indicators:
- Coherent long-form explanations
- Conditional reasoning
- Explicit assumptions

---

### Phase B: Context Extension

Purpose:
- Adapt model to longer sequences

Changes:
- Context: 4096 → 5120
- Attention: block-local
- Learning rate: reduced (e.g., 1e-4)

Notes:
- Same data distribution
- No new objectives

---

### Phase C: Domain Fine-Tuning

Purpose:
- Shape *how* the model reasons

Data:
- Counterfactual system analyses
- Engineering stability scenarios

Rules:
- No answers
- No labels
- No chain-of-thought forcing

---

## 6. Evaluation Framework (Zero-Shot)

### 6.1 Prompt Style

- Open-ended analytical prompts
- No examples
- No formatting instructions

---

### 6.2 Evaluation Criteria

Qualitative dimensions:
- Causal depth
- Conditional reasoning
- Explicit uncertainty
- Cross-paragraph consistency
- Compression without loss

---

### 6.3 Long-Context Tests

- Compare 2k vs 5k context
- Check:
  - Reference to earlier assumptions
  - Stability of conclusions
  - Absence of contradiction

---

## 7. Diagnostics & Ablations

Compare:
- Full vs block-local attention
- Short vs long context
- With vs without landmark tokens

Track:
- Training stability
- Memory usage
- Attention patterns
- Reasoning consistency

---

## 8. What This Project Demonstrates

- Training an SLM from scratch
- Controlled dataset curation
- Long-context architectural techniques
- Reasoning-first evaluation
- Interpretability over scale

---

## 9. Suggested Repository Structure

```
slm/
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── tokenized/
├── tokenizer/
├── model/
│   ├── attention.py
│   ├── rope.py
│   └── transformer.py
├── train/
│   ├── pretrain.py
│   ├── extend_context.py
│   └── finetune.py
├── eval/
├── configs/
└── README.md
```

---

## Final Principle

Long context is not the goal.
**Controlled reasoning under extended context is.**

This model is designed to surface reasoning limits, not hide them behind scale.