# From-Scratch Long-Context Small Language Model for Counterfactual Safety Reasoning

## Overview

This project documents the design, training, and evaluation of a **from-scratch Small Language Model (SLM)** developed to study **reasoning in safety-critical engineered systems under counterfactual constraints**. The model is trained **without pretrained weights**, **without retrieval-augmented generation**, and **without instruction tuning**, using **only pre-1986 textual knowledge**.

The core motivation behind this work is to explore whether a language model, trained exclusively on foundational scientific and engineering knowledge, can reason about system instability, feedback loops, and risk **without exposure to historical accidents or hindsight explanations**.

Rather than optimizing for benchmark performance or conversational fluency, the project prioritizes **reasoning depth, causal analysis, and interpretability**.

---

## Motivation

Many large-scale engineering failures were not caused by unknown physics, but by:
- Positive feedback loops
- Control-system instabilities
- Human–machine interaction failures
- Organizational and procedural blind spots

In many cases, the underlying equations and principles were already known. What failed was the ability to reason holistically about system behavior under unusual or transient conditions.

This project investigates whether a deliberately constrained language model can act as a **reasoning probe**—surfacing latent instability or risk by reasoning from first principles, rather than recalling historical outcomes.

---

## Core Principles

This work is guided by several non-negotiable principles:

- **From-scratch training only**  
  All model parameters are randomly initialized. No pretrained models or weights are used at any stage.

- **Counterfactual realism**  
  The training corpus is strictly limited to material published before 1986, preventing historical leakage from major industrial and nuclear accidents.

- **No retrieval or memory augmentation**  
  The model reasons only over its internal representations and the provided context window.

- **Reasoning-first evaluation**  
  The model is evaluated using zero-shot analytical prompts rather than instruction-following benchmarks.

- **Small, interpretable scale**  
  Model size (approximately 250–300M parameters) is intentionally constrained to encourage abstraction over memorization.

---

## Model Architecture

The model is a **decoder-only Transformer** designed to balance expressive capacity with interpretability.

Key architectural characteristics include:
- Dense Transformer backbone (no Mixture-of-Experts)
- Rotary Position Embeddings (RoPE) for relative positional encoding
- Block-local sparse attention to enable extended context lengths
- Progressive context extension from approximately 2k to ~5k tokens

These choices allow the model to process long system descriptions while avoiding the quadratic scaling limitations of vanilla self-attention.

---

## Dataset Design

The dataset is curated to teach the model **how to reason**, not **what conclusions to reach**.

### Base Pretraining Data
- Public-domain English prose
- Mathematics and physics textbooks
- Classical engineering and control theory texts
- Scientific and technical writing

### Fine-Tuning Data
- Synthetic system descriptions
- Counterfactual engineering scenarios
- Stability and failure-mode analyses

### Explicit Exclusions
- Accident reports or post-event analyses
- Safety verdicts or conclusions
- Post-1986 material of any kind

This separation ensures that any observed safety reasoning emerges from first principles rather than memorized outcomes.

---

## Training Methodology

Training proceeds in three carefully controlled stages:

1. **Base Pretraining**  
   Learning language structure, mathematical reasoning, and technical discourse at a moderate context length.

2. **Context Extension**  
   Gradually extending the context window using RoPE scaling and block-local attention.

3. **Domain Fine-Tuning**  
   Shaping analytical reasoning over engineered systems without introducing labels, answers, or outcome supervision.

At no stage is the model instruction-tuned or exposed to example-based prompting.

---

## Evaluation Philosophy

The model is evaluated **exclusively in a zero-shot setting** using open-ended analytical prompts.

Evaluation focuses on:
- Identification of causal dependencies
- Explicit statement of assumptions
- Handling of uncertainty and incomplete information
- Consistency across paraphrased prompts
- Ability to reason over long contexts without contradiction

There is no single quantitative score; both successes and failures are treated as meaningful research outcomes.

---

## Scope and Contributions

This project demonstrates:
- End-to-end training of a language model from scratch
- Dataset curation under strict historical constraints
- Practical implementation of modern long-context techniques
- A reasoning-centered evaluation framework for safety-critical domains

The repository is intended as a **research and learning artifact**, not a production system.

---

## Ethical Framing

This work does **not** claim that artificial intelligence systems could have prevented historical disasters.

Instead, it explores whether constrained language models can:
- Surface latent instability in complex systems
- Expose reasoning blind spots
- Assist human analysts by acting as structured reasoning aids

All conclusions are framed with appropriate uncertainty and humility.

---

## Final Note

This project prioritizes **clarity over scale** and **understanding over performance**.

If the model succeeds, it provides insight into how reasoning can emerge from foundational knowledge.
If it fails, the failure itself is treated as a valuable scientific result.

Both outcomes contribute meaningfully to the study of reasoning in language models.