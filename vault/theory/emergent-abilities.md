---
title: Emergent Abilities in Large Language Models
tags: [emergence, emergent-abilities, scaling, in-context-learning, phase-transition]
aliases: [emergence in LLMs, emergent capabilities, phase transitions in scaling, in-context learning emergence]
difficulty: 2
status: complete
related: [scaling-laws, autoregressive-models, prompt-engineering, instruction-tuning, generalization-bounds]
---

# Emergent Abilities in Large Language Models

---

## Fundamental

### Definition

An **emergent ability** is a capability that is absent (near-random performance) for small models and appears sharply at some scale threshold, seemingly unpredicted by smaller-scale behavior (Wei et al., 2022).

This contrasts with standard scaling where performance improves smoothly and predictably across all scales.

**Examples documented in BIG-Bench:**
- Arithmetic (3-digit addition): near 0% accuracy up to ~10^23 FLOPs, then jumps to >80%
- Chain-of-thought reasoning: absent in GPT-3, present in GPT-4 / PaLM 540B
- Multi-step word problems, analogical reasoning, logical deduction

---

## Intermediate

### The Measurement Debate

**Schaeffer et al. (2023)** challenged the emergence narrative:

The appearance of emergence depends strongly on the **metric used**:
- **Discontinuous metrics** (exact-match accuracy, pass@1 for code): appear to show sudden jumps
- **Continuous metrics** (BPB — bits per byte, probability of correct token): show smooth, predictable improvement

Their argument: emergence is a **metric artifact**. If you measure the probability of the correct answer (a continuous quantity), scaling is always smooth. Binary correct/incorrect thresholds create apparent phase transitions from smooth underlying progress.

**Counter-argument:** some phenomena genuinely require multiple capabilities to co-occur. Multi-step reasoning requires each step to be individually reliable enough that a chain of $k$ steps succeeds — probability $p^k$. This creates a nonlinear threshold effect: $p^k$ increases sharply once $p$ passes the point where $p^k > 0.5$.

### In-Context Learning as an Emergent Ability

**In-context learning (ICL):** the ability to learn a new task from a few examples in the prompt, with no weight updates. Small models (< 1B params) show negligible ICL; large models (> 10B) show strong ICL.

ICL is considered emergent because:
- It requires the model to simultaneously represent the task, the examples, and the query
- It requires compositional generalization — applying rules seen in context to novel inputs
- It does not exist in the same form in smaller models; it's not just a weaker version

**Mechanistic hypotheses for ICL:**
1. The model implements **implicit gradient descent** in its forward pass — attention heads act as gradient-update steps (Akyürek et al., Dai et al.)
2. ICL is **Bayesian inference** — the model computes a posterior over tasks consistent with the examples (Xie et al.)
3. ICL retrieves from a **vast implicit lookup table** of (context, continuation) patterns memorized during training

---

## Advanced

### Phase Transitions and Grokking

**Grokking** (Power et al., 2022) is a related phenomenon: on modular arithmetic tasks (e.g., $a + b \mod p$), networks first memorize the training set (achieving 100% train accuracy but 0% validation accuracy), then — after many more gradient steps — suddenly generalize. The generalization appears as a phase transition.

This suggests that generalization sometimes requires **global circuit formation** (the model discovers the underlying algorithm) rather than incremental improvement. Regularization (weight decay) and training beyond apparent convergence are key enablers.

### Chain-of-Thought as Emergent Reasoning

**Chain-of-thought (CoT) prompting** elicits step-by-step reasoning by including "Let's think step by step" or few-shot CoT examples. This dramatically improves performance on math and logic tasks — but only for models above ~100B parameters.

Why CoT requires scale:
- Each intermediate reasoning step must be generated correctly
- The model must maintain state across steps in context
- Error accumulation across steps requires high per-step reliability

**Self-consistency sampling:** generate multiple CoT paths independently, aggregate by majority vote. Improves reliability by averaging out errors across chains.

### Predictability of Emergence

The key practical question: can emergence be predicted from small-scale experiments?

- **Smooth metrics allow extrapolation** (power-law fits from small-scale)
- **Emergent behaviors on discontinuous metrics cannot be reliably predicted** from smaller scale
- This creates a challenge for compute budget allocation: you can't know in advance if a model will exhibit CoT reasoning until it's large enough

The current practical response: train at multiple scales, observe which capabilities emerge, then plan larger runs accordingly.

*See also: [[scaling-laws]] · [[prompt-engineering]] · [[autoregressive-models]] · [[instruction-tuning]]*
