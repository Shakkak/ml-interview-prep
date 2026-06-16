---
title: Grokking
tags: [grokking, generalization, memorization, delayed-generalization, phase-transition]
aliases: [grokking, delayed generalization, memorization then generalization]
difficulty: 2
status: complete
related: [bias-variance-double-descent, regularization-weight-decay, generalization-bounds, emergent-abilities, spectral-bias]
depends_on: [bias-variance-double-descent, regularization-weight-decay, generalization-bounds]
---

# Grokking

---

## Fundamental

### The Phenomenon

**Grokking** (Power et al., 2022) is a training dynamic where a neural network:

1. First achieves 100% training accuracy by **memorizing** the training set (validation accuracy remains near random)
2. Then, after many additional training steps — well past apparent convergence — suddenly **generalizes** (validation accuracy jumps from chance to near-perfect)

The term comes from the sci-fi verb "to grok" — to understand deeply after a period of confusion.

**Original setting:** small transformers trained on modular arithmetic ($a \circ b \pmod{p}$ for operations $+, \times$, permutations). Training set is ~50% of all possible pairs. Grokking occurs reliably with sufficient weight decay.

---

## Intermediate

### Conditions for Grokking

Key enabling conditions identified experimentally:

1. **Weight decay:** without weight decay, the model stays in the memorization phase indefinitely. Weight decay continuously pushes the model toward simpler solutions — eventually a generalizing circuit becomes cheaper than a memorizing lookup table.

2. **Limited training data:** with enough data, the model generalizes quickly. Grokking is most dramatic with small datasets (e.g., 30–50% of the input space).

3. **Sufficient model capacity:** the model must be able to both memorize and generalize. Undercapacity models don't grok; they never memorize cleanly.

4. **Longer training:** grokking can require 10–100× more steps than memorization. This is invisible without long training runs.

### Mechanistic Interpretation

**Two competing circuits** hypothesis (Nanda et al., 2023): during training, two circuits form simultaneously:
- A **memorization circuit:** stores training examples as lookup patterns (low training loss, high parameters needed)
- A **generalizing circuit:** implements the true algorithm compactly (e.g., Fourier representation for modular addition)

Weight decay penalizes large weights, making the generalizing circuit increasingly cheaper than memorization. At a critical point, the generalizing circuit dominates — validation accuracy spikes.

**Fourier features in modular arithmetic (Nanda et al.):** the generalizing circuit for $a + b \pmod{p}$ uses Fourier representations: $\cos(2\pi k a / p)$, $\sin(2\pi k b / p)$ for specific frequencies $k$. The algorithm is elegant — and interpretable via mechanistic interpretability tools.

---

## Advanced

### Grokking Beyond Modular Arithmetic

Grokking has been observed in:
- **Algorithmic tasks:** binary addition, sorting, permutation composition
- **Formal logic:** theorem proving with limited axioms
- **Language tasks:** syntax learning in small transformers
- **In-context learning:** grokking of in-context learning rules in larger models

The broader pattern: any task with a compact underlying structure (that can be learned with fewer parameters than memorization) is a candidate for grokking.

### Relationship to Double Descent

Grokking is a **temporal** analogue of the double descent phenomenon (see [[bias-variance-double-descent]]):
- Double descent: as model capacity increases, generalization first degrades then improves (at interpolation threshold)
- Grokking: as training continues, generalization first stagnates (memorization) then improves (generalization circuit)

Both reflect the tension between memorization and compression in over-parameterized models.

### Implications for LLM Training

Grokking suggests that **very long training** on fixed data may reveal generalization capabilities not apparent at standard training durations. Evidence:
- Math and reasoning capabilities improve nonlinearly with training steps in some settings
- Continued pretraining on code improves math performance (the algorithm for math is shared with code structure)
- Emergent abilities (see [[emergent-abilities]]) may sometimes be grokking rather than purely scale-driven

**Practical takeaway:** early stopping based on validation loss may cut off grokking. For tasks with structured outputs, training longer (past apparent convergence) with weight decay may discover generalizing solutions.

## Links

- [[bias-variance-double-descent]] — grokking is a second-phase generalization phenomenon; the model first memorizes (low training error, high test error) then undergoes a phase transition to generalization
- [[regularization-weight-decay]] — weight decay is the primary driver of grokking: without it, models memorize and stay there; weight decay slowly penalizes the memorizing solution until a generalizing one is found
- [[generalization-bounds]] — grokking challenges uniform convergence bounds (which predict generalization at interpolation); it demonstrates that implicit regularization dynamics matter beyond parameter count
- [[emergent-abilities]] — grokking and emergence share the "phase transition" structure; both may be artifacts of threshold evaluation metrics rather than true discontinuities
- [[spectral-bias]] — grokking is accompanied by a shift in the model's frequency spectrum; networks first learn high-frequency (memorizing) solutions, then transition to low-frequency (generalizing) ones
