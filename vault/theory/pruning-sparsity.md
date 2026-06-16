---
title: Pruning and Network Sparsity
tags: [pruning, sparsity, network-compression, magnitude-pruning, structured-pruning]
aliases: [weight pruning, network pruning, structured pruning, unstructured pruning, lottery ticket, RIGL]
difficulty: 2
status: complete
related: [lora-quantization, knowledge-distillation, regularization-weight-decay, hessian-curvature, generalization-bounds]
depends_on: [backpropagation, hessian-curvature, regularization-weight-decay]
---

# Pruning and Network Sparsity

---

## Fundamental

### Why Prune?

Trained neural networks are heavily over-parameterized. Many weights are near-zero and contribute little to predictions. Removing them (setting to zero) produces a **sparse network** that:
- Requires less storage (zero weights can be compressed)
- Can be faster at inference (with sparse hardware support)
- May generalize better (implicit regularization)

**Pruning pipeline:**
1. Train a dense model fully
2. Prune according to some criterion (remove weights)
3. Fine-tune the pruned model to recover accuracy

### Magnitude Pruning

The simplest pruning criterion: remove weights with small absolute value.

**Unstructured (weight) pruning:** zero out the smallest $k\%$ of weights globally:
$$w_i = 0 \text{ if } |w_i| < \tau$$

where $w_i$ = individual weight, $\tau$ = magnitude threshold chosen so that $k\%$ of weights fall below it.

**Lottery Ticket Hypothesis:** Frankle & Carlin (2018) showed that dense networks contain small sparse subnetworks ("winning tickets") that can be trained from scratch to match the full network's accuracy. These tickets are found by:
1. Train dense network
2. Prune the smallest $p\%$ of weights
3. **Rewind** remaining weights to their initial values (not post-training values)
4. Retrain only the remaining weights from those initial values

---

## Intermediate

### Structured vs Unstructured Pruning

**Unstructured pruning:** any individual weight can be pruned. Produces irregular sparsity — the weight matrix has scattered zeros. Hard to accelerate without specialized sparse hardware.

**Structured pruning:** prune entire structures:
- Filter pruning: remove entire convolutional filters (reduces channel count)
- Head pruning: remove entire attention heads in transformers
- Layer pruning: remove entire layers

Structured pruning produces dense models with smaller dimensions — directly accelerable on standard hardware (no sparse matmul needed).

**Tradeoff:** unstructured can achieve higher sparsity at same accuracy; structured trades accuracy for hardware efficiency.

### Pruning Criteria Beyond Magnitude

**Gradient-based (Taylor expansion):** importance of weight $w_i$ ≈ change in loss if removed: $|\delta \mathcal{L}| \approx |g_i w_i|$, where $g_i = \partial \mathcal{L}/\partial w_i$ = gradient of loss w.r.t. $w_i$ and the product $|g_i w_i|$ approximates the first-order loss change from zeroing $w_i$.

**Second-order (OBD, OBS):** use Hessian diagonal to estimate importance: $\frac{w_i^2}{2[H^{-1}]_{ii}}$. More accurate but expensive.

**Activation-based:** for structured pruning, rank filters by the L1 norm of their output activations — low-activation filters contribute little.

### Dynamic Sparse Training (RiGL)

**RiGL (Evci et al., 2020)** maintains a fixed sparsity budget throughout training without the train-prune-finetune cycle:
1. Start with random sparse mask
2. Periodically: drop $\Delta$ weights with smallest magnitude, grow $\Delta$ weights with largest gradient magnitude
3. This explores the sparse space during training, converging to a good sparse solution

Achieves quality close to magnitude pruning + finetuning in a single training run.

---

## Advanced

### Structured Pruning for Transformers

**Attention head pruning:** Michel et al. (2019) showed that 70-80% of attention heads can be pruned in BERT with minimal performance loss. Different heads specialize (syntactic, semantic, positional) and the least-used can be removed.

**Block / layer pruning:** pruning transformer layers — alternating layers prune better than the last $k$ layers (the later layers are not strictly more important). ShortGPT, LaCo and similar methods find that large LLMs have high layer redundancy.

### Quantization + Pruning Interaction

Pruning and quantization (see [[lora-quantization]]) are complementary compression methods:
- Pruning: reduces the number of parameters (eliminates connections)
- Quantization: reduces bits per parameter (lower precision)

Combined ("sparse quantization"), they can achieve 10–100× compression with minimal accuracy loss. SparseGPT applies second-order pruning to LLMs in a single forward pass without retraining.

## Links

- [[backpropagation]] — pruning during training uses gradient-based saliency scores; the OBD (Optimal Brain Damage) and OBS (Optimal Brain Surgeon) use second-order (Hessian) information to identify low-saliency weights
- [[hessian-curvature]] — second-order pruning (OBD, SparseGPT) estimates the curvature-weighted importance of each weight: $\Delta L \approx \frac{1}{2}w_i^2 H_{ii}^{-1}$; flat curvature = safe to prune
- [[regularization-weight-decay]] — L1 regularization promotes sparsity by driving low-importance weights to exactly zero; it is a differentiable approximation to the $\ell_0$ pruning objective
- [[lora-quantization]] — pruning and LoRA are complementary: pruning removes weights, LoRA reduces the rank of remaining weight updates; SparseGPT combines both for LLM compression
- [[knowledge-distillation]] — structured pruning (removing entire channels/heads) requires fine-tuning the pruned network to recover performance; distillation from the full network accelerates recovery
