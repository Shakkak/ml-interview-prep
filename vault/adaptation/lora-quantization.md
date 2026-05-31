---
title: LoRA, Quantization, and Efficient Fine-tuning
tags: [fine-tuning, lora, quantization, llm, efficiency]
aliases: [LoRA, QLoRA, low-rank adaptation, NF4, PEFT]
difficulty: 2
status: complete
related: [attention-mechanism, backpropagation, rlhf]
---

# LoRA, Quantization, and Efficient Fine-tuning

---

## Fundamental

Full fine-tuning a GPT-3 scale model (~175B parameters) requires storing weights + gradients + Adam moments: roughly 2.8 TB in FP32. Even a 7B model needs ~112 GB. The goal is to adapt the model to a new task with a fraction of this cost.

**Why quantize:** memory for model weights scales with precision:
- FP32: 4 bytes/param → 7B model = 28 GB
- FP16/BF16: 2 bytes/param → 7B model = 14 GB
- INT8: 1 byte/param → 7B model = 7 GB
- NF4: 0.5 bytes/param → 7B model = 3.5 GB

**Matrix rank:** the rank of $A \in \mathbb{R}^{m \times n}$ is the number of linearly independent rows/columns. If rank $r < \min(m,n)$: $A = UV^T$ where $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$. Storage drops from $mn$ to $(m+n)r$ numbers — much less if $r \ll \min(m,n)$.

---

## Intermediate

### Why Weight Updates Are Low-Rank

Empirical finding (Aghajanyan et al., 2020): **weight changes during fine-tuning have low intrinsic dimensionality**. Pre-trained models already encode rich features; fine-tuning on a downstream task requires only a small "correction" in a low-dimensional subspace. Training GPT-3 with random projection matrices: 90% of fine-tuning performance is recoverable with rank ≤ 64.

### LoRA: Low-Rank Adaptation

For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$, decompose the update:

$$\Delta W = B A, \quad B \in \mathbb{R}^{d \times r},\, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

Modified forward pass: $h = W_0 x + \frac{\alpha}{r} B A x$

- $W_0$ is **frozen** — never updated; only $A$ and $B$ are trained
- Initialization: $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ → $\Delta W = 0$ at start (preserves pretrained behavior)
- Typical: $r = 4$–$16$; applied to $W_Q, W_K, W_V, W_O$ in attention layers

**Parameter savings:** 4096×4096 weight matrix with $r=8$:
- Full fine-tuning: $4096^2 = 16.7M$ parameters
- LoRA: $(4096+4096) \times 8 = 65.5K$ parameters — **255× fewer**

**Zero inference latency:** merge the adapter after training: $W = W_0 + \frac{\alpha}{r}BA$. Same shape as $W_0$, no extra compute.

### Uniform Quantization

Map a range $[\min, \max]$ to integer values $[0, 2^b - 1]$:

$$\hat{x} = \text{round}\left(\frac{x - \min}{\max - \min} \cdot (2^b - 1)\right)$$

**Problem:** neural weights follow approximately Gaussian distributions. Uniform bins waste resolution on the tails and have too little resolution near zero where most weights cluster.

---

## Advanced

### NF4: NormalFloat 4-bit

NF4 (used in QLoRA) uses equal-area bins under the standard Gaussian instead of equal-width bins:

$$q_i = \Phi^{-1}\left(\frac{i + 0.5}{2^4}\right)$$

High resolution where most weights are (near zero), coarser in the tails. Empirically outperforms INT4 for neural network weights.

### QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines NF4 quantization with LoRA adapters:

```
Frozen base model (NF4, ~0.5 B/p)  ← stays quantized
         ↓ dequantize on-the-fly during forward pass
Add LoRA contribution (BF16)         ← trained
```

Three components:
1. **NF4 quantization:** base weights stored in 4-bit; dequantized to BF16 block-by-block during forward pass, then discarded.
2. **Double quantization:** quantize the quantization constants (scale factors) themselves to FP8. Saves ~0.4 bits/param additional.
3. **Paged optimizer states:** Adam moments paged to CPU RAM when GPU memory is tight — prevents OOM during gradient accumulation.

Memory for a 65B model:

| Component | Memory |
|-----------|--------|
| Base weights (NF4) | ~33 GB |
| LoRA adapters ($r=16$, BF16) | ~0.3 GB |
| Optimizer states (LoRA only) | ~1.2 GB |
| Activations + gradients | ~2 GB |
| **Total** | **~37 GB** |

Fits on a single A100 80GB. Without QLoRA: 130+ GB in FP16.

### PEFT Methods Comparison

| Method | Params Trained | Strength |
|--------|---------------|---------|
| Full fine-tuning | All | Best performance |
| LoRA | ~0.1% (attention matrices) | Near full perf, swappable, mergeable |
| Prefix tuning | ~0.1% (KV prefix per layer) | More expressive than input prompt tuning |
| Prompt tuning | <0.01% (input tokens only) | Very cheap; weaker on hard tasks |
| Adapter layers | ~1% (after FFN sublayer) | Older approach; similar trade-off to LoRA |

---

*See also: [[attention-mechanism]] · [[rlhf]] · [[backpropagation]]*
