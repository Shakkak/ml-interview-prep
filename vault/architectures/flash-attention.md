---
title: Flash Attention — IO-Aware Exact Attention
tags: [flash-attention, attention, memory-efficiency, gpu-optimization, tiling, hbm, sram]
aliases: [FlashAttention, Flash Attention, IO-aware attention, fused attention kernel]
difficulty: 3
status: complete
related: [attention-mechanism, arch-kv-cache, vision-transformer, normalization-layers]
---

# Flash Attention — IO-Aware Exact Attention

---

## Fundamental

Standard attention computes $\text{softmax}(QK^T / \sqrt{d_k}) V$. For sequence length $N$, the $QK^T$ matrix has $N^2$ entries. At $N=4096$: 16M entries × 2 bytes = 32 MB **per head per layer**.

Modern GPUs have two memory levels:
- **HBM (High Bandwidth Memory):** large (40–80 GB on A100), 2 TB/s bandwidth — slow relative to compute.
- **SRAM (on-chip):** tiny (192 KB on A100), ~19 TB/s bandwidth — ~10× faster.

Standard attention writes the $N \times N$ matrix to HBM, reads it for softmax, writes it back, then reads it for the $V$ multiplication. **The bottleneck is HBM bandwidth, not FLOPs.** The computation is memory-bound.

**Flash Attention** computes exact attention **without materializing the $N \times N$ matrix in HBM**, by keeping intermediate computations in fast SRAM using block-by-block tiling.

---

## Intermediate

### The Key Insight: Online Softmax

Standard [[activation-softmax|softmax]] requires seeing all $N$ scores before normalizing — this seems to require the full $N$-length vector in memory. But softmax can be computed **incrementally**.

Maintain two running statistics: $m$ (running maximum) and $\ell$ (running normalizer $\sum e^{x_i - m}$). When a new block arrives with local max $m_{\text{new}}$:

$$m' = \max(m,\, m_{\text{new}}), \qquad \ell' = e^{m - m'} \cdot \ell + e^{m_{\text{new}} - m'} \cdot \ell_{\text{new}}$$

> [!tip] Why the $e^{m - m'}$ rescaling keeps the normalizer exact ([[activation-softmax]])
> $\ell$ tracks $\sum_{i \in \text{old}} e^{x_i - m}$, so $\sum_{i \in \text{old}} e^{x_i} = \ell \cdot e^m$.
> After updating the max to $m'$, all exponentials must be re-expressed relative to the new baseline:
> $\sum_{i \in \text{old}} e^{x_i - m'} = \ell \cdot e^m \cdot e^{-m'} = \ell \cdot e^{m - m'}$.
> Adding the new block's contribution $\ell_{\text{new}} \cdot e^{m_{\text{new}} - m'}$ gives $\ell'$ — the exact same sum as if all scores had been processed together from the start.

This is mathematically equivalent to standard softmax applied to the full sequence — no approximation.

### The Tiling Algorithm

Split $Q, K, V$ into blocks fitting in SRAM (block size $B_r \times B_c$). For each query block $Q_i$:
1. Load $Q_i$ to SRAM. Initialize $O_i = 0$, $\ell_i = 0$, $m_i = -\infty$.
2. For each key/value block $K_j, V_j$:
   - Load to SRAM (overwrites previous block).
   - Compute scores $S_{ij} = Q_i K_j^T / \sqrt{d_k}$.
   - Update $m_i$, $\ell_i$, $O_i$ via online softmax.
3. Rescale $O_i$ by $1/\ell_i$ → final attention output.
4. Write $O_i$ to HBM **once**.

The $N \times N$ attention matrix never touches HBM. Only the $N \times d$ output is written.

### Complexity

| | Standard | Flash Attention |
|--|----------|----------------|
| HBM reads/writes | $O(N^2)$ | $O(N^2 d / M)$ where $M$ = SRAM |
| Memory (attention matrix) | $O(N^2)$ | $O(N)$ — not stored |
| FLOPs | $O(N^2 d)$ | $O(N^2 d)$ — **exact same computation** |
| Empirical speedup (A100) | baseline | **2–4×** |

The speedup comes entirely from reducing HBM traffic. Zero approximation.

---

## Advanced

### Backward Pass: Recomputation as Checkpointing

Standard attention stores the $N \times N$ attention matrix for the backward pass (needed to compute softmax gradients). Flash Attention does not.

Instead, during the backward pass, it **recomputes** the attention weights by re-running the forward-pass tiling. This trades ~2× extra FLOPs for eliminating $O(N^2)$ activation storage — equivalent to [[backpropagation-advanced|gradient checkpointing]] applied to the attention operation itself.

For training, this is almost always a win: GPU memory is the bottleneck, not FLOPs. Freeing the $O(N^2)$ attention matrix allows significantly larger batch sizes or longer sequences.

### FlashAttention-2 Improvements

FlashAttention-2 (Dao, 2023) achieves additional gains:
1. **Reduced non-matmul FLOPs:** simplifies the online softmax rescaling to fewer operations.
2. **Parallelism over sequence length:** original FA1 parallelizes over batch and heads only; FA2 also parallelizes over the $N$ dimension, enabling better GPU utilization for long sequences.
3. **Warp specialization:** better distribution of work across GPU warps reduces shared memory synchronization.

Result: ~2× speedup over FlashAttention-1; reaches ~70% of theoretical A100 peak FLOP throughput for attention.

### Practical Impact

- **Context length:** standard attention makes training at $N > 2048$ prohibitively memory-intensive. Flash Attention makes $N = 32768$ (and beyond) feasible on a single GPU.
- **Modern adoption:** all major training frameworks (Megatron-LM, Hugging Face, PyTorch `F.scaled_dot_product_attention`) use Flash Attention by default.
- **Short sequences:** at $N = 512$, the speedup is modest (~1.3×) because the $N \times N$ matrix already fits in cache. The benefit scales with $N$.

### Comparison with Approximate Methods

| Method | Exact? | HBM Memory | Approach |
|--------|:---:|:---:|---------|
| Flash Attention | ✓ | $O(N)$ | IO-aware tiling |
| Linformer | ✗ | $O(N)$ | Low-rank $K, V$ approximation |
| Performer | ✗ | $O(N)$ | Random feature softmax approximation |
| Longformer | Partial | $O(Nw)$ | Sparse local + global attention |

Flash Attention is unique: **exact** computation with $O(N)$ memory footprint.

---

*See also: [[attention-mechanism]] · [[arch-kv-cache]] · [[vision-transformer]] · [[backpropagation-advanced]] · [[activation-softmax]]*
