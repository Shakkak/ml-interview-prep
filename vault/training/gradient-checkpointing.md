---
title: Gradient Checkpointing
tags: [gradient-checkpointing, activation-recomputation, memory-efficiency, training, backpropagation]
aliases: [gradient checkpointing, activation recomputation, rematerialization, activation checkpointing]
difficulty: 1
status: complete
related: [backpropagation, mixed-precision, large-batch-training, normalization-layers, lora-quantization]
depends_on: [backpropagation, mixed-precision]
---

# Gradient Checkpointing

---

## Fundamental

### The Memory Problem in Backpropagation

During a forward pass, neural networks store all intermediate activations — they are needed during the backward pass to compute gradients. For a network with $L$ layers and batch size $B$:

$$\text{Activation memory} \approx L \times B \times d \times \text{bytes}$$

where $L$ = number of layers, $B$ = batch size, $d$ = hidden dimension, bytes = 2 for FP16 or 4 for FP32.

For a GPT-2 Large (36 layers, $d = 1280$) with batch 8 and sequence 1024 in FP16: $\approx 36 \times 8 \times 1024 \times 1280 \times 2 \approx$ **1.2 GB** just for activations. For larger models, activations dominate memory usage.

### The Checkpointing Trick

**Gradient checkpointing** (Chen et al., 2016) trades compute for memory:

- **Forward pass:** only save activations at $\sqrt{L}$ "checkpoint" locations (not all $L$)
- **Backward pass:** recompute the activations between checkpoints as needed by re-running the forward pass for each segment

**Memory:** $O(\sqrt{L})$ activation storage instead of $O(L)$
**Compute:** $O(1/3)$ extra forward compute (each layer computed ~1.33 times on average)

For $L = 36$ layers: store $\sqrt{36} = 6$ checkpoints instead of 36 — 6× memory reduction at 33% compute overhead.

---

## Intermediate

### Practical Usage

**PyTorch:**
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Recompute this block during backward instead of storing activations
    x = checkpoint(self.transformer_block, x)
    return x
```

Each `checkpoint` call: activations of the wrapped function are not stored after the forward pass. During backward, the function is re-run from the checkpoint inputs.

**Granularity:** checkpoint at transformer layer boundaries (most common), or at finer granularity (individual attention/FFN sublayers) for extreme memory savings.

**Memory savings vs compute cost:**

| Checkpointing level | Memory reduction | Compute overhead |
|--------------------|-----------------|-----------------|
| Every $\sqrt{L}$ layers | $\sim \sqrt{L}\times$ | ~33% |
| Every layer | $L\times$ | $\sim 2\times$ |
| Every sublayer | $2L\times$ | $\sim 3\times$ |

### Interaction with Mixed Precision

With FP16/BF16 training (see [[mixed-precision]]), activations are stored in low precision. Checkpointing reduces the number of activations stored; mixed precision reduces their size. Combined, they allow training models 4–8× larger than baseline for a given GPU memory budget.

**Order of impact:** for a 7B parameter model, enabling both gradient checkpointing + BF16 can reduce activation memory from ~40 GB to ~5 GB — making single-GPU fine-tuning feasible.

---

## Advanced

### Optimal Checkpointing Schedule

The $\sqrt{L}$-checkpoint strategy is optimal (in terms of memory-compute tradeoff) for uniform layer cost. For non-uniform layers (e.g., early layers are cheaper), optimal checkpointing saves more expensive layers' activations preferentially.

**Selective checkpointing:** in practice, only apply checkpointing to expensive layers (transformer blocks) but not cheap ones (LayerNorm, residual additions). This balances memory savings against compute overhead.

### Checkpointing for Activation-Heavy Operations

Beyond standard layers, some operations are particularly activation-hungry:
- **Flash Attention:** recomputes attention softmax during backward from Q, K, V (stored) — no O/S activations stored. Effectively free checkpointing for attention.
- **Custom CUDA kernels:** fused operations (e.g., SwiGLU activation in one kernel) can be recomputed cheaply if the input activations are available.

This is why Flash Attention + gradient checkpointing together is the standard memory-efficient training recipe for transformers.

## Links

- [[backpropagation]] — standard backpropagation stores all intermediate activations for the backward pass; gradient checkpointing discards and recomputes them, trading compute for memory
- [[mixed-precision]] — FP16 halves activation memory; gradient checkpointing further reduces peak memory; together they enable training much larger models on the same GPU
- [[large-batch-training]] — gradient checkpointing enables larger effective batch sizes by reducing activation memory; it is commonly combined with gradient accumulation
- [[flash-attention]] — Flash Attention avoids storing the $n\times n$ attention matrix by tiling and recomputing; it is effectively gradient checkpointing applied to the attention operation
