---
title: Gradient Accumulation
tags: [gradient-accumulation, large-batch, memory, training, micro-batch]
aliases: [gradient accumulation, micro-batches, simulated large batch]
difficulty: 1
status: complete
related: [optimizer-adam, optimizer-sgd-momentum, large-batch-training, normalization-layers, mixed-precision]
depends_on: [backpropagation, optimizer-sgd-momentum, large-batch-training]
---

# Gradient Accumulation

---

## Fundamental

### The Memory Problem

Training with large batch sizes improves gradient quality and enables higher learning rates, but GPU memory limits the batch size that fits in a single forward-backward pass. For a 7B parameter model, even a single sequence of 4096 tokens may already consume most of GPU memory.

**Gradient accumulation** simulates a large effective batch without needing it to fit in memory: split the desired batch into $K$ **micro-batches**, run forward-backward on each, accumulate the gradients, then take a single optimizer step.

```
for step in range(total_steps):
    for micro in range(K):             # accumulate K micro-batches
        loss = model(batch[micro]) / K  # scale loss by 1/K
        loss.backward()                 # accumulate .grad
    optimizer.step()
    optimizer.zero_grad()
```

The effective batch size is $K \times \text{micro\_batch\_size}$.

---

## Intermediate

### Correctness Conditions

**Loss scaling:** divide the loss by $K$ before each backward pass (or equivalently, average gradients). Without this, each micro-batch contributes as if it were the full batch, and the accumulated gradient is $K\times$ too large.

**No optimizer step between micro-batches:** the optimizer state ($m_t$, $v_t$ in Adam) must only update once per logical step. If optimizer.step() is called $K$ times, the effective learning rate is $K\times$ higher than intended.

**Batch norm interaction:** BatchNorm computes statistics over the micro-batch, not the full logical batch. This means per-micro-batch statistics are noisier than if the full batch were available. Solutions:
- Use LayerNorm (independent of batch size) — preferred for transformers
- Use SyncBatchNorm across devices (if using multi-GPU)
- Accept noisier BN statistics (often fine in practice)

### Equivalence to Large Batches

For SGD, gradient accumulation over $K$ micro-batches with averaged loss is mathematically equivalent to one pass over the concatenated batch (assuming the same random samples, no batch norm). For Adam, it is approximately equivalent — the second moment estimate is computed on micro-batch statistics, not the full batch, introducing a small discrepancy.

### Combining with Mixed Precision

With FP16/BF16 training (see [[mixed-precision]]), gradients are in low precision. Gradient accumulation can cause underflow over many micro-batches if the individual gradients are small. **Loss scaling** (multiplying the loss by a large constant before backward, then dividing accumulated gradients before the optimizer step) prevents this.

PyTorch `GradScaler` handles this automatically when `autocast` is used.

---

## Advanced

### Multi-GPU Interaction

In multi-GPU data parallel training, each GPU runs on its own micro-batches and gradients are all-reduced across GPUs before the optimizer step. Gradient accumulation on each GPU independently before the all-reduce further reduces communication frequency:

- Without accumulation: all-reduce every forward-backward
- With $K$-step accumulation: all-reduce every $K$ forward-backwards

This can significantly reduce communication overhead for large $K$. In PyTorch DDP, use `model.no_sync()` context during accumulation steps to suppress premature all-reduces.

### Hyperparameter Scaling

When increasing effective batch size via accumulation:
- **Learning rate:** scale linearly with effective batch size (linear scaling rule) up to a limit, then use sqrt scaling
- **Warmup steps:** keep warmup in terms of optimizer steps (not micro-batch steps)
- **Weight decay:** unaffected (applied per optimizer step)

## Links

- [[backpropagation]] — gradient accumulation sums gradients over micro-batches before the optimizer step; gradients are accumulated in the same buffers used by backpropagation
- [[optimizer-sgd-momentum]] — accumulation simulates larger batch size; the effective batch size = micro-batch size × accumulation steps; learning rate should be scaled accordingly
- [[large-batch-training]] — gradient accumulation is the memory-constrained version of large-batch training; it achieves the same effective batch size without requiring the full batch in GPU memory simultaneously
- [[normalization-layers]] — batch norm statistics are computed per micro-batch with accumulation, not over the full effective batch; this is why sync-BN or layer norm is preferred in low-memory training
- [[mixed-precision]] — FP16 training with gradient accumulation requires careful loss scaling; the scaling factor must account for the accumulated gradients across all micro-batches
