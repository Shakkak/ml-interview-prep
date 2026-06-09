---
title: Mixed Precision Training (FP16 / BF16 / AMP)
tags: [mixed-precision, fp16, bf16, amp, loss-scaling, numerical-precision, training-efficiency]
aliases: [mixed precision, AMP, automatic mixed precision, fp16 training, bf16 training, half precision]
difficulty: 2
status: complete
related: [optimizer-adam, large-batch-training, normalization-layers, backpropagation-advanced]
---

# Mixed Precision Training (FP16 / BF16 / AMP)

---

## Fundamental

Modern GPUs have tensor cores that run FP16/BF16 matrix multiplications **2–8× faster** than FP32, and reducing precision halves activation memory, enabling larger batch sizes or models.

**Float format comparison:**

| Format | Exponent bits | Mantissa bits | Max value | Min positive | Precision |
|--------|--------------|---------------|-----------|--------------|-----------|
| FP32 | 8 | 23 | ~3.4×10³⁸ | ~1.2×10⁻³⁸ | ~7 decimal digits |
| FP16 | 5 | 10 | 65504 | ~6.1×10⁻⁵ | ~3 decimal digits |
| BF16 | 8 | 7 | ~3.4×10³⁸ | ~1.2×10⁻³⁸ | ~2 decimal digits |

**Key insight:** FP16 has a narrow exponent range — it overflows above 65504 and underflows below ~6×10⁻⁵. Gradients in deep nets commonly fall in $10^{-6}$ to $10^{-3}$ — well below FP16's minimum, causing gradient underflow to zero and stalled training. BF16 has the same exponent range as FP32, eliminating overflow and underflow entirely at the cost of mantissa precision.

**The mixed precision recipe** (Micikevicius et al., 2018):
1. **FP16 forward and backward pass:** all matrix multiplications, convolutions, and activations in FP16.
2. **FP32 master weights:** a full-precision copy of all parameters maintained in FP32; updated after each step; copied to FP16 for the next forward pass. This preserves small weight updates that would be lost in FP16.
3. **Loss scaling:** multiply the loss by constant $S$ before backprop, divide gradients by $S$ before the optimizer step. Shifts small gradients out of the FP16 underflow range.

---

## Intermediate

**Loss scaling procedure:**
1. Compute loss $L$ in FP16.
2. Multiply: $L_\text{scaled} = S \cdot L$ (typically $S = 2^{10}$ to $2^{15}$).
3. Backpropagate through $L_\text{scaled}$ — gradients are $S\times$ larger.
4. Unscale gradients: divide by $S$.
5. Check for overflow (inf/nan). If found: skip this step, reduce $S$.

**Dynamic loss scaling (preferred):** start with large $S$, halve whenever overflow detected, double every $N$ successful steps. Automatically finds the right scale.

**Weight update loss example:** $w = 1024.0$, $\Delta w = 0.001$. In FP16, $1024.0 + 0.001 = 1024.0$ (update entirely lost due to limited mantissa bits). In FP32, the update is preserved. This is why FP32 master weights are critical.

**Operations that must stay in FP32:**

| Operation | Why FP32 needed |
|-----------|----------------|
| [[activation-softmax\|Softmax]] | Exponentials can overflow in FP16 |
| Log-sum-exp | Same overflow risk |
| [[normalization-layers\|Batch norm]] statistics (mean/var) | Accumulation of many small values loses precision |
| Loss computation | Final reduction of many terms |
| Optimizer step | Operates on FP32 master weights |

Frameworks (PyTorch AMP, JAX, TF AMP) handle this automatically via op whitelisting.

**BF16 simplifies implementation:** same exponent as FP32 → no loss scaling needed, no overflow/underflow risk. Supported on Google TPUs, NVIDIA A100/H100, AMD MI200+. In practice, BF16 often matches FP16 in final accuracy while being less fragile. Use FP16 only for older hardware (V100, T4) that lacks native BF16 support.

**PyTorch AMP usage:**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()  # dynamic loss scaling

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():                          # FP16/BF16 inside
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                    # skip if inf/nan
    scaler.update()                           # adjust scale factor
```

Always unscale gradients before clipping — clipping scaled gradients clips against the wrong magnitude.

---

## Advanced

**Memory analysis for a 1B parameter model with Adam:**

| Component | FP32 | Mixed FP16 | Mixed BF16 |
|-----------|------|-----------|-----------|
| Weights (active) | 4 GB | 2 GB | 2 GB |
| Weights (master) | — | 4 GB | 4 GB |
| [[optimizer-adam\|Adam]] $m_1$ (FP32) | 4 GB | 4 GB | 4 GB |
| Adam $m_2$ (FP32) | 4 GB | 4 GB | 4 GB |
| Activations | 4 GB | 2 GB | 2 GB |
| **Total** | **16 GB** | **12 GB** | **12 GB** |

Optimizer states dominate and stay in FP32 regardless — the 25% savings comes mainly from activations. For very large models, ZeRO-1/2/3 shards optimizer states across GPUs, multiplying the effective memory savings.

**FP8 training** (emerging): NVIDIA H100 supports FP8 (E4M3 and E5M2 formats) via the Transformer Engine. E4M3 (4-bit exponent, 3-bit mantissa) is used for activations and weights; E5M2 is used for gradients. Requires per-tensor or per-block scaling (similar to loss scaling but applied per tensor). Scaling factors must be tracked and updated per step. Narayanan et al. (2021) demonstrated FP8 training matching FP16 for GPT-3-class models. Key challenge: attention layers are numerically sensitive and may require FP16 fallback for softmax.

**Gradient underflow vs [[backpropagation-advanced|vanishing gradients]]:** these are related but distinct phenomena. Vanishing gradients arise from the mathematical structure of the network (activation saturation, deep chains of Jacobians). Gradient underflow arises purely from the numerical format — a gradient of $10^{-7}$ is mathematically valid but rounds to 0 in FP16. Loss scaling addresses underflow without affecting the vanishing gradient problem, which requires architectural solutions (residual connections, normalization, activation choice).

**ZeRO and mixed precision for trillion-parameter models:** the ZeRO (Zero Redundancy Optimizer) family of memory optimizations (Rajbhandari et al., 2020) partitions optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data-parallel GPUs. Combined with BF16 mixed precision, ZeRO-3 achieves near-linear scaling for models with 100B+ parameters across thousands of GPUs. The memory cost per GPU for ZeRO-3 is $O(1/N_{GPUs})$ for optimizer states, making trillion-parameter training feasible without model parallelism. DeepSpeed implements ZeRO in practice.

---

*See also: [[optimizer-adam]] · [[large-batch-training]] · [[normalization-layers]] · [[backpropagation-advanced]]*
