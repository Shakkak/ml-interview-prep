---
title: Model Compression
tags: [model-compression, pruning, quantization, distillation, low-rank, efficiency]
aliases: [model compression, neural network compression, efficient inference, compression techniques]
difficulty: 1
status: complete
related: [lora-quantization, knowledge-distillation, pruning-sparsity, mixed-precision, lora-quantization]
---

# Model Compression

---

## Fundamental

### Why Compress?

Large models (GPT-4, LLaMA-70B) have too many parameters for deployment on edge devices, low-latency serving, or cost-constrained environments. Compression reduces:
- **Memory footprint** (model size)
- **Inference latency** (fewer FLOPs)
- **Serving cost** (less hardware)

The four main families of compression are orthogonal and can be combined.

---

## Intermediate

### The Four Families

**1. Pruning:** remove weights, neurons, or layers.
- Unstructured: zero out individual weights (irregular sparsity) — reduces storage but requires sparse hardware for speedup
- Structured: remove entire filters, heads, or layers — directly reduces architecture size, hardware-friendly
- See [[pruning-sparsity]] for details

**2. Quantization:** reduce numerical precision of weights/activations.
- Post-training quantization (PTQ): convert FP32 → INT8/INT4 after training — fast but quality loss
- Quantization-aware training (QAT): simulate quantization during training — better quality, slower to set up
- GPTQ, AWQ, GGUF: popular PTQ methods for LLMs
- See [[lora-quantization]] (quantization section)

**3. Knowledge Distillation:** train a small student model to mimic a large teacher.
- Soft labels: student matches teacher's output probabilities (softer, more informative than hard labels)
- Feature distillation: student matches intermediate representations
- See [[knowledge-distillation]]

**4. Low-Rank Factorization:** decompose weight matrices into products of smaller matrices.
- $W \approx UV^\top$ where $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, $r \ll \min(m,n)$
- LoRA (see [[lora-quantization]]): train only the low-rank delta $\Delta W = AB$
- SVD-based compression: truncate singular values below threshold

### Comparison

| Method | When best | Speedup type | Quality impact |
|--------|-----------|-------------|----------------|
| Unstructured pruning | Storage constrained | Minimal (no sparse HW) | Low at 50–80% sparsity |
| Structured pruning | Latency constrained | Direct (smaller model) | Moderate |
| INT8 quantization | Any deployment | 1.5–2× throughput | Minimal |
| INT4 quantization | Memory constrained | 2–3× throughput | Small with calibration |
| Distillation | Flexibility needed | Smaller model | Depends on student size |
| Low-rank (LoRA) | Fine-tuning only | None at inference | Minimal for PEFT |

---

## Advanced

### Combining Methods

In practice, methods are stacked:
- **LLM.int8() + LoRA:** quantize the base model to INT8, add LoRA adapters in FP16 — fine-tune efficiently
- **SparseGPT + GPTQ:** prune 50% of weights + quantize to 4 bits — 10× compression with modest quality loss
- **Distillation → Quantization:** distill to a medium model, then quantize — two-stage approach for extreme compression

**Rule of thumb for LLMs:**
- 2–4× compression needed: quantization (INT8 or INT4)
- 5–10× compression needed: quantization + structured pruning
- 10–100× compression needed: distillation to a smaller architecture

### Compression for Specific Scenarios

**Edge deployment (mobile, IoT):** structured pruning + INT8, ideally with hardware-aware NAS to find the right architecture from scratch.

**Long-context serving:** KV cache quantization (INT4) + Flash Decoding — memory savings directly enables longer context windows at the same hardware cost.

**Fine-tuning at scale:** LoRA + gradient checkpointing + BF16 — fine-tune 70B models on 1–2 A100s.

*See also: [[pruning-sparsity]] · [[knowledge-distillation]] · [[lora-quantization]] · [[mixed-precision]]*
