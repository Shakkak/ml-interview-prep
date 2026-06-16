---
title: GGUF and k-Quant Quantization
tags: [gguf, ggml, quantization, k-quant, llm-inference, cpu-inference]
aliases: [GGUF, GGML, k-quants, Q4_K_M, llama.cpp, quantization formats]
difficulty: 1
status: complete
depends_on: [lora-quantization, linear-algebra-fundamentals]
related: [lora-quantization, mixed-precision, model-compression, autoregressive-models, arch-kv-cache]
---

# GGUF and k-Quant Quantization

---

## Fundamental

### The Problem

Running large language models locally requires fitting model weights in RAM/VRAM. A 7B parameter model at FP32 requires 28 GB; at FP16, 14 GB. Most consumer hardware has 8–24 GB of memory.

**Quantization** reduces precision: instead of 16-bit floats, store weights as 4-bit or 8-bit integers. A 7B model at 4-bit needs only ~4 GB — runnable on most modern GPUs and even CPUs.

**Intuition:** trained neural network weights don't need full floating-point precision to produce accurate predictions. Most of the "meaning" in a weight matrix is captured by its rough magnitude and sign. Quantization replaces each 32-bit or 16-bit float with a small integer (4 or 8 bits), storing a single scale factor per group of weights to convert back. Some precision is lost in the rounding, but the model retains most of its capability because the loss landscape is relatively flat near the trained minimum — small weight perturbations don't change predictions much.

### GGML and GGUF

**GGML (Georgi Gerganov's ML library):** a C-based tensor library for efficient CPU inference. The original format for quantized LLaMA models.

**GGUF (GGML Unified Format, 2023):** replaced GGML with a more flexible binary format:
- Single file containing model weights + all metadata (tokenizer, architecture config, quantization type)
- Supports partial loading (memory-map specific tensors)
- Backward compatible as new quantization types are added
- Supported by llama.cpp, Ollama, LM Studio, Jan, and most local LLM tools

---

## Intermediate

### k-Quant Families

**k-quants** are GGUF quantization schemes that quantize blocks of weights together, allowing "important" blocks to use higher precision:

| Format | Bits per weight | Size (7B) | Quality |
|--------|----------------|-----------|---------|
| `Q8_0` | 8 | ~7 GB | Near-lossless |
| `Q6_K` | 6 | ~5.5 GB | Excellent |
| `Q5_K_M` | 5.5 (mixed) | ~5 GB | Very good |
| `Q4_K_M` | 4.5 (mixed) | ~4.1 GB | Good, recommended |
| `Q4_K_S` | 4.4 (small) | ~3.9 GB | Slightly lower |
| `Q3_K_M` | 3.9 | ~3.3 GB | Acceptable |
| `Q2_K` | 2.6 | ~2.7 GB | Noticeable degradation |

**`_M` vs `_S`:** mixed (`_M`) uses higher precision for "sensitive" layers (embedding layers, output layers); small (`_S`) uses uniform low precision everywhere. `_M` is better quality; `_S` is smaller.

### How k-Quants Work

Weights are grouped into blocks of $k$ values (e.g., $k = 256$). For each block:
1. Compute the min and max (or a quantile range) of the block
2. Store one or two scale values per block (quantized to 6 or 8 bits)
3. Quantize each weight relative to the block scale

**Advantage over naive quantization:** block-wise scaling allows different regions of the weight matrix to use different effective ranges, reducing rounding error for weights with high variance.

**Importance-weighted quantization (GPTQ, AWQ):** for even higher quality, apply the quantization using calibration data to minimize the output error rather than just minimizing weight error. These methods (requiring GPU) produce better 4-bit models than naive k-quants but are more complex to run.

---

## Advanced

### Perplexity vs Compression

The standard quality measure for quantized LLMs is **perplexity** on a standard text corpus (Wikitext-2 or similar). Perplexity measures how surprised the model is by real text: $\text{PPL} = \exp(-\frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i | x_{<i}))$, where the average negative log-likelihood over $N$ tokens is exponentiated — a model that assigns high probability to real text has low perplexity. Lower perplexity = better language model.

**Typical perplexity increase over FP16:**
- Q8_0: <0.1% increase (essentially lossless)
- Q4_K_M: ~1–2% increase (barely noticeable in practice)
- Q3_K_M: ~3–5% increase (noticeable on hard tasks)
- Q2_K: 10–20%+ increase (significant degradation)

**Practical threshold:** Q4_K_M is the sweet spot — 4× compression with minimal quality loss. Most practitioners use Q4_K_M or Q5_K_M as the default.

### Mixed Precision Quantization

Modern approaches quantize different parts of the model differently:
- **Embedding layers:** stored at higher precision (FP16 or Q8) — small number of parameters, high sensitivity
- **Attention weights:** Q4 or Q5
- **FFN weights:** Q4 (largest, most compressible)
- **Output logits layer:** FP16 — directly affects token probabilities

This mixed approach is what `_M` formats do automatically. More aggressive mixed precision (per-layer bit assignment based on sensitivity) is used by SpQR, QuIP#, and similar research methods achieving near-lossless 4-bit quantization.

## Links

- [[lora-quantization]] — QLoRA combines GGUF-style 4-bit quantization with LoRA adapters; the quantization reduces memory, the adapters enable fine-tuning
- [[linear-algebra-fundamentals]] — quantization maps float32/float16 weight matrices to low-bit integers; the quantization bins partition the weight value range
- [[mixed-precision]] — GGUF uses mixed precision within a block: scale factors and zeros in float16/float32, activations in float16/bfloat16
- [[model-compression]] — GGUF is one of several model compression techniques; others include pruning, knowledge distillation, and structured quantization
- [[autoregressive-models]] — GGUF enables autoregressive LLaMA/Mistral inference on consumer CPUs and mobile devices via llama.cpp
- [[arch-kv-cache]] — quantized KV cache (K-quant for the cache) further reduces inference memory; GGUF supports quantized KV by default in llama.cpp
