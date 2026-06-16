---
title: Grouped Query Attention (GQA)
tags: [gqa, mqa, multi-head-attention, kv-cache, inference-efficiency]
aliases: [GQA, grouped query attention, multi-query attention, MQA, KV-sharing]
difficulty: 1
status: complete
depends_on: [attention-mechanism, arch-kv-cache]
related: [attention-mechanism, arch-kv-cache, flash-attention, autoregressive-models, rotary-embeddings]
---

# Grouped Query Attention (GQA)

---

## Fundamental

### Memory Bottleneck at Inference

**Intuition:** multi-head attention uses separate key/value matrices per head because heads are meant to attend to different aspects of the input. But in practice, many heads learn very similar key/value projections — you can share them across groups of heads with negligible quality loss and dramatic memory savings. GQA is the middle ground: not one shared KV (MQA, too lossy) and not one KV per head (MHA, too expensive), but one KV per group of heads.

In standard Multi-Head Attention (MHA), each head $i$ has independent keys $K_i$ and values $V_i$. The KV cache stores all these for previously generated tokens:

$$\text{KV cache size} = 2 \times n_\text{heads} \times L \times d_\text{head} \times \text{bytes per element}$$

For LLaMA-1 70B (96 heads, $d = 128$, sequence length 4096, FP16): 96 × 4096 × 128 × 2 × 2 bytes ≈ **200 GB** per sequence. This bottleneck limits batch size and throughput more than compute.

### Multi-Query Attention (MQA)

**MQA (Shazeer, 2019):** use a single shared K and V head for all query heads:

$$q_i = x W_{Q_i}, \quad k = x W_K, \quad v = x W_V \quad \text{(shared across all } i\text{)}$$

$$\text{Attn}_i = \text{softmax}\!\left(\frac{q_i k^\top}{\sqrt{d_k}}\right) v$$

**KV cache reduction:** from $n_h$ KV pairs → $1$ KV pair. 96× memory reduction for LLaMA-70B.

**Tradeoff:** quality degradation on long-context tasks, especially retrieval. The model can't use different "perspectives" (different K/V projections) per head.

---

## Intermediate

### Grouped Query Attention (GQA)

**GQA (Ainslie et al., 2023)** is a middle ground: $G$ groups of $n_h/G$ query heads share a single K/V head:

$$\text{MHA: } n_h \text{ Q heads, } n_h \text{ K heads, } n_h \text{ V heads}$$
$$\text{MQA: } n_h \text{ Q heads, } 1 \text{ K head, } 1 \text{ V head}$$
$$\text{GQA: } n_h \text{ Q heads, } G \text{ K heads, } G \text{ V heads}$$

KV cache reduction: $n_h/G \times$ smaller than MHA. For $G = 8$, $n_h = 32$: 4 KV heads (8× savings).

**Quality:** GQA significantly closes the quality gap with MHA while keeping most of MQA's memory savings. The paper shows GQA with $G=8$ matches MHA quality within 0.1% on most tasks.

| Variant | Quality | KV cache | 
|---------|---------|----------|
| MHA | Best | Largest ($n_h$ heads) |
| GQA | Near MHA | Medium ($G$ heads) |
| MQA | Moderate | Smallest (1 head) |

### Models Using GQA

- LLaMA-2 34B, 70B: GQA with $G=8$ (32 Q heads, 8 KV heads)
- LLaMA-3 all sizes: GQA
- Mistral 7B: GQA with $G=8$ (32 Q heads, 8 KV heads)
- Gemma, Phi-3, Qwen: GQA
- GPT-4 (estimated): likely MQA or GQA

---

## Advanced

### Converting MHA Models to GQA

GQA can be applied to an existing MHA-trained model without full retraining via **uptraining** (Ainslie et al., 2023):
1. Mean-pool the $n_h/G$ KV heads within each group to create $G$ merged heads
2. Fine-tune for a small number of steps (5% of original training compute)

The converted GQA model recovers ~99% of the original MHA quality.

### GQA and Flash Attention Interaction

Flash Attention (see [[flash-attention]]) works naturally with GQA — the tiled computation proceeds identically, just with fewer distinct K/V matrices. In Flash Attention 2, GQA is directly supported via broadcasting the shared KV heads across the grouped Q heads without additional memory.

### KV Cache and Quantization

The KV cache can also be quantized independently of model weights:
- KV cache in INT8 or INT4 reduces the 200 GB example to 50–100 GB
- Some degradation in long-context precision; key/value importance differs by position
- KVQuant and similar methods apply non-uniform quantization (more bits for important KV entries)

## Links

- [[attention-mechanism]] — GQA is a drop-in variant of multi-head attention; it shares key and value projections across groups of query heads
- [[arch-kv-cache]] — GQA's primary motivation is reducing KV cache size at inference; fewer KV heads means proportionally less cache memory per token
- [[flash-attention]] — Flash Attention and GQA compose directly; Flash Attention's tiling works with any number of KV heads
- [[rotary-embeddings]] — RoPE is applied to queries and keys before the grouped dot products; GQA does not change how position is encoded
- [[autoregressive-models]] — LLaMA-2, Mistral, and most production autoregressive models use GQA to balance quality vs. inference efficiency
