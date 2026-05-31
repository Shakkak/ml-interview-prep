---
title: KV Cache
tags: [architecture, transformers, inference, efficiency]
aliases: [KV cache, key-value cache, inference cache]
difficulty: 2
status: complete
related: [attention-mechanism, arch-positional-encoding]
---

# KV Cache

---

## Fundamental

In autoregressive transformers, generating token $t+1$ requires attending to all previous tokens $1, \ldots, t$. Without caching, to generate token $t+1$ the full sequence $[x_1, \ldots, x_t]$ is re-processed through all transformer layers — recomputing K and V for every position at every step. Generating a 1000-token sequence requires:
$$1 + 2 + \cdots + 999 = 499{,}500 \text{ forward-pass token computations}$$
Quadratic in sequence length.

**The key insight:** for positions already computed, the K and V projections do not change when generating new tokens (they depend only on past context, which is fixed after the prompt). Cache them.

**KV cache operation:**
1. For the prompt, compute K, V once and store per layer.
2. When generating token $t+1$: compute Q, K, V for the *new token only* (one row).
3. Append new K, V rows to the cache.
4. Compute attention: $Q_{t+1}$ attends over full cached $K_{1:t+1}$ and $V_{1:t+1}$.

Each new token: one forward pass, one token's worth of projection.

---

## Intermediate

### Memory Cost

$$\text{Memory} = 2 \times L \times d_k \times H \times T \times \text{bytes}$$

where $2$ = K and V, $L$ = layers, $H$ = heads, $T$ = sequence length.

**LLaMA-7B** ($L=32$, $d_k=128$, $H=32$, float16 = 2 bytes):

| Sequence length | KV cache size |
|-----------------|---------------|
| 4,096 | $2 \times 32 \times 128 \times 32 \times 4096 \times 2 = $ **2 GB** |
| 32,768 | **16 GB** |

LLaMA-7B weights ≈ 14 GB in float16. A long-context KV cache can *exceed* the model weights.

### Numerical Example: Cache vs No Cache

Generating 100 tokens from a 10-token prompt:

| Method | K,V computations | Savings |
|--------|-----------------|---------|
| No cache | $11 + 12 + \cdots + 110 = 6{,}050$ | — |
| With cache | $10 \text{ (prompt)} + 100 \text{ (new tokens)} = 110$ | **55×** |

### Prefill vs Decode

**Prefill phase:** process the prompt (all tokens at once). Compute-bound — GPU utilization is high, attention over all pairs.

**Decode phase:** generate one token at a time. Memory-bandwidth bound — the bottleneck is reading the growing KV cache from GPU memory, not computation. This is why token generation throughput is limited by memory bandwidth, not FLOPS.

Modern inference systems (TensorRT-LLM, vLLM) use different optimization strategies for each phase: chunked prefill to pipeline compute, paged KV attention (vLLM) to prevent memory fragmentation.

---

## Advanced

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

**Standard MHA:** each of $H$ heads has its own K, V projections. KV cache size $\propto H$.

**MQA (Shazeer, 2019):** share one K, V head across all $H$ query heads.
- KV cache: $H$ times smaller (e.g., 32× for 32 heads)
- Quality: small degradation (~0.5% on benchmarks)

**GQA (Ainslie et al., 2023):** share K, V among groups of $G$ query heads.
- $G = 1$: MQA. $G = H$: standard MHA.
- Cache: $\frac{H}{G}$ times smaller

Used in: LLaMA-2 70B (GQA, $G=8$), Mistral, Gemma. Enables practical long-context inference.

**Example:** LLaMA-2 70B, 32K context, $H=64$ heads, GQA with $G=8$:
$$\text{KV cache} = \frac{64}{8} = 8 \text{ head pairs} \times 32 \times 128 \times 32768 \times 2 \approx 17 \text{ GB (vs 136 GB standard)}$$

### Speculative Decoding

Use a small "draft" model to speculatively generate $k$ tokens in parallel, then verify all $k$ with the large model in one forward pass. Accept/reject each token using a probability ratio test that preserves the target distribution exactly.

Speedup: $1.5$–$3\times$ for tasks with predictable continuations (code, structured text). The key insight: the large model processes $k$ tokens at once (parallel, efficient) instead of one at a time.

### KV Cache Compression

**Quantization:** INT4 KV cache halves memory at small quality cost.

**Token eviction:** drop K, V entries for past tokens based on importance heuristics (attention scores, recency). StreamingLLM maintains a "sink" of early tokens (which receive high attention) plus a rolling window. Enables unbounded context with bounded memory.

**Prefix caching:** if many requests share the same system prompt, cache its K, V computation once and reuse across requests. vLLM implements this as automatic prefix caching.

---

*See also: [[attention-mechanism]] · [[arch-positional-encoding]] · [[flash-attention]]*
