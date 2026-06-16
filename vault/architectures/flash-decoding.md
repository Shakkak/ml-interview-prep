---
title: Flash Decoding
tags: [flash-decoding, inference, long-context, kv-cache, attention-inference]
aliases: [Flash Decoding, FlashDecoding, long-context inference, parallel KV attention]
difficulty: 2
status: complete
depends_on: [flash-attention, arch-kv-cache, grouped-query-attention]
related: [flash-attention, arch-kv-cache, speculative-decoding, grouped-query-attention, autoregressive-models]
---

# Flash Decoding

---

## Fundamental

### The Inference Bottleneck Flash Attention Doesn't Solve

Flash Attention (see [[flash-attention]]) optimizes the training-time attention computation by tiling the attention matrix to avoid HBM reads/writes. It achieves near-peak flop utilization for the **prefill phase** (processing the prompt in parallel).

But during **token-by-token decoding**, each step generates one token by attending over the full KV cache. With batch size $B = 1$ (common in serving):
- The attention computation is: $q \in \mathbb{R}^{1 \times d}$ attends over $K, V \in \mathbb{R}^{L \times d}$ (KV cache of length $L$)
- This is a matrix-vector product: $O(Ld)$ per step
- With $L = 100K$ tokens: enormous KV cache to read from HBM

The bottleneck is **memory bandwidth** (reading the KV cache), not arithmetic. Standard Flash Attention is designed for the prefill case and doesn't help here.

**Intuition:** the fix is to parallelize across the KV cache length, the same way Flash Attention parallelizes across the batch. Split the KV cache into chunks, compute partial attention in parallel across chunks, then merge the partial results using the same online softmax trick — each chunk's contribution is re-scaled to the global max before combining.

---

## Intermediate

### Flash Decoding Algorithm

**Flash Decoding (Dao et al., 2023)** parallelizes the decoding attention across the sequence length dimension:

**Standard decoding:** one thread computes $\text{softmax}(q K^\top / \sqrt{d}) V$ sequentially over all $L$ KV positions.

**Flash Decoding:**
1. **Partition:** split the KV cache into $C$ chunks along the sequence dimension
2. **Parallel reduction:** each chunk independently computes partial attention outputs and the local softmax normalization constants (max and sum)
3. **Online softmax merge:** combine the partial results using the online softmax trick to compute the globally normalized output

The key math: for partial sums $m_c = \max_i s_{ci}$, $\ell_c = \sum_i e^{s_{ci} - m_c}$, $o_c = \sum_i e^{s_{ci} - m_c} v_i$, the global output is:

$$o = \frac{\sum_c \ell_c e^{m_c - m} o_c}{\sum_c \ell_c e^{m_c - m}}, \quad m = \max_c m_c$$

where $C$ = number of KV-cache chunks, $m_c$ = the maximum attention logit within chunk $c$ (used for numerically stable softmax), $\ell_c$ = the sum of exponentials $\sum_i e^{s_{ci} - m_c}$ within chunk $c$ (the local normalizer), $o_c$ = the unnormalized partial output $\sum_i e^{s_{ci} - m_c} v_i$ for chunk $c$, and $m = \max_c m_c$ = the global maximum across all chunks.

This merging is the same as the online softmax trick in Flash Attention, extended to independent chunks.

**Parallelism:** chunks run in parallel across CUDA thread blocks → GPU utilization scales with $L$, not just batch size.

### When Flash Decoding Wins

| Scenario | Standard decoding | Flash Decoding |
|----------|------------------|----------------|
| Short context ($L < 4K$) | Fast (KV fits in L2) | Marginal benefit |
| Long context ($L = 64K+$) | Memory-bandwidth bottleneck | **2–8× speedup** |
| Large batch ($B > 16$) | Already parallelized | Marginal benefit |
| Small batch ($B = 1$) serving | Underutilized GPU | **Big win** |

Flash Decoding is most impactful for long-context single-request serving — exactly the setting of RAG, document QA, and code generation over large files.

---

## Advanced

### Flash Decoding++ and Unified Flash Attention

**Flash Decoding++:** further optimizations:
- Asynchronous prefetch of KV cache chunks from HBM while processing previous chunks
- Partial softmax accumulation in registers, reducing SRAM pressure
- Merged softmax normalization for GQA (grouped query attention)

**Unified Flash Attention (UFA):** a single kernel that automatically selects Flash Attention (for prefill, high batch size) or Flash Decoding (for decode, long context, small batch) based on the query sequence length. The selection criterion: if $q_\text{len} > 1$ → Flash Attention; if $q_\text{len} = 1$ → Flash Decoding.

### Relationship to KV Cache Quantization

Flash Decoding accelerates the memory-bandwidth bottleneck of reading the KV cache. KV cache quantization (INT8/INT4) reduces the same bottleneck by shrinking the cache size. Both are complementary: a 4-bit KV cache + Flash Decoding can achieve 4–16× speedup for long-context decoding.

## Links

- [[flash-attention]] — Flash Attention targets training (memory IO during forward/backward); Flash Decoding targets inference (parallel KV-head reduction across sequence positions)
- [[arch-kv-cache]] — Flash Decoding operates on the KV cache directly; it parallelizes across the sequence length dimension which the KV cache accumulates
- [[grouped-query-attention]] — GQA reduces KV cache size; Flash Decoding improves throughput for long-context inference by parallelizing over the remaining KV entries
- [[speculative-decoding]] — both are inference acceleration techniques; Flash Decoding improves attention latency, speculative decoding reduces the number of attention calls
- [[autoregressive-models]] — Flash Decoding is most impactful for long-context autoregressive generation where the KV cache grows large and attention dominates latency
