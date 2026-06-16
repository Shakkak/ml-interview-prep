---
title: Sparse Attention
tags: [sparse-attention, longformer, bigbird, local-attention, efficient-transformers]
aliases: [sparse attention, local attention, strided attention, BigBird, Longformer, block sparse attention]
difficulty: 2
status: complete
depends_on: [attention-mechanism, flash-attention]
related: [attention-mechanism, flash-attention, arch-kv-cache, arch-positional-encoding, state-space-models]
---

# Sparse Attention

---

## Fundamental

### The Quadratic Bottleneck

Standard self-attention computes all $N^2$ pairwise similarities between $N$ tokens. For long sequences:
- $N = 4096$: $16M$ attention weights
- $N = 16384$: $268M$ attention weights  
- $N = 100000$: practically infeasible with dense attention

**Intuition:** most tokens only need local context — "the" doesn't need to attend to a token 10,000 positions away. Sparse attention keeps the long-range connections that matter (via global tokens or strided patterns) and drops the local pairs that are redundant, cutting quadratic cost to near-linear.

**Sparse attention** restricts which token pairs attend to each other, reducing complexity from $O(N^2)$ to $O(N \cdot k)$ where $k \ll N$ is the average attention span.

### Attention Patterns

**Local attention (sliding window):** each token attends only to its $w$ nearest neighbors:
$$\text{attend}(i, j) = 1 \text{ if } |i - j| \leq w/2$$

Complexity: $O(Nw)$. Good for local dependencies (text, audio). Misses long-range dependencies entirely.

**Strided / dilated attention:** attend to every $s$-th position beyond the local window:
$$\text{attend}(i, j) = 1 \text{ if } j \equiv i \pmod{s} \text{ or } |i-j| \leq w/2$$

Captures regular long-range patterns (periodic structure) without covering all pairs.

**Global tokens:** a few special tokens (e.g., `[CLS]`, section headings) attend to all positions and receive attention from all. These aggregate global context for the sparse local tokens.

---

## Intermediate

### Longformer

**Longformer (Beltagy et al., 2020):** combines local windowed attention + global tokens:
- All tokens: local attention with window size $w$ (default 512)
- Task-specific tokens (e.g., `[CLS]`, question tokens): global attention to all positions
- Complexity: $O(N \cdot (w + g))$ where $g$ = number of global tokens

Fine-tuned from RoBERTa; handles sequences up to 4096 tokens for document-level NLP tasks.

### BigBird

**BigBird (Zaheer et al., 2020):** local + global + random attention:
- **Local:** attend to $w$ nearest neighbors
- **Global:** $g$ global tokens (like Longformer)  
- **Random:** $r$ randomly sampled positions per query

Theoretical result: BigBird is a universal approximator of sequence-to-sequence functions (like full attention) with $O(N)$ complexity — random connections provide provably sufficient global coverage.

**BigBird preserves:** tasks requiring long-range reasoning (genome classification, question answering over long documents) while being 8× more efficient than BERT at sequence length 4096.

### Block Sparse Attention

Instead of token-level sparsity, operate on **blocks** of tokens:
- Divide sequence into blocks of size $b$
- Each block attends to: local blocks + $k$ randomly selected distant blocks
- Complexity: $O(N \cdot (b + kb)) = O(Nb)$

**GPT-3 sparse attention:** uses block-sparse attention with local + strided patterns in some layers, dense attention in others.

---

## Advanced

### Learned Sparsity

Rather than fixed patterns, **let the model learn** which pairs to attend to:

**Reformer (Kitaev et al., 2020):** locality-sensitive hashing (LSH) groups similar queries and keys into buckets; each query only attends to keys in its bucket. Complexity: $O(N \log N)$. Works when keys and queries are naturally similar to each other (self-attention).

**Routing Transformers:** learn a router that assigns tokens to "attention clusters." Tokens in the same cluster attend to each other. Iteratively refined during training.

**Sparse Transformers (Child et al., 2019):** alternating layers of local + strided attention. First paper showing sparse attention works well in practice for image generation (GPT-2-scale on sequences of 12,288 pixels).

### Comparison with Linear Attention

| Approach | Complexity | Quality | Long-range |
|----------|-----------|---------|-----------|
| Dense attention | $O(N^2)$ | Best | Yes |
| Sparse (local+global) | $O(N \cdot k)$ | Near-dense | Via globals |
| Linear attention | $O(N)$ | Worse | Approximate |
| SSMs (Mamba) | $O(N)$ | Competitive | Via state |

Sparse attention is preferable when exact attention is needed on a subset of positions (retrieval, reasoning). SSMs and linear attention are better for streaming/unlimited context without retrieval.

## Links

- [[attention-mechanism]] — sparse attention restricts the full $O(N^2)$ attention to a subset of positions; the attention computation is otherwise identical
- [[flash-attention]] — Flash Attention computes full attention IO-efficiently; sparse attention reduces the number of pairs attended to; both target long-sequence scalability
- [[state-space-models]] — SSMs (Mamba) are the leading alternative to sparse attention for long sequences; they achieve linear complexity via recurrent state rather than local windows
- [[arch-kv-cache]] — sparse attention patterns (local windows, strided) reduce the effective KV cache entries attended at each step during inference
- [[arch-positional-encoding]] — sparse attention patterns like sliding window require relative position encodings; absolute encodings don't capture distance in local windows well
