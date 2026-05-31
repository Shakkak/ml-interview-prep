---
title: The Attention Mechanism
tags: [transformers, attention, deep-learning, nlp, vision]
aliases: [self-attention, scaled dot-product attention, QKV attention, multi-head attention]
difficulty: 2
status: complete
related: [arch-positional-encoding, flash-attention, arch-kv-cache, normalization-layers, vision-transformer]
---

# The Attention Mechanism

---

## Fundamental

Recurrent networks process sequences step-by-step: $h_t = f(h_{t-1}, x_t)$. Every piece of information from token $x_1$ must survive $n-1$ sequential state updates to reach $h_n$. Two fatal consequences: (1) a fixed-size vector bottleneck that compresses arbitrarily long context, and (2) $O(n)$ sequential computation that prevents GPU parallelism.

**Attention** abandons the bottleneck. Every output position directly reads from every input position in $O(1)$ hops. The sequence of updates is replaced by a single weighted retrieval over all positions simultaneously.

### The QKV Abstraction

Given token embeddings $X \in \mathbb{R}^{n \times d}$, project into three spaces with learned matrices:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V \quad \text{where } W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$$

- **Query $Q$**: what token $i$ is looking for
- **Key $K$**: what token $j$ advertises about itself
- **Value $V$**: what token $j$ contributes when retrieved

A high dot product $q_i \cdot k_j$ means "token $j$'s content is relevant to token $i$'s query." Compute scores for all pairs at once:

$$E = QK^T \in \mathbb{R}^{n \times n}, \quad \tilde{E} = E / \sqrt{d_k}$$

**Why divide by $\sqrt{d_k}$?** If $q$ and $k$ have independent unit-variance entries, $\text{Var}(q \cdot k) = d_k$. Without scaling, large $d_k$ (e.g., 512) pushes dot products into the softmax saturation region where gradients vanish.

### Scaled Dot-Product Attention

Convert scores to a probability distribution and compute a weighted sum of values:

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V}$$

Row $i$ of the output is a convex combination of all value vectors, with weights determined by how well query $i$ matches each key. The $n \times n$ weight matrix is never seen by the user — it exists transiently during computation.

---

## Intermediate

### Multi-Head Attention

A single head computes one similarity metric. Language and vision require multiple relationship types simultaneously (subject-verb agreement, coreference, spatial proximity). Run $h$ heads in parallel with smaller per-head dimension $d_k = d_{model}/h$:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$$

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O, \quad W_O \in \mathbb{R}^{h d_k \times d_{model}}$$

**Parameter count:** each head uses $3 d_{model} d_k = 3 d_{model}^2/h$ parameters. For $h$ heads: $3 d_{model}^2$ total, plus $d_{model}^2$ for $W_O$ — exactly the same as one large attention head. Multiple heads are computationally free.

### Causal Masking

For autoregressive language models, position $i$ must not attend to future positions $j > i$. Add $-\infty$ before softmax:

$$M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}, \qquad \text{Attention} = \text{softmax}\!\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$$

$e^{-\infty} = 0$ after softmax — future positions contribute nothing. This is implemented as a triangular mask over the $n \times n$ score matrix.

### The Full Transformer Block

Each block applies two sub-layers with residual connections and layer normalization (pre-norm variant):

$$x' = x + \text{MHA}(\text{LN}(x))$$
$$\text{out} = x' + \text{FFN}(\text{LN}(x'))$$

where $\text{FFN}(x) = W_2\,\text{GELU}(W_1 x)$ with $W_1 \in \mathbb{R}^{4d \times d}$ (4× expansion) and $W_2 \in \mathbb{R}^{d \times 4d}$.

**Division of labor:** attention = communication (each token reads from all others); FFN = computation (each token processed independently). This is why transformers store factual knowledge in FFN weights — attention routes information, FFN processes it.

### Cross-Attention

In encoder-decoder models (T5, original transformer for MT), decoder queries attend to encoder outputs:

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc})$$

Queries are computed from decoder state; keys and values come from the encoder. The decoder learns *which source tokens* are relevant for generating each target token. The same mechanism appears in diffusion model U-Nets for text conditioning.

---

## Advanced

### The $O(n^2)$ Bottleneck and Flash Attention

The $QK^T$ product takes $O(n^2 d)$ FLOPs and produces an $n \times n$ matrix requiring $O(n^2)$ memory. At $n = 4096$: 16M entries × 2 bytes = 32 MB **per head per layer**. This is the fundamental scaling limit of transformers.

Standard attention writes this matrix to HBM (the GPU's main memory, ~2 TB/s), reads it for softmax, writes back, then reads again for the $V$ multiplication — total $O(n^2)$ HBM reads/writes despite the computation being memory-bound, not FLOP-bound.

**Flash Attention** avoids materializing the $n \times n$ matrix entirely by tiling the computation into SRAM (on-chip, ~10× faster). It computes exact attention with $O(n)$ HBM memory footprint. The key technical primitive is **online softmax**: maintaining running maximum $m$ and normalizer $\ell$ so softmax can be updated block-by-block without seeing the full row.

### Attention as Bilinear Similarity Search

The attention operation can be interpreted as a **differentiable key-value store**. Given a query $q_i$, we retrieve a value $o_i = \sum_j \alpha_{ij} v_j$ weighted by soft matches. Unlike a hard database lookup, the gradient flows through $\alpha_{ij}$ back to both $q_i$ and $k_j$ — so the network learns what queries and keys to use.

This bilinear score $q_i^T k_j$ defines a particular notion of similarity. Multi-head attention runs $h$ independent bilinear forms simultaneously, each in a different subspace of the embedding — essentially learning $h$ different distance metrics over the token representation.

### Complexity of Attention Variants

| Variant | Time complexity | Memory | Notes |
|---------|:-:|:-:|---|
| Full self-attention | $O(n^2 d)$ | $O(n^2)$ | Standard |
| Causal (masked) | $O(n^2 d)$ | $O(n^2)$ | Lower triangle only |
| Flash Attention | $O(n^2 d)$ | $O(n)$ | Same math, IO-efficient |
| Sparse (Longformer) | $O(n w d)$ | $O(nw)$ | Window size $w$ |
| Linformer | $O(nd)$ | $O(n)$ | Approximation (loses exact attention) |

The key insight: Flash Attention achieves $O(n)$ **memory** while keeping **exact** computation — it reduces HBM bandwidth, not FLOPs. Approximate methods (Linformer, Performer) reduce FLOPs but sacrifice exactness.

### Interpretability: What Attention Heads Learn

Empirical studies (Voita et al., 2019; Clark et al., 2019) reveal consistent specialization across models:
- **Positional heads**: attend to immediately adjacent tokens
- **Syntactic heads**: track subject-verb, noun-adjective dependencies
- **Rare-word heads**: attend to BOS/EOS tokens (carrying sentence-global information)

Many attention heads are redundant: pruning 20% of heads in BERT produces <1% performance drop. This redundancy reflects the fact that $4d_{model}^2$ parameters per block gives the optimizer considerable latitude to find degenerate solutions.

---

*See also: [[arch-positional-encoding]] · [[flash-attention]] · [[arch-kv-cache]] · [[vision-transformer]] · [[normalization-layers]]*
