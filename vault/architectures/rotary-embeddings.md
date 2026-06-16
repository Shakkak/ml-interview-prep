---
title: Rotary Position Embeddings (RoPE)
tags: [rope, rotary-embeddings, positional-encoding, relative-position, llm]
aliases: [RoPE, rotary embeddings, rotary position encoding, RoPE scaling, NTK-aware RoPE]
difficulty: 2
status: complete
depends_on: [arch-positional-encoding, attention-mechanism, linear-algebra-fundamentals]
related: [arch-positional-encoding, attention-mechanism, arch-kv-cache, autoregressive-models, flash-attention]
---

# Rotary Position Embeddings (RoPE)

---

## Fundamental

### Motivation

Standard absolute positional encodings (sinusoidal or learned, see [[arch-positional-encoding]]) add a position vector to the token embedding. The attention score $q_m^\top k_n$ then implicitly captures the positions $m$ and $n$ — but not their relative difference $m - n$ cleanly.

**Relative position** is what matters for language: "the cat that sat on the mat" — the relationship between "cat" and "sat" is relative (2 positions), not absolute (position 2 and 4).

**RoPE (Su et al., 2021)** encodes position by rotating the query and key vectors before the dot product, such that $q_m^\top k_n$ depends only on the content and the **relative displacement** $m - n$.

### The Rotation Idea

For 2D vectors, multiply each pair of dimensions by a rotation matrix at angle $m\theta$:

$$\mathbf{q}_m = R_m \mathbf{q}, \quad R_m = \begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix}$$

where $\mathbf{q}$ = original 2D query vector, $m$ = absolute position of the token in the sequence, $\theta$ = rotation frequency (a fixed constant), and $R_m$ = a 2D rotation matrix that rotates $\mathbf{q}$ by angle $m\theta$. The key identity: $R_m^\top R_n = R_{n-m}$ (rotating forward by $n$ then backward by $m$ = net rotation of $n-m$).

Then: $\mathbf{q}_m^\top \mathbf{k}_n = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m} \mathbf{k}$

The inner product depends only on $n - m$ — relative position — not on absolute positions $m$ and $n$.

---

## Intermediate

### Full RoPE for $d$-Dimensional Vectors

For $d$-dimensional query/key vectors, pair up dimensions and apply 2D rotations at different frequencies:

$$\text{RoPE}(\mathbf{x}, m)_{2i}, \text{RoPE}(\mathbf{x}, m)_{2i+1} = \begin{pmatrix}x_{2i}\cos m\theta_i - x_{2i+1}\sin m\theta_i \\ x_{2i}\sin m\theta_i + x_{2i+1}\cos m\theta_i\end{pmatrix}$$

where $\mathbf{x}$ = original query or key vector, $m$ = token position, $i$ = dimension-pair index (ranges 0 to $d/2 - 1$), $x_{2i}, x_{2i+1}$ = the two components of dimension pair $i$, and $\theta_i = 10000^{-2i/d}$ = rotation frequency for pair $i$ (same base as sinusoidal PE — small $i$ rotates fast, large $i$ rotates slowly).

**Properties:**
- Each dimension pair rotates at a different frequency; low-index pairs rotate faster, high-index pairs rotate slower
- Outer product form: $\text{RoPE}(\mathbf{x}, m) = \text{Re}(\mathbf{x} \cdot e^{im\Theta})$ where $\Theta = (\theta_0, \theta_1, \ldots)$
- Efficient implementation: only requires element-wise multiplications (no full rotation matrix)

### Advantages Over Absolute PE

| Property | Sinusoidal PE | Learned PE | RoPE |
|----------|--------------|-----------|------|
| Relative position | Implicit | Implicit | Explicit |
| Context extension | Poor | Fixed vocab | Extrapolatable |
| Causally consistent | Yes | Yes | Yes |
| Parameters added | None | $L \times d$ | None |
| Long-context performance | Degrades | Degrades | Better |

RoPE is used in LLaMA, Mistral, Qwen, Gemma, and most modern open-source LLMs.

---

## Advanced

### Context Length Extension: NTK-Aware Scaling

RoPE was originally trained at context length $L = 2048$ or $4096$. Extending to longer contexts (16K, 128K) is non-trivial because:
- Low-frequency dimensions ($\theta_i$ small) complete less than one full rotation over $L$ tokens — they see all positions distinctly
- High-frequency dimensions ($\theta_i$ large) complete many rotations — they can't distinguish $m$ from $m + 2\pi/\theta_i$

**Linear scaling / RoPE scaling:** scale all positions by $L'/L$ (slow down rotations). Simple but degrades quality because high-frequency dimensions become too slow.

**NTK-aware RoPE (bloc97, 2023):** increase the base $b = 10000$ to $b' = b \cdot (L'/L)^{d/(d-2)}$. This shifts the rotation spectrum — high-frequency dimensions slow down proportionally more. Allows zero-shot extrapolation to $2\times$–$4\times$ training context.

**YaRN (Peng et al., 2023):** combines NTK scaling with an attention temperature correction and fine-tuning on a small set of long-context examples. Enables 128K+ context from a 4K-trained model.

### RoPE in Multi-Head Attention

RoPE is applied per-head to the Q and K projections only — not to V. This preserves the relative-position inner products while leaving values (which aggregate content) untouched.

In grouped query attention (see [[grouped-query-attention]]), RoPE is applied to all Q heads and to the fewer K heads independently.

## Links

- [[arch-positional-encoding]] — RoPE supersedes absolute sinusoidal and learned position embeddings by encoding positions as rotations in QK space
- [[attention-mechanism]] — RoPE is applied to queries and keys before the dot product; the rotation means $q_m^\top k_n$ depends only on the relative offset $m-n$
- [[linear-algebra-fundamentals]] — rotation matrices in 2D pairs of dimensions encode position; the block-diagonal structure keeps computation $O(d)$
- [[arch-kv-cache]] — RoPE rotations must be applied at inference time when extending context beyond training length; NTK-aware scaling adjusts the base frequency for longer contexts
- [[autoregressive-models]] — autoregressive transformers (LLaMA, Mistral, GPT-4) use RoPE for its superior length-generalization properties
- [[flash-attention]] — Flash Attention implementations must account for RoPE being applied inside the kernel
