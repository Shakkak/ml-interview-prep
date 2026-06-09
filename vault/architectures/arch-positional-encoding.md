---
title: Positional Encoding
tags: [architecture, transformers, attention]
aliases: [positional encoding, sinusoidal encoding, RoPE, ALiBi, learned position embedding]
difficulty: 2
status: complete
related: [attention-mechanism, arch-kv-cache]
---

# Positional Encoding

---

## Fundamental

[[attention-mechanism|Self-attention]] is permutation-equivariant: shuffle the input tokens and the outputs shuffle identically. Attention has no built-in notion of order. "The dog bit the man" ≠ "The man bit the dog," but vanilla attention cannot tell them apart.

**Solution:** inject position information into token embeddings before attention.

### Sinusoidal Encoding (Vaswani et al., 2017)

For position $t$ and dimension index $i$ (of $d_{\text{model}}$ total):

$$PE(t, 2i) = \sin\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right), \qquad PE(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)$$

Add to the token embedding: $x_t \leftarrow x_t + PE(t, :)$

**Design rationale:** dimensions cycle at different frequencies — low dimensions (small $i$) cycle fast (nearby positions differ), high dimensions cycle slowly (useful for global position). This is the [[fourier-transform|Fourier]] principle applied to discrete positions: each dimension is a sinusoidal basis function at a different frequency. The base $10000$ ensures the last dimension has period $\approx 62{,}832$ tokens — larger than typical sequences.

**Worked example** ($d = 4$, $t = 1$):

| dim | formula | value |
|-----|---------|-------|
| 0 | $\sin(1/10000^0) = \sin(1)$ | $0.841$ |
| 1 | $\cos(1/10000^0) = \cos(1)$ | $0.540$ |
| 2 | $\sin(1/100) = \sin(0.01)$ | $0.010$ |
| 3 | $\cos(1/100) = \cos(0.01)$ | $1.000$ |

At $t = 100$: dims 0–1 have completed ~16 full cycles (fine-grained); dims 2–3 are barely into their first cycle (global position).

---

## Intermediate

### Learned Positional Embeddings

Train a lookup table $E_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d}$ where row $t$ = position embedding for position $t$.

Used in [[bert-mlm|BERT]], GPT-2, [[vision-transformer|ViT]]. Slightly better on fixed-length tasks. Cannot generalize beyond $T_{\max}$ without fine-tuning — attempting longer sequences produces out-of-distribution embeddings.

### ALiBi: Attention with Linear Biases

Add a position-dependent penalty to attention logits before softmax:

$$\text{Attention}(Q, K) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} - m \cdot |i - j|\right)$$

$m$ is a head-specific slope (e.g., $m_h = 2^{-8h/H}$, geometric sequence across heads). $|i-j|$ is the token distance.

**Intuition:** penalize attending to distant tokens. No parameters to learn.

**Extrapolation:** since the bias depends on *distance*, not *absolute position*, ALiBi naturally extends to longer sequences at inference. A model trained on 2048 tokens can process 8192 tokens — quality degrades gracefully with distance rather than catastrophically.

### Comparison Table

| Method | Relative position | Extrapolation | Parameters | Used In |
|--------|:---:|:---:|:---:|---------|
| Sinusoidal | Via linear transform | Limited | 0 | Original Transformer |
| Learned | No (absolute) | Poor | $T_{\max} \times d$ | BERT, GPT-2, ViT |
| RoPE | Yes (native) | Good | 0 | LLaMA, Mistral, GPT-NeoX |
| ALiBi | Yes (via bias) | Excellent | 0 | BLOOM, MPT |

---

## Advanced

### RoPE: Rotary Position Embedding

Instead of adding positions to embeddings, **rotate** Q and K vectors by a position-dependent angle before the dot product:

$$\tilde{q}_t = R_{\Theta,t}\, q_t, \qquad \tilde{k}_s = R_{\Theta,s}\, k_s$$

For each pair of dimensions $(2i, 2i+1)$, the rotation matrix is:

$$R_{\Theta,t}^{(i)} = \begin{bmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{bmatrix}, \qquad \theta_i = 10000^{-2i/d}$$

**Key property:** the dot product depends only on the relative position $t - s$:

$$\tilde{q}_t^\top \tilde{k}_s = q_t^\top R_{\Theta,t-s}\, k_s = f(q_t, k_s, t-s)$$

> [!tip] Why $R_{\Theta,t}^\top R_{\Theta,s} = R_{\Theta,s-t}$ ([[linear-algebra-fundamentals]])
> A 2D rotation by angle $\alpha$ has transpose $R_\alpha^\top = R_{-\alpha}$ (rotating backwards is the inverse rotation).
> Rotation matrices are a group under multiplication: $R_\alpha R_\beta = R_{\alpha+\beta}$.
> Therefore: $R_t^\top R_s = R_{-t} R_s = R_{s-t}$.
> RoPE applies this independently to each pair of dimensions $(2i, 2i+1)$ with angle $t\theta_i$.
> The inner product becomes $\tilde{q}_t^\top \tilde{k}_s = q_t^\top R_t^\top R_s k_s = q_t^\top R_{s-t} k_s$ — only the distance $s-t$ survives.

The inner product extracts only the relative angle.

**Why this is better than additive PE:** absolute position embeddings encode $t$ and $s$ separately; the model must learn to compute $t - s$ implicitly. RoPE encodes relative distance directly into the attention score — the model sees relative positions natively.

### Context Length Extension with RoPE

Extending beyond training length: if trained at context $L$, directly applying RoPE to $L' > L$ tokens extrapolates poorly (the model has not seen those frequencies).

**NTK-aware scaling (LocalLLaMA, 2023):** scale the base from $10000$ to $10000 \cdot (L'/L)^{d/(d-2)}$. This ensures frequencies at the new length $L'$ behave like frequencies at the training length $L$ under the [[neural-tangent-kernel|Neural Tangent Kernel]] approximation.

**YaRN (Peng et al., 2023):** additionally applies attention temperature scaling (divide by $\sqrt{n}$ where $n$ is the scale factor) to counteract the softmax getting sharper at longer distances. Achieves near-perfect extrapolation to 128K context from 4K training.

---

*See also: [[attention-mechanism]] · [[arch-kv-cache]] · [[bert-mlm]] · [[vision-transformer]] · [[fourier-transform]] · [[neural-tangent-kernel]]*
