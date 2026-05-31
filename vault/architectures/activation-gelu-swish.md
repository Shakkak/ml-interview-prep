---
title: GELU and Swish
tags: [activation, training, transformers]
aliases: [GELU, Swish, SiLU, smooth activations]
difficulty: 2
status: complete
related: [activation-relu-variants, activation-sigmoid-tanh, normalization-layers]
---

# GELU and Swish

---

## Fundamental

ReLU is non-differentiable at 0 and has zero gradient for $x < 0$. For transformers and modern architectures, smoother activations improve gradient flow and empirical performance.

**GELU (Gaussian Error Linear Unit):**

$$\text{GELU}(x) = x \cdot \Phi(x) \qquad \text{where} \quad \Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Intuition:** the output is the input multiplied by the probability that a standard Gaussian sample is less than $x$. For large positive $x$, $\Phi(x) \approx 1$ (pass through); for large negative $x$, $\Phi(x) \approx 0$ (suppress). This is stochastic gating made deterministic. Used in BERT, GPT-2/3, ViT.

**Swish / SiLU:**

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

SiLU (Sigmoid Linear Unit) is the same function. Proposed by Google Brain (2017); found empirically superior to ReLU in many architectures. Used in EfficientNet, MobileNetV3, LLaMA.

---

## Intermediate

### Numerical Comparison

| $x$ | ReLU | GELU | Swish |
|-----|------|------|-------|
| $-3$ | $0$ | $\approx -0.004$ | $\approx -0.142$ |
| $-1$ | $0$ | $\approx -0.159$ | $\approx -0.269$ |
| $-0.5$ | $0$ | $\approx -0.154$ | $\approx -0.186$ |
| $0$ | $0$ | $0$ | $0$ |
| $0.5$ | $0.5$ | $\approx 0.345$ | $\approx 0.311$ |
| $1$ | $1$ | $\approx 0.841$ | $\approx 0.731$ |
| $3$ | $3$ | $\approx 2.996$ | $\approx 2.858$ |

Both GELU and Swish have a small negative output region: minimum near $x \approx -0.75$ for GELU, $x \approx -0.28$ for Swish. For large $|x|$: both converge to ReLU behavior.

**Swish derivative:**

$$\text{Swish}'(x) = \sigma(x)\left(1 + x(1 - \sigma(x))\right)$$

At $x=0$: $\text{Swish}'(0) = 0.5$. Always positive for $x > 0$. Slightly negative near $x \approx -1$.

**GELU fast approximation** (used in practice — avoids `erf` computation):

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

### When to Use Which

| Activation | Architecture | Rationale |
|-----------|--------------|-----------|
| ReLU | Classic CNNs, ResNet | Fast, well-understood |
| Leaky/PReLU | ResNets, GANs | Avoids dying neurons |
| GELU | BERT, GPT, ViT | Smooth, probabilistic gating |
| Swish/SiLU | EfficientNet, LLaMA | Empirically strong, slightly faster than GELU |
| SwiGLU | LLaMA, PaLM, Mistral | State-of-art for LLM FFN layers |

---

## Advanced

### GLU Variants: Gating in the FFN Layer

Gate Linear Unit (GLU) splits the input in two and multiplicatively gates one half:

$$\text{GLU}(x, W, V) = \sigma(xW) \otimes (xV)$$

**SwiGLU** replaces sigmoid with Swish (used in LLaMA, PaLM, Mistral):

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \otimes (xV)$$

In a transformer FFN block with SwiGLU:

$$\text{FFN}(x) = \text{SwiGLU}(xW_1, xW_2) \cdot W_3$$

**Critical parameter count detail:** the standard FFN uses hidden dimension $4d$ (one weight matrix $W \in \mathbb{R}^{d \times 4d}$). SwiGLU uses *two* weight matrices of size $d \times \frac{8d}{3}$ each. The $\frac{8d}{3}$ factor keeps the total parameter count matched to the standard $4d$ FFN:
$$2 \times d \times \frac{8d}{3} + \frac{8d}{3} \times d = \frac{16d^2}{3} + \frac{8d^2}{3} = 8d^2 = 2 \times 4d^2$$

This is why LLaMA intermediate sizes are multiples of $\frac{8}{3}$ of the model dimension (rounded up for hardware efficiency) — not an arbitrary choice.

**Why SwiGLU works:** the gating mechanism allows the FFN to selectively suppress certain hidden units based on the input, similar to attention but operating within each token independently. Noam Shazeer's original SwiGLU paper found it empirically superior to GELU and ReLU FFNs across all sizes, with no theoretical explanation — "the practical benefits are unexplained."

### Connection to Attention

Both attention (softmax-weighted sum) and SwiGLU (element-wise gating) are multiplicative interactions. The attention mechanism gates *values* based on *query-key compatibility*; SwiGLU gates *hidden features* based on the same input. Modern architectures stack both: attention for inter-token mixing, SwiGLU-FFN for per-token nonlinear transformation.

---

*See also: [[activation-relu-variants]] · [[activation-sigmoid-tanh]] · [[attention-mechanism]]*
