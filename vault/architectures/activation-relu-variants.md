---
title: ReLU and Variants
tags: [activation, training, architecture]
aliases: [ReLU, Leaky ReLU, PReLU, ELU, dying ReLU]
difficulty: 1
status: complete
related: [activation-gelu-swish, activation-sigmoid-tanh, backpropagation-advanced, optimizer-sgd-momentum]
---

# ReLU and Variants

---

## Fundamental

An **activation function** introduces non-linearity into a neural network — without it, stacking layers collapses to a single linear transformation. **ReLU (Rectified Linear Unit)** is the activation that made deep networks practical: it passes positive values unchanged and zeros out negatives, keeping gradients alive across many layers.

$$\text{ReLU}(x) = \max(0, x)$$

Gradient: $1$ if $x > 0$, else $0$. Computationally trivial.

**Why ReLU replaced [[activation-sigmoid-tanh|sigmoid]] in hidden layers:** sigmoid's maximum gradient is 0.25 — every layer multiplies the gradient by at most 0.25. For 10 layers, the gradient at layer 1 is at most $(0.25)^{10} \approx 10^{-6}$ of the final gradient. ReLU passes the full gradient through active neurons. It introduced the modern era of deep networks.

**The dying ReLU problem:** if a neuron's pre-activation is negative for *all* training examples, its gradient is permanently 0 — the neuron is dead. Causes: large learning rate pushing $w \cdot x + b < 0$ for all inputs, or very negative bias initialization. Magnitude: 5–40% of neurons in poorly-tuned networks.

---

## Intermediate

### Leaky ReLU and PReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases}$$

Typical $\alpha = 0.01$. Gradient is $\alpha$ (not 0) for negative inputs — neurons cannot die. **PReLU:** same formula but $\alpha$ is a learned parameter (one per channel). He et al. used PReLU in the original ResNet submission; it gave a small but consistent improvement.

### ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x \geq 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}$$

Typical $\alpha = 1.0$. Smooth at $x = 0$. The negative region drives mean activations toward zero — beneficial for [[normalization-layers|batch normalization]]. Cost: $e^x$ computation.

### Numerical Comparison

| $x$ | ReLU | Leaky ($\alpha=0.01$) | ELU ($\alpha=1$) |
|-----|------|-----------------------|------------------|
| $-2$ | $0$ | $-0.02$ | $e^{-2}-1 \approx -0.865$ |
| $-0.5$ | $0$ | $-0.005$ | $e^{-0.5}-1 \approx -0.393$ |
| $0$ | $0$ | $0$ | $0$ |
| $0.5$ | $0.5$ | $0.5$ | $0.5$ |
| $2$ | $2$ | $2$ | $2$ |

Gradient comparison at $x = -2$:

| ReLU | Leaky | ELU |
|------|-------|-----|
| $0$ | $0.01$ | $e^{-2} \approx 0.135$ |

ELU preserves far more gradient signal than Leaky ReLU in the negative region.

---

## Advanced

### SELU: Self-Normalizing Networks

$$\text{SELU}(x) = \lambda \cdot \text{ELU}(x, \alpha)$$

$\lambda \approx 1.0507$, $\alpha \approx 1.6733$ — not chosen empirically but derived analytically.

**The theorem (Klambauer et al., 2017):** under LeCun normal initialization, a feedforward network using SELU has a fixed point at $(\mu, \nu) = (0, 1)$ in the space of mean and variance of activations across layers. That is, if a layer's input has mean $\mu$ close to 0 and variance $\nu$ close to 1, the output also has mean $\approx 0$ and variance $\approx 1$. This is self-normalization — no BatchNorm needed.

**Why these constants:** the fixed-point conditions require:
1. $\alpha \lambda e^{\alpha \lambda} = 1$ (mean preservation)
2. $\lambda^2 [2\alpha \lambda e^{\alpha \lambda} - (\alpha \lambda)^2 e^{2\alpha \lambda}] = 1$ (variance preservation)

Solving these simultaneous constraints yields the specific $(\lambda, \alpha)$ values. The derivation involves the Banach fixed-point theorem on the mapping of mean/variance distributions across layers.

**Requirements that limit practical adoption:** LeCun normal initialization (not Kaiming), AlphaDropout (not standard dropout), fully connected layers only (not conv), no residual connections. Rarely used outside of niche settings.

### Full Comparison Table

| Activation | Dying neurons | Saturation | Smooth at 0 | Zero-mean output |
|------------|:---:|:---:|:---:|:---:|
| Sigmoid | — | Yes (both sides) | Yes | No (0.5 mean) |
| Tanh | — | Yes (both sides) | Yes | Yes |
| ReLU | Yes | No | No | No (~0.5·mean) |
| Leaky ReLU | No | No | No | Near zero |
| PReLU | No | No | No | Learned |
| ELU | No | No | Yes | Near zero |
| SELU | No | No | Yes | Exactly zero |
| [[activation-gelu-swish\|GELU / Swish]] | No | No | Yes | Near zero |

---

*See also: [[activation-gelu-swish]] · [[activation-sigmoid-tanh]] · [[backpropagation-advanced]] · [[normalization-layers]] · [[backpropagation]]*
