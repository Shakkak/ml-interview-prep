---
title: Weight Initialization
tags: [training, initialization, variance, deep-learning]
aliases: [Xavier initialization, He initialization, Glorot, Kaiming, weight init]
difficulty: 2
status: complete
related: [backpropagation, backpropagation-advanced, normalization-layers, activation-relu-variants, activation-sigmoid-tanh, activation-gelu-swish, optimizer-adam, regularization-weight-decay]
---

# Weight Initialization

---

## Fundamental

### Why Initialization Matters

All parameters are zero before training. Naively setting all weights to zero gives zero gradients by symmetry — all neurons in a layer receive the same gradient update and learn the same function forever, wasting all capacity. **Symmetry must be broken** at initialization.

**Two catastrophic failure modes without proper init:**

1. **Vanishing signals:** if weights are too small, activations shrink with each layer — at depth 50, a signal of 1 becomes $0.9^{50} \approx 0.005$. Backpropagating gradients shrink equivalently.

2. **Exploding signals:** if weights are too large, activations grow with each layer — at depth 50, a signal of 1 becomes $1.1^{50} \approx 117$. Gradients explode catastrophically.

**The target:** every layer's output should have approximately the same variance as its input — maintaining signal scale throughout the network.

### Variance Preservation Analysis

For a single linear layer $y = Wx$ with no activation, where $W \in \mathbb{R}^{n_{out} \times n_{in}}$:

$$\text{Var}(y_i) = \text{Var}\!\left(\sum_j W_{ij} x_j\right) = n_{in} \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_j)$$

assuming $W_{ij}$ and $x_j$ are i.i.d. and zero-mean, $\text{Cov}(W_{ij}, x_j) = 0$.

For variance preservation ($\text{Var}(y) = \text{Var}(x)$), we need:

$$\text{Var}(W_{ij}) = \frac{1}{n_{in}}$$

This is the **LeCun initialization** — optimal for linear activations. In practice, initialize $W_{ij} \sim \mathcal{N}(0, 1/n_{in})$ or $\text{Uniform}(-\sqrt{3/n_{in}}, \sqrt{3/n_{in}})$.

---

## Intermediate

### Xavier/Glorot Initialization (for tanh and sigmoid)

For tanh or sigmoid activations (smooth, centered near zero), the gradient of the activation is approximately 1 near zero but shrinks toward the tails. Glorot & Bengio (2010) derived an initialization that preserves variance for both the forward pass AND the backward pass simultaneously:

**Forward variance preservation:** $\text{Var}(W) = 1/n_{in}$
**Backward variance preservation:** $\text{Var}(W) = 1/n_{out}$

These two requirements are incompatible when $n_{in} \neq n_{out}$. The Xavier compromise uses their harmonic mean:

$$\text{Var}(W_{ij}) = \frac{2}{n_{in} + n_{out}}$$

**In practice:**
- Normal: $W \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$
- Uniform: $W \sim \text{Uniform}\!\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$

For a balanced layer ($n_{in} = n_{out} = 256$): $\text{std} = \sqrt{2/512} \approx 0.063$.

### He/Kaiming Initialization (for ReLU)

ReLU zeroes half its inputs — this halves the variance at each layer. The forward variance equation becomes:

$$\text{Var}(y_i) = n_{in} \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_j) \cdot \underbrace{\mathbb{E}[\mathbf{1}[z_j > 0]]}_{\approx 1/2}$$

To compensate for the factor of $1/2$:

$$\text{Var}(W_{ij}) = \frac{2}{n_{in}}$$

This is **He initialization** (He et al., 2015), also called Kaiming init:

$$W \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**Intuition:** ReLU kills exactly half the neurons on average. To preserve variance, we need twice as large weights to compensate. $\sqrt{2/n_{in}}$ vs $\sqrt{1/n_{in}}$ — a factor of $\sqrt{2}$ larger than LeCun.

**Fan mode choices:**
- `fan_in` (default): $\text{std} = \sqrt{2/n_{in}}$ — preserves forward-pass variance
- `fan_out`: $\text{std} = \sqrt{2/n_{out}}$ — preserves backward-pass gradient variance
- For most cases, `fan_in` is preferred (He et al. recommend it)

### Practical Initialization Table

| Activation | Init | Formula | Notes |
|-----------|------|---------|-------|
| Linear | LeCun | $\mathcal{N}(0, 1/n_{in})$ | Exactly preserves forward variance |
| tanh / sigmoid | Xavier | $\mathcal{N}(0, 2/(n_{in}+n_{out}))$ | Compromise forward+backward |
| ReLU | He (fan_in) | $\mathcal{N}(0, 2/n_{in})$ | Compensates 50% zeroing |
| LeakyReLU($a$) | He-Leaky | $\mathcal{N}(0, 2/((1+a^2)n_{in}))$ | Accounts for nonzero slope |
| SELU | LeCun | $\mathcal{N}(0, 1/n_{in})$ | SELU is designed to be self-normalizing |

### Initialization for Transformers

Transformers have residual connections: $x_{l+1} = x_l + F_l(x_l)$. With $L$ layers, the residual stream is the sum of all layer contributions. If each $F_l$ contributes variance $\sigma^2_F$, the total variance grows as $L\sigma^2_F + \sigma^2_x$. For deep transformers ($L = 100+$), this makes the residual stream grow as $O(\sqrt{L})$ — requiring careful initialization.

**Scaled initialization (T-Fixup, Zhang et al., 2019):** scale down weight matrices in residual branches by $L^{-1/4}$ (for attention and FFN combined), so the accumulated variance stays $O(1)$ regardless of depth. This enables training deep transformers without layer normalization.

**Standard practice (GPT-2, LLaMA):** normal He/Xavier init for attention/FFN weights, but with output projections (the last linear layer in each sublayer) initialized with std $= 0.02 / \sqrt{2L}$. The $1/\sqrt{2L}$ factor ensures that at initialization, the $L$ residual contributions don't overwhelm the residual stream.

---

## Advanced

### Orthogonal Initialization

For deep linear networks, orthogonal weight matrices perfectly preserve both the forward signal and backward gradient — isometry in both directions. For a square weight matrix $W \in \mathbb{R}^{n \times n}$, orthogonal means $W^\top W = WW^\top = I$, so $\|Wx\| = \|x\|$ exactly.

**Construction:** generate $W \sim \mathcal{N}(0, 1)^{n \times n}$, compute QR decomposition $W = QR$, use $Q$ as the initialized weight.

**For non-square layers:** use the left singular vectors from SVD. For $W \in \mathbb{R}^{m \times n}$ with $m < n$: $W = U \Sigma V^\top$, set $W \leftarrow U V^\top$ (a semi-orthogonal matrix).

**Empirical performance:** orthogonal initialization significantly outperforms random Gaussian init for very deep networks (>50 layers) without normalization, and remains competitive with He init for networks with BatchNorm. It's especially effective for RNNs, where the recurrent weight matrix needs near-orthogonal initialization to prevent vanishing/exploding gradients over long sequences.

### Megatron-LM / GPT-3 Initialization at Scale

Training language models with billions of parameters at $\sim 3 \times 10^{23}$ FLOPs revealed that standard initialization strategies that work at small scale break at large scale. The key insight from Megatron-LM (Shoeybi et al., 2019):

For a model with $L$ transformer layers, the residual stream accumulates $2L$ sublayer contributions (attention + FFN per layer). The output projection weights in each sublayer should be initialized as:

$$\text{std} = \frac{0.02}{\sqrt{2L}}$$

This ensures the sum of $2L$ independent contributions has std $\approx 0.02$, matching the embedding layer's std and maintaining consistent activation scale throughout the network at initialization.

**Maximal Update Parameterization ($\mu$P, Yang et al., 2022):** addresses initialization and learning rate jointly. $\mu$P derives initialization and LR scales as a function of width $d$ and depth $L$ such that:
- Activations have $O(1)$ magnitude at any width
- Gradients have $O(1)$ magnitude at any width
- Weight updates are $O(1)$ in magnitude regardless of width

The key result: in the $\mu$P regime, optimal hyperparameters transfer from small to large models — you can tune LR on a small model and it works for the billion-parameter version. Used in GPT-4 training and many large-scale LLMs.

### Why BatchNorm Changes Initialization Requirements

With BatchNorm, the actual activation scale is reset to $(0, 1)$ after each layer — making the initialization scale mostly irrelevant for the activation magnitudes. However, initialization still matters for the gradient scale at step 0 (before BN statistics accumulate) and for the learned $\gamma$ parameter.

**Standard practice with BatchNorm:** use He initialization for conv/linear weights, initialize $\gamma = 1, \beta = 0$ for BN. For ResNets, the last BN in each residual block is initialized with $\gamma = 0$ — this means the residual block outputs zero at initialization, making the network behave like a shallower network at start and gradually deepening as the $\gamma$ values grow (Goyal et al., 2017). This **zero-gamma trick** significantly stabilizes training of very deep ResNets and is standard in high-performance training recipes.

---

*See also: [[backpropagation-advanced]] · [[normalization-layers]] · [[activation-relu-variants]] · [[activation-gelu-swish]] · [[regularization-weight-decay]]*
