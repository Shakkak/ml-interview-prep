---
title: Softmax
tags: [activation, probability, classification]
aliases: [softmax, softmax function, normalized exponential]
difficulty: 1
status: complete
depends_on: [linear-algebra-fundamentals, loss-cross-entropy]
related: [activation-sigmoid-tanh, loss-cross-entropy, attention-mechanism, backpropagation-advanced]
---

# Softmax

---

## Fundamental

**Softmax** converts a vector of raw scores (logits) — one per class — into a probability distribution over all classes. Every output is positive and they all sum to 1, making softmax the standard final layer for multi-class classification. The exponential function ensures larger logits get disproportionately higher probability, creating sharp decisions when scores differ by even a few units.

For a vector of logits $z \in \mathbb{R}^K$:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Output is a probability distribution: all values in $(0, 1)$, sum to 1.

**Worked example.** Logits $z = [2.0, 1.0, 0.1]$:
$$e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105, \quad \text{sum} = 11.212$$
$$p = [0.659, 0.242, 0.099]$$

Note: a logit difference of 1.0 corresponds to a probability ratio of $e^{1.0} \approx 2.72$ (not 2×). Logit differences are additive; probability ratios are multiplicative.

**Temperature scaling:** dividing logits by $\tau$ before softmax controls sharpness:

| $\tau$ | $p_1$ | $p_2$ | $p_3$ | Effect |
|--------|--------|--------|--------|--------|
| 0.1 | 0.999 | 0.001 | ≈0 | Sharp, near one-hot |
| 1.0 | 0.659 | 0.242 | 0.099 | Standard |
| 2.0 | 0.519 | 0.313 | 0.168 | Softer |
| 10.0 | 0.356 | 0.341 | 0.337 | Near-uniform |

$\tau \to 0$: argmax (hard selection). $\tau \to \infty$: uniform distribution.

---

## Intermediate

### Softmax Jacobian

The gradient of softmax is a matrix. For $p_i = \text{softmax}(z)_i$:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

**Diagonal** ($i = j$): $\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$ — identical in form to the [[activation-sigmoid-tanh|sigmoid]] derivative.

**Off-diagonal** ($i \neq j$): $\frac{\partial p_i}{\partial z_j} = -p_i p_j$ — negative, because increasing $z_j$ pulls probability mass toward $p_j$ from all others.

In practice, softmax + [[loss-cross-entropy|cross-entropy]] simplifies: the gradient of $L = -\sum_k y_k \log p_k$ w.r.t. logits is simply $p_k - y_k$ (predicted minus true probability). The Jacobian is never computed explicitly.

### Numerical Stability

Computing $e^{z_k}$ overflows for large logits (e.g., $e^{1000} = \infty$ in float32). Standard fix: subtract the max before exponentiation:

$$\text{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

This doesn't change the output (max cancels in numerator and denominator) but all exponents are $\leq 0$, preventing overflow.

With $z = [2.0, 1.0, 0.1]$, subtract 2.0: $[0, -1.0, -1.9]$
$$e^0=1,\ e^{-1}=0.368,\ e^{-1.9}=0.150,\ \text{sum}=1.518 \quad \to p=[0.659, 0.242, 0.099] \checkmark$$

### Where Softmax Appears

| Location | Purpose |
|----------|---------|
| Final layer, multi-class | Convert logits to probabilities |
| Attention scores | Normalize attention weights over positions |
| [[knowledge-distillation\|Knowledge distillation]] | Soft targets (high $\tau$) |
| Gumbel-Softmax | Differentiable discrete sampling |
| [[loss-nt-xent\|NT-Xent]] contrastive loss | Normalize similarity scores over negatives |

---

## Advanced

### Softmax as a Boltzmann Distribution

Softmax is identical to the Boltzmann distribution from statistical physics:

$$p_k = \frac{e^{-E_k / (k_B T)}}{\sum_j e^{-E_j / (k_B T)}}$$

where $E_k$ is the energy of state $k$, $T$ is temperature, and $k_B$ is Boltzmann's constant (a physical constant $\approx 1.38 \times 10^{-23}$ J/K). Note: $k_B$ here is the physical constant, not a class index — the class index $k$ used elsewhere in this file is a different variable. Setting $E_k = -z_k$ and $k_B T = \tau$: exactly the temperature-scaled softmax.

This connection is not cosmetic. It implies:
- Logits are **negative energies** — the model assigns lower energy to more probable classes
- Temperature $\tau$ has a precise physical meaning: low temperature = peaked around the ground state; high temperature = thermal disorder
- [[loss-kl-divergence|KL divergence]] from a softmax distribution to a uniform distribution equals the **free energy** of the system

### Temperature Scaling for Post-Hoc Calibration

Neural networks trained with standard cross-entropy are often **[[model-calibration|overconfident]]**: a model that predicts 95% confidence is right only 80% of the time. The cause is that maximizing cross-entropy drives logits toward large magnitudes, saturating softmax near 1.0.

**Post-hoc temperature scaling** (Guo et al., 2017): after training, find a single scalar $T > 1$ on a held-out validation set that minimizes cross-entropy of $\text{softmax}(z/T)$ (model parameters frozen). The output probabilities shrink toward uniform, reducing overconfidence.

- $T = 1$: original predictions (overconfident)
- $T > 1$: softer predictions (better calibrated)
- $T < 1$: sharper predictions (even more overconfident — useful for greedy decoding of LLMs)

Guo et al. showed this single-parameter recalibration matches or outperforms more complex methods (isotonic regression, Platt scaling with multiple parameters) on modern deep networks. Key insight: the **rank order** of predictions is unchanged — only confidence magnitudes shift. The model's accuracy is unchanged; only its confidence estimates improve.

**Connection to training:** label smoothing (see [[regularization-label-smoothing]]) achieves a similar effect *during* training by capping the optimal logit gap. Temperature scaling applies the same correction *after* training without touching model weights. Models trained with label smoothing often need less post-hoc temperature scaling.

### Gumbel-Softmax: Differentiable Discrete Sampling

Sampling a discrete category from a softmax is non-differentiable. The Gumbel-Softmax trick (Jang et al., 2017) adds Gumbel noise to logits before softmax to simulate sampling:

$$y_k = \frac{\exp((z_k + g_k)/\tau)}{\sum_j \exp((z_j + g_j)/\tau)}, \qquad g_k \sim \text{Gumbel}(0,1) = -\log(-\log U_k), \quad U_k \sim \text{Uniform}(0,1)$$

**Why Gumbel noise:** the argmax of $(z_k + g_k)$ is distributed as a sample from $\text{Categorical}(\text{softmax}(z))$ — this is the Gumbel-max trick. Adding temperature makes the operation smooth and differentiable; gradients flow through the sampling. Used in [[variational-autoencoders|VAEs]] with discrete latents, DALL-E, and neural architecture search.

As $\tau \to 0$: Gumbel-Softmax converges to a one-hot vector (hard, non-differentiable argmax). Practical training uses a "straight-through" estimator: forward pass uses hard argmax, backward pass uses Gumbel-Softmax gradients.

### Attention Scaling and Softmax Saturation

In scaled dot-product attention, dividing by $\sqrt{d_k}$ before softmax is necessary because without it, the dot products grow as $O(d_k)$ in variance. Large inputs drive softmax into its saturation regime where gradients are near zero. The factor $\sqrt{d_k}$ normalizes the variance to 1 regardless of dimension — preventing the "peaky softmax" problem where the model attends to exactly one position and gradients vanish.

---

## Links

- [[linear-algebra-fundamentals]] — softmax is a mapping from $\mathbb{R}^K$ to the probability simplex $\Delta^{K-1}$; it is not linear but is differentiable
- [[loss-cross-entropy]] — softmax + cross-entropy is the canonical multi-class output; the combined gradient simplifies to $\hat{p} - y$ (see backpropagation-advanced)
- [[attention-mechanism]] — attention uses softmax to normalize dot products into a probability distribution over positions
- [[backpropagation-advanced]] — the Jacobian of softmax is $\text{diag}(p) - pp^\top$; composed with cross-entropy, this simplifies to $p - y$
- [[loss-nt-xent]] — NT-Xent uses softmax over the similarity scores of all negative pairs; temperature scales the softmax sharpness
- [[knowledge-distillation]] — soft targets are softmax outputs with high temperature; distillation trains the student on these smooth distributions
- [[loss-kl-divergence]] — KL divergence between the softmax output and the true distribution is the categorical cross-entropy; minimizing CE = minimizing KL
- [[model-calibration]] — overconfident softmax outputs (probabilities too close to 0 or 1) signal poor calibration; temperature scaling is the standard fix
