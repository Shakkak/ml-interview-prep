---
title: Dropout
tags: [regularization, training, overfitting]
aliases: [dropout, inverted dropout, DropBlock]
difficulty: 1
status: complete
related: [regularization-weight-decay, normalization-layers, bias-variance-double-descent]
---

# Dropout

---

## Fundamental

During each forward pass of training, randomly set each activation to 0 with probability $p$ (typically $p=0.5$ for FC layers, $p=0.1$–$0.2$ for conv layers). At inference, use all neurons.

**Inverted dropout** (standard implementation): scale by $1/(1-p)$ during training so inference requires no scaling:
```python
mask = (torch.rand(x.shape) > p).float()
x = x * mask / (1 - p)   # training only
```

**Example:** $h = [1.0, 2.0, 3.0, 4.0]$, $p=0.5$, mask $=[1,0,1,0]$:
$$h_\text{drop} = [1.0, 0, 3.0, 0] \xrightarrow{\times 2} [2.0, 0, 6.0, 0]$$

Expected value check: $\mathbb{E}[h_{\text{scaled},i}] = (1-p) \cdot h_i/(1-p) = h_i$ — the expected activation is unchanged.

**Three views of why dropout works:**
1. **Ensemble view:** training with dropout ≈ training $2^N$ different sub-networks (exponential in $N$ neurons); inference averages their predictions approximately.
2. **Prevents co-adaptation:** neurons cannot rely on specific other neurons always being present; each must learn independently useful features.
3. **Noise injection:** Bernoulli noise on activations acts as regularization, similar to data augmentation.

**Where to apply:**

| Location | Typical $p$ |
|----------|-------------|
| FC layers (MLP) | 0.5 |
| CNN feature maps | 0.1–0.3 |
| Transformer attention | 0.1 |
| Transformer FFN | 0.1 |

---

## Intermediate

**DropBlock** (Ghiasi et al., 2018) — structured dropout for CNNs: standard dropout on feature maps drops individual pixels. Adjacent pixels are highly correlated, so zeroing one barely hurts — information leaks through neighbors. DropBlock drops contiguous $k \times k$ square regions:

```
Standard (4×4 map):    DropBlock (3×3 block):
1 0 1 1                0 0 0 1
1 1 0 1                0 0 0 1
0 1 1 0                0 0 0 1
1 0 1 1                1 1 1 1
```

More effective regularization, especially at high-resolution feature maps. Keep rate parameter $\gamma$ (fraction of blocks to drop) is typically set inversely proportional to block size to maintain the same expected sparsity.

**Dropout interaction with BatchNorm:** dropout changes the variance of activations, which disrupts BatchNorm's running statistics. If using [[normalization-layers|BatchNorm]], reduce or remove dropout. If both are needed, place dropout *after* the activation and *before* BatchNorm in the layer order.

**MC Dropout for uncertainty estimation** (Gal & Ghahramani, 2016): keep dropout active at inference and run $T$ forward passes. The variance of predictions estimates **epistemic uncertainty** (model uncertainty from limited data):
$$\text{uncertainty} \approx \text{Var}_{t=1}^T[f_t(x)]$$

Used in medical imaging, autonomous driving — domains where knowing *when the model doesn't know* is critical. MC Dropout with $T=50$ passes approximates a Bayesian neural network's posterior predictive distribution at low cost.

**Spatial dropout / channel dropout:** a variant that drops entire channels in a feature map rather than individual activations. Useful for regularizing fully convolutional networks where channels encode semantic features — dropping a channel forces the network to learn redundant feature representations across channels.

**Optimal dropout rate depends on model size:** larger models benefit from higher dropout rates; smaller models may be hurt by aggressive dropout. Empirically, $p=0.5$ for large FC layers, $p=0.1$ for small ones, and $p=0.1$–$0.2$ for transformers. For vision transformers, dropout on attention weights (attn_drop) and FFN activations (ffn_drop) of 0.1 is standard.

---

## Advanced

**Dropout as a Bayesian approximation:** Gal & Ghahramani (2016) showed that training with dropout followed by MC Dropout inference is equivalent to approximate Bayesian inference in a deep Gaussian process. The network posterior $p(\theta|\text{data})$ is approximated by a mixture of two Gaussians per weight (zero mean, non-zero variance) — the Bernoulli mask samples from this mixture. This theoretical connection explains why MC Dropout uncertainty estimates are reasonably calibrated without any explicit Bayesian inference, though the approximation is coarse and underestimates uncertainty in many practical settings.

**Alpha-dropout for SELU networks:** SELU (Self-Normalizing Neural Network, Klambauer et al., 2017) activation combined with specific initialization guarantees that activations maintain zero mean and unit variance throughout the forward pass — a form of self-normalization without BatchNorm. Standard dropout breaks this property (the expected activation after dropout is $(1-p)$ times the original, not zero-mean unit-variance). Alpha-dropout preserves the mean and variance by transforming dropped activations to a specific value $\alpha'$ rather than 0, maintaining SELU's self-normalizing property. Used exclusively with SELU activations.

**Curriculum dropout (Morerio et al., 2017):** the optimal dropout rate changes during training. Early in training, low dropout allows rapid feature learning. Late in training, high dropout enforces stronger regularization. Curriculum dropout starts with $p=0$ and gradually increases to the target rate following a schedule (e.g., linear or sigmoid). Reported improvements of 0.5–1% on CIFAR-10/100 and better convergence stability. The intuition: early aggressive dropout wastes capacity before the network has learned basic features.

**Dropout vs weight decay on overparameterized networks:** on modern large models, [[regularization-weight-decay|weight decay]] alone is often sufficient regularization — dropout adds limited benefit and can slow convergence. The shift away from dropout in transformers (LLMs use $p=0.1$ or none, whereas ResNets used $p=0.5$) reflects the move to overparameterized regimes where implicit regularization from SGD noise and weight decay dominate. For fine-tuning, reducing or removing dropout is often beneficial since the pretrained features are already regularized.

---

*See also: [[regularization-weight-decay]] · [[normalization-layers]] · [[bias-variance-double-descent]] · [[ensemble-methods]] · [[bayesian-inference]]*
