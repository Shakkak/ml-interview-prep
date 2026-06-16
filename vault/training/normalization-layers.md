---
title: Normalization Layers
tags: [training, deep-learning, batch-norm, layer-norm, normalization]
aliases: [BatchNorm, LayerNorm, GroupNorm, RMSNorm, InstanceNorm]
difficulty: 2
status: complete
depends_on: [backpropagation, initialization, backpropagation-advanced]
related: [backpropagation-advanced, attention-mechanism, cnn-architectures-guide]
---

# Normalization Layers

---

## Fundamental

**Internal covariate shift:** during training, the distribution of each layer's inputs changes as earlier layers' parameters update. Each layer must constantly adapt to a shifting input distribution — slowing learning, destabilizing gradients, and preventing large learning rates.

**Batch Normalization** (Ioffe & Szegedy, 2015): for a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$:

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i, \qquad \sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

where:
- $m$ — number of samples in the mini-batch
- $\mu_\mathcal{B}$ — batch mean: average activation value over the batch
- $\sigma^2_\mathcal{B}$ — batch variance: average squared deviation from the mean
- $\hat{x}_i$ — normalized activation: shifted to zero mean, scaled to unit variance
- $\epsilon$ — small constant (e.g., $10^{-5}$) for numerical stability (prevents division by zero when variance is near 0)
- $\gamma, \beta$ — learned per-channel scale and shift parameters; allow the network to undo normalization if it helps (e.g., if the optimal output is not zero-mean)
- $y_i$ — final output after rescaling

**Numerical example:** mini-batch activations $z = (2, 4, 6)$.
- $\mu = 4$, $\sigma^2 = 8/3 \approx 2.667$, $\sigma \approx 1.633$
- $\hat{z} \approx (-1.225, 0, +1.225)$

**Training vs inference:** during training, use actual batch statistics. At inference, use exponential moving averages maintained during training: $\mu_{running} \leftarrow (1-\alpha)\mu_{running} + \alpha\mu_\mathcal{B}$. **Never forget `model.eval()`** before inference — using batch statistics on a single test sample gives output = 0 for all activations (every sample is its own mean).

> [!warning] Forgetting `model.eval()` silently zeroes all BN outputs
> With batch size 1 at inference, the batch mean $\mu_\mathcal{B} = x$ (the single sample is its own mean) and
> $\sigma_\mathcal{B}^2 = 0$ (no variance in a one-element batch). Therefore $\hat{x} = (x - x)/\sqrt{0 + \epsilon} \approx 0$,
> and the layer outputs $y = \gamma \cdot 0 + \beta = \beta$ for every input — a constant, regardless of the
> actual activations. The model degrades silently: no error is thrown, predictions are just wrong.
> Fix: always call `model.eval()` before inference to switch to stored running statistics.

**Which dimensions are normalized:**

```
Batch Norm:     normalize over (N, H, W) for each C
Layer Norm:     normalize over (C, H, W) for each N
Instance Norm:  normalize over (H, W) for each N, C
Group Norm:     normalize over (H, W, C/G) for each N, group
```

---

## Intermediate

**Why BatchNorm works (four mechanisms):**
1. Controlled distribution → activations stay in non-saturating regime → gradients flow better
2. Batch statistics noise (computed over random mini-batch) → implicit regularization → can reduce dropout
3. Larger learning rates become stable
4. Smoothes the loss landscape → easier optimization (Santurkar et al., 2018 showed this is the primary mechanism, not reducing covariate shift)

**Layer Normalization** (Ba et al., 2016): normalizes over the feature dimension for each sample independently — no batch dimension involved. For a feature vector $x \in \mathbb{R}^d$:
$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_j = \gamma_j \hat{x}_j + \beta_j$$

where $j$ indexes the $d$ features, $\mu = \frac{1}{d}\sum_j x_j$ and $\sigma^2 = \frac{1}{d}\sum_j (x_j - \mu)^2$ are computed over all $d$ features of that single sample (no batch averaging), $\epsilon$ is a numerical stability constant, and $\gamma_j, \beta_j$ are per-feature learned scale and shift parameters.

**Why transformers use LayerNorm, not BatchNorm:**
1. Variable-length sequences: batch statistics need masking — awkward and error-prone.
2. Single-sample inference: LayerNorm uses the sample's own statistics (works for batch size 1); BatchNorm would use running averages.
3. No train/eval mode difference: same computation always.
4. Token-centric: normalizing across the feature dimension for each token independently is the natural operation.

**Instance Normalization:** normalize over $(H, W)$ per sample per channel. Used in style transfer — the channel statistics encode image style, and IN removes them, making features style-agnostic. Swapping $\gamma, \beta$ parameters injects the target style (Adaptive Instance Normalization, AdaIN).

**Group Normalization** (Wu & He, 2018): divide channels into $G$ groups, normalize over $(H, W, C/G)$ per sample. Batch-size independent like LayerNorm but operates on channel groups. Used in detection and video models where batch size is 1 per GPU due to high resolution.

**RMSNorm** (simplified LayerNorm, used in LLaMA, Mistral):
$$\hat{x}_j = \frac{x_j}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_j x_j^2}, \qquad y_j = \gamma_j \hat{x}_j$$

where $d$ is the feature dimension, $j$ indexes each feature, $\text{RMS}(x)$ is the root-mean-square of the feature vector (equivalent to the $L^2$ norm divided by $\sqrt{d}$), and $\gamma_j$ is the learned per-feature scale (no bias $\beta$ — mean subtraction is omitted entirely).

No mean subtraction, no bias $\beta$. Faster (~10–15% speedup) and empirically matches LayerNorm quality. The re-centering step turns out to be unnecessary.

**Pre-Norm vs Post-Norm in transformers:**
- Post-Norm (original transformer): $x' = \text{LayerNorm}(x + \text{Sublayer}(x))$ — residual passes through normalization.
- Pre-Norm (LLaMA, GPT-2+): $x' = x + \text{Sublayer}(\text{LayerNorm}(x))$ — residual bypasses normalization.

Pre-Norm is preferred: gradients flow back through the residual stream without normalization → consistent gradient magnitudes at all depths → stable training without warmup.

---

## Advanced

**BatchNorm's gradient has three terms:** from the chain rule through the batch statistics (see [[backpropagation-advanced]]). The three terms arise from differentiating through $\hat{x}_i$, through $\mu_B$ (which depends on all $x_j$), and through $\sigma_B^2$ (also depends on all $x_j$). The mean subtraction and variance division terms reduce gradient correlations between samples and normalize gradient scale — keeping Jacobian singular values near 1 and preventing vanishing/exploding gradients. This is why BatchNorm is a more effective gradient stabilizer than just weight normalization.

**The flatness of loss landscape with BN (Santurkar et al., 2018):** a key experimental finding: training without BatchNorm on CIFAR-10 results in loss landscape gradient magnitudes varying by $\sim 10^3$; with BatchNorm, the variation is $\sim 10^1$. This directly allows larger learning rates. The claim that BN reduces "internal covariate shift" was shown empirically to be secondary — even adding noise to undo BN's distributional normalization still preserves the landscape smoothing benefit.

**Post-Norm requires careful initialization for deep transformers:** with post-norm, the gradient through each layer is $\frac{\partial \text{LayerNorm}}{\partial x}$ applied to the sum of residual and sublayer. Deep post-norm transformers require smaller initialization scale and careful warmup scheduling; without these, gradients at the early layers are extremely small. Admin initialization (Liu et al., 2020) adds learnable residual weights $\omega$ as $x' = \text{LayerNorm}(\omega_l x + \text{Sublayer}(x))$ initialized to small values, stabilizing post-norm training without warmup.

**Deep Norm** (Wang et al., 2022, used in DeepNet): a new normalization placement that scales the residual by $\alpha$:
$$x' = \text{LayerNorm}(\alpha x + \text{Sublayer}(x))$$
where $\alpha > 1$ is a fixed scalar that upweights the residual stream (making the skip path dominate early in training), and weights are initialized smaller by a factor $\beta < 1$. Theoretical analysis shows this bounds the expected gradient update at initialization to $O(1)$ regardless of depth, enabling stable training of 1000-layer transformers without warmup — a significant improvement over both Pre-Norm and Post-Norm for extreme depths.

---

## Links

- [[backpropagation]] — normalization changes the gradient landscape that backprop traverses; BN keeps Jacobian singular values near 1
- [[initialization]] — with BatchNorm, initialization scale matters less for activations but still affects gradient scale at step 0
- [[backpropagation-advanced]] — the full BatchNorm gradient has three terms from differentiating through batch statistics
- [[attention-mechanism]] — transformers use LayerNorm (not BatchNorm) because batch statistics are incompatible with variable-length sequences
- [[optimizer-adam]] — Pre-Norm stabilizes training and reduces reliance on warmup scheduling in Adam
- [[cnn-architectures-guide]] — BatchNorm was introduced for CNNs; architecture choices evolved around it
