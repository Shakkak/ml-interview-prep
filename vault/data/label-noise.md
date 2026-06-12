---
title: Label Noise and Noisy Labels
tags: [label-noise, noisy-labels, confident-learning, co-teaching, robust-training]
aliases: [label noise, noisy labels, confident learning, Co-teaching, symmetric noise, asymmetric noise]
difficulty: 2
status: complete
related: [regularization-label-smoothing, data-augmentation, cross-validation, loss-cross-entropy, bias-variance-double-descent]
---

# Label Noise and Noisy Labels

---

## Fundamental

### Types of Label Noise

Real-world datasets contain mislabeled examples. Label noise can be:

**Random (symmetric) noise:** a label is randomly flipped to any other class with probability $\epsilon$ independently of the true class. Uniform across all classes.

**Class-conditional (asymmetric) noise:** flipping probability depends on the class — certain classes are more likely to be confused (e.g., "dog" and "wolf" look similar → dog gets mislabeled as wolf more often than as "airplane").

**Instance-dependent noise:** flipping probability depends on both class and features — ambiguous examples are more likely to be mislabeled (the most realistic model).

### Effect on Training

With $\epsilon$ fraction of noisy labels:
- **Bias:** the learned decision boundary shifts to accommodate mislabeled examples
- **Variance:** training loss can decrease by memorizing mislabeled examples in overparameterized models
- **Double descent:** noisy labels create a "hard interpolation threshold" at the boundary between generalization and memorization

**Empirically:** models trained on ImageNet with 20–40% noise still achieve surprisingly good accuracy (~75% top-1) because:
1. The majority class is still correct
2. Cross-entropy provides soft gradients (mislabeled examples push less than correctly labeled ones at the right scale)

---

## Intermediate

### Label Smoothing as Noise Regularization

**Label smoothing** (Szegedy et al., 2016) replaces hard targets $y \in \{0,1\}^K$ with soft targets:

$$\tilde{y}_k = (1 - \epsilon) y_k + \epsilon / K$$

This intentionally introduces label uncertainty, preventing the model from becoming overconfident and making it more robust to actual label noise. With label smoothing $\epsilon = 0.1$ and true noise rate $\epsilon_\text{noise}$, the combined effect is $(1 - \epsilon)(1 - \epsilon_\text{noise})$ effective signal — the smoothing noise and label noise partially cancel.

### Confident Learning

**Confident Learning (Northcutt et al., 2021)** identifies and characterizes label errors in datasets:

1. Estimate the **joint distribution** of true labels vs noisy labels: $Q_{\tilde{y},y^*} \in \mathbb{R}^{K \times K}$
2. Use out-of-sample predicted probabilities $p(y | x)$ (from cross-validation) and calibrated thresholds
3. Identify examples where the model is confidently predicting a class different from the given label

**Prune-then-retrain:** remove identified noisy examples → retrain on cleaned dataset.

Used in **Cleanlab** (open-source library) — applied to find errors in ImageNet (~6% label error rate for some classes), Amazon Reviews, and many NLP benchmarks.

### Co-Teaching

**Co-Teaching (Han et al., 2018):** train two networks simultaneously, each selecting "clean" examples for the other based on small loss:

1. Network A selects examples with smallest cross-entropy loss (likely clean)
2. Passes them to Network B for training (and vice versa)

The two networks have different initialization → different error patterns → less likely to agree on noisy examples. This mutual filtering is more robust than self-filtering.

---

## Advanced

### Memorization of Noisy Labels

**DNN memorization dynamics:** clean examples are learned first (low loss, well-fitted early), noisy examples are memorized later (high loss, fitted slowly). This is the **early learning** phenomenon — exploited by many robust methods that use small-loss selection early in training.

**Why easy examples first:** clean examples are consistent with the majority — stochastic gradients from clean examples reinforce each other. Noisy examples are idiosyncratic — gradients conflict, slowing learning.

### Noise-Robust Loss Functions

Some loss functions are theoretically robust to symmetric label noise:

**Mean Absolute Error (MAE):** $\mathcal{L} = \|p - y\|_1$ is noise-robust in theory but hard to optimize in practice.

**Generalized Cross-Entropy:** $\mathcal{L} = (1 - p_{y^*}^q) / q$ interpolates between MAE ($q \to 0$) and CE ($q \to 1$). A small $q \in [0.5, 0.9]$ gives a robust version of cross-entropy.

**Symmetric Cross-Entropy:** combines forward and reverse KL — $\mathcal{L}_\text{SCE} = \alpha \cdot \text{CE}(p, y) + \beta \cdot \text{CE}(y, p)$ — the reverse KL term is robust to noisy labels.

*See also: [[regularization-label-smoothing]] · [[cross-validation]] · [[loss-cross-entropy]] · [[data-augmentation]]*
