---
title: Adversarial Training and Robustness
tags: [adversarial-training, adversarial-examples, pgd, fgsm, certified-robustness]
aliases: [adversarial training, adversarial examples, FGSM, PGD attack, certified robustness, adversarial robustness]
difficulty: 2
status: complete
related: [data-augmentation, regularization-weight-decay, loss-cross-entropy, generalization-bounds, normalization-layers]
---

# Adversarial Training and Robustness

---

## Fundamental

### Adversarial Examples

**Adversarial examples** (Szegedy et al., 2013) are inputs with imperceptible perturbations that cause confident misclassification:

$$x_\text{adv} = x + \delta, \quad \|\delta\|_p \leq \epsilon, \quad f(x_\text{adv}) \neq f(x)$$

A well-trained ImageNet classifier can be fooled by a perturbation $\delta$ with $\|\delta\|_\infty \leq 4/255$ (invisible to humans).

**Why they exist:** neural networks learn non-robust features — high-frequency statistical regularities in the training distribution that happen to be highly predictive. These features can be exploited by small, targeted perturbations.

### FGSM: Fast Gradient Sign Method

**FGSM (Goodfellow et al., 2014):** single-step attack in the $\ell_\infty$ ball:

$$x_\text{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f_\theta(x), y))$$

Take one gradient step in the direction that maximally increases the loss. Fast but weak — easily defended against.

---

## Intermediate

### PGD: Projected Gradient Descent

**PGD attack (Madry et al., 2018):** multi-step FGSM with projection back to the $\ell_p$ ball after each step:

$$x^{t+1} = \Pi_{\|x - x_0\|_p \leq \epsilon}\!\left(x^t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L})\right)$$

PGD with many steps ($k = 20$–$100$) is considered the standard strong attack. Madry et al. proved that PGD approximate solves the inner maximization in the minimax adversarial training objective.

### Adversarial Training

**Madry et al. (2018)** formulate robust training as minimax optimization:

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}}\!\left[\max_{\|\delta\|_p \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y)\right]$$

**Adversarial training algorithm:**
1. For each batch, generate adversarial examples using PGD
2. Train the model on the adversarial examples

This is significantly more expensive than standard training ($k$ PGD steps per batch = $k\times$ more forward-backward passes) but produces models that are robust to $\ell_p$ perturbations up to $\epsilon$.

**Accuracy-robustness tradeoff:** adversarially trained models have lower standard accuracy (~10–15% lower) and higher adversarial accuracy. This tradeoff is fundamental — cannot simultaneously maximize both.

### Trades and Other Methods

**TRADES (Zhang et al., 2019):** decomposes robustness into natural accuracy + boundary-smoothness regularization:

$$\mathcal{L}_\text{TRADES} = \mathcal{L}_\text{CE}(f(x), y) + \beta \max_{\|\delta\|_p \leq \epsilon} \text{KL}(f(x) \| f(x+\delta))$$

The KL term encourages the model's predictions to be consistent across perturbations — smooths the decision boundary near training points. Often better tradeoff than pure adversarial training.

---

## Advanced

### Certified Robustness

**Empirical robustness:** no attack found a counterexample (no guarantee).

**Certified robustness:** provably no adversarial example exists within the $\ell_p$ ball.

**Randomized smoothing (Cohen et al., 2019):** construct a smoothed classifier $\bar{f}(x) = \mathbb{E}_{\delta \sim \mathcal{N}(0,\sigma^2 I)}[f(x + \delta)]$ and give probabilistic certificates:

If $\bar{f}(x)$ returns class $c_A$ with probability $p_A > 0.5$, then $\bar{f}$ is certified for any perturbation $\|\delta\|_2 \leq R$ where $R = \sigma \Phi^{-1}(p_A)$.

Randomized smoothing is scalable (works with any black-box classifier) but weakens the classifier (Gaussian noise blurs fine details).

### Adversarial Training as Data Augmentation

Adversarial examples can be viewed as a form of hard data augmentation — they expose the model to its worst-case inputs. This connects adversarial training to:
- Mixup (interpolated examples)
- CutMix (spatial mixing)

But adversarial examples are input-adaptive and specifically crafted to fool the model, making them much more challenging than random augmentations.

*See also: [[data-augmentation]] · [[loss-cross-entropy]] · [[regularization-weight-decay]] · [[generalization-bounds]]*
