---
title: Test-Time Strategies — TTA, MC Dropout, and Model Ensembling
tags: [tta, test-time-augmentation, mc-dropout, model-ensembling, uncertainty, deep-ensembles, inference]
aliases: [test-time augmentation, TTA, MC Dropout, model ensembling, deep ensembles, snapshot ensemble]
difficulty: 2
status: complete
related: [regularization-dropout, data-augmentation, model-calibration, ensemble-methods, bayesian-inference]
---

# Test-Time Strategies — TTA, MC Dropout, and Model Ensembling

---

## Fundamental

A single deterministic forward pass produces one prediction. Accuracy and uncertainty estimates both improve by aggregating multiple predictions from related model evaluations. Three mechanisms achieve this:

1. **Test-Time Augmentation (TTA):** run the same model on multiple augmented views of the input.
2. **MC Dropout:** run multiple stochastic forward passes through the same model.
3. **Deep Ensembles:** average predictions from multiple independently trained models.

All three reduce **prediction variance** at the cost of additional inference compute.

### Test-Time Augmentation

Apply $K$ augmentations to a test image, run inference on each, average probabilities:

$$\hat{p}(y \mid x) = \frac{1}{K}\sum_{k=1}^K p(y \mid \text{aug}_k(x))$$

**Standard augmentations for TTA:**
- Horizontal flip ($K=2$): always, cheap, reliable +0.3–0.5% on ImageNet.
- Multi-crop ($K=5$): 4 corners + center crop at standard size.
- Multi-scale ($K=3$): resize to {480, 512, 544} and predict on each.

**Aggregation:** soft voting (mean of softmax probabilities) outperforms hard voting (majority of argmax). Soft voting preserves calibration information in the tail of the distribution.

**Cost vs benefit:** TTA($K$) requires $K$ forward passes per example. For $K \leq 5$ and moderate model size, the compute cost is acceptable. For real-time inference or very large models, TTA is often impractical.

---

## Intermediate

### MC Dropout for Uncertainty Estimation

Standard dropout is disabled at test time (activations scaled by keep probability $1-p$). **MC Dropout** (Gal & Ghahramani, 2016) keeps dropout active and runs $K$ stochastic forward passes:

$$\hat{p}(y \mid x) = \frac{1}{K}\sum_{k=1}^K p_{\theta^{(k)}}(y \mid x)$$

where $\theta^{(k)}$ is the model with independently sampled dropout masks for pass $k$.

**Epistemic uncertainty** (model uncertainty, reducible with more data):
$$\text{Var}[p(y \mid x)] = \frac{1}{K-1}\sum_k\left(p_{\theta^{(k)}} - \hat{p}\right)^2$$

High variance across passes signals an out-of-distribution input — the model's functional uncertainty is high. Gal & Ghahramani showed this approximates the posterior predictive distribution in a Bayesian neural network via the correspondence between dropout and variational inference with a Bernoulli approximate posterior.

**Requirement:** the model must have been trained with dropout layers. ResNets without dropout cannot use MC Dropout directly without retraining.

### Deep Ensembles

Train $M$ independent models with different random seeds (initialization + data order). Average predictions:

$$\hat{p}(y \mid x) = \frac{1}{M}\sum_{m=1}^M p_{\theta_m}(y \mid x)$$

**Why better than any single model:** different seeds find different local minima in weight space, each making different errors on different examples. The ensemble averages out uncorrelated errors. Lakshminarayanan et al. (2017) showed that $M=5$ deep ensembles consistently outperform single models by 0.5–2% accuracy and provide better-calibrated uncertainties than MC Dropout.

**Cost:** $M \times$ memory. But inference can run in parallel — latency is unchanged if you have $M$ GPUs. Production compromise: $M=3$ or $M=5$ with shared backbone weights (train shared encoder, ensemble only the classification heads).

### Cost vs Accuracy Table

| Method | Accuracy gain | Uncertainty | Inference cost |
|---|:-:|:-:|:-:|
| Single model | baseline | Poorly calibrated | 1× |
| TTA ($K=5$) | +0.5–1.5% | Not available | 5× |
| MC Dropout ($K=30$) | +0.3–0.8% | Epistemic ✓ | 30× |
| Deep ensemble ($M=5$) | +0.8–2.0% | Epistemic + calibrated ✓ | 5× (parallel) |
| Ensemble + TTA | +1.5–3.0% | Best | $5M$× |

---

## Advanced

### Uncertainty Decomposition

The **predictive uncertainty** of a model can be decomposed into:

$$\underbrace{H[y \mid x, \mathcal{D}]}_{\text{total}} = \underbrace{I(y; \theta \mid x, \mathcal{D})}_{\text{epistemic}} + \underbrace{\mathbb{E}_{p(\theta \mid \mathcal{D})}[H[y \mid x, \theta]]}_{\text{aleatoric}}$$

**Epistemic** (model) uncertainty: reducible by observing more labeled data. High for out-of-distribution inputs. Captured by disagreement between ensemble members or MC Dropout passes.

**Aleatoric** (data) uncertainty: irreducible — caused by genuine label noise or inherent ambiguity in the input. Does not decrease with more data. Captured by the average entropy of individual models, not their disagreement.

Deep ensembles estimate total uncertainty; the epistemic component is the variance across members. This decomposition matters practically: if an input has high epistemic uncertainty, collect more training data near it. If it has high aleatoric uncertainty, collecting more data won't help — the input is genuinely ambiguous.

### Snapshot Ensembles: Free Diversity from One Training Run

Train with a **cyclical learning rate** schedule (warm restarts). Each restart converges to a different local minimum. Save a snapshot at each cycle end. Ensemble the snapshots.

Total training cost ≈ one normal training run. Diversity is lower than independently initialized models (the snapshots start from related initializations) but substantially better than a single model. Standard in Kaggle competitions where training large models is expensive.

**Stochastic Weight Averaging (SWA)** extends this: average the weights (not predictions) of snapshots taken during the last epochs of training. SWA gives better generalization than the last checkpoint alone and is now implemented natively in PyTorch (`torch.optim.swa_utils`).

### When to Use Each Strategy

| Scenario | Recommendation |
|---|---|
| Competition, accuracy maximization | Deep ensemble + TTA |
| Production, uncertainty-aware (medical, autonomous) | Deep ensemble (reliable epistemic uncertainty) |
| Cheap uncertainty estimate, model has dropout | MC Dropout (30 passes) |
| Real-time inference, strict latency budget | Single model (possibly with distillation from ensemble) |
| Training budget tight, want some diversity | Snapshot ensemble |

---

*See also: [[regularization-dropout]] · [[data-augmentation]] · [[model-calibration]] · [[ensemble-methods]] · [[bayesian-inference]]*
