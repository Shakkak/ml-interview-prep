---
title: Class Imbalance
tags: [class-imbalance, smote, oversampling, undersampling, focal-loss, cost-sensitive]
aliases: [class imbalance, imbalanced dataset, SMOTE, oversampling, undersampling, focal loss]
difficulty: 1
status: complete
related: [loss-focal, loss-cross-entropy, data-augmentation, evaluation-metrics-guide, logistic-regression]
depends_on: [loss-cross-entropy, evaluation-metrics-guide, distributions-overview]
---

# Class Imbalance

---

## Fundamental

### The Problem

In many real-world problems, one class is much rarer than others: fraud detection (0.1% fraud), medical diagnosis (1% positive), object detection (many background regions, few objects). Training a model naively on heavily imbalanced data leads to:

- Predicting the majority class for everything (high accuracy, useless model)
- Loss dominated by majority class examples — minority class barely influences gradients
- Decision boundary biased toward majority class

**Metrics that hide the problem:** accuracy. A model that always predicts "no fraud" achieves 99.9% accuracy on a 0.1% fraud dataset but is completely useless. Use **precision, recall, F1, AUC-PR** (see [[evaluation-metrics-guide]]).

---

## Intermediate

### Resampling Approaches

**Oversampling (minority class):**
- Random oversampling: duplicate minority examples — cheap but can overfit exact copies
- **SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al., 2002):** generate synthetic examples by interpolating between a minority example and its $k$-nearest minority neighbors: $x_\text{new} = x_i + \lambda(x_j - x_i)$, $\lambda \sim U[0,1]$. Creates new examples along the feature-space manifold of the minority class.
- **ADASYN:** adaptive SMOTE — generate more synthetic examples for minority points in dense majority regions (harder boundary points).

**Undersampling (majority class):**
- Random undersampling: discard majority examples — fast but discards information
- **Tomek links:** remove pairs of nearby opposite-class samples (both are borderline) — cleans the decision boundary without large information loss
- **Cluster centroids:** replace majority clusters with their centroid — reduces majority while preserving structure

**Combined:** SMOTE + Tomek links (oversample minority, then clean noisy boundary samples) is a common default.

### Class-Weighted Loss

Instead of resampling, assign higher loss weight to minority class examples:

$$\mathcal{L} = -\sum_i w_{y_i} \log p(y_i | x_i), \quad w_k \propto \frac{1}{n_k} \text{ or } \frac{N}{K \cdot n_k}$$

where $n_k$ is the count of class $k$. This is equivalent to oversampling but more numerically stable. Available as `class_weight='balanced'` in scikit-learn, `pos_weight` in PyTorch BCE.

### Focal Loss

**Focal loss (Lin et al., 2017, RetinaNet)** dynamically down-weights easy examples:

$$\mathcal{L}_\text{focal} = -(1 - p_t)^\gamma \log p_t$$

where $p_t = p$ if $y=1$, $p_t = 1-p$ if $y=0$, and $\gamma \geq 0$ is the focusing parameter.

- Easy examples (high $p_t$): $(1-p_t)^\gamma \to 0$ — nearly no loss contribution
- Hard examples (low $p_t$): $(1-p_t)^\gamma \approx 1$ — full loss

With $\gamma = 2$, an example classified with $p_t = 0.9$ contributes $0.01\times$ as much loss as an equivalent hard example. This naturally focuses training on minority/hard examples without explicit resampling.

---

## Advanced

### When to Use Which

| Approach | Best for | Caution |
|----------|----------|---------|
| SMOTE | Low-dimensional tabular data | Can generate unrealistic examples in high-D |
| Class weights | Any model, quick fix | Doesn't change the data distribution |
| Focal loss | Neural networks, detection | $\gamma$ hyperparameter to tune |
| Undersampling | Very large majority class | Discards majority data |
| Two-stage threshold | Post-training calibration | Requires validation set |

**Decision threshold tuning:** for binary classification, the default threshold 0.5 is rarely optimal for imbalanced data. Plot the precision-recall curve and choose the threshold that maximizes $F_\beta$ for the desired $\beta$ (recall-weighted for fraud detection, precision-weighted for spam).

### Deep Learning and Imbalance

For deep networks, class imbalance interacts with:
- **Batch sampling:** ensure each batch contains at least a few minority examples (stratified sampling within batches)
- **Early stopping:** monitor minority class recall, not overall accuracy
- **Transfer learning:** pretrained features may already encode minority class structure — fine-tuning with class weights often sufficient

## Links

- [[loss-cross-entropy]] — standard cross-entropy treats all classes equally; class-weighted cross-entropy scales the loss by $1/\text{class frequency}$ to up-weight minority classes
- [[loss-focal]] — focal loss $(1-p_t)^\gamma$ down-weights easy (well-classified majority) examples and focuses training on hard minority examples; used in RetinaNet for object detection
- [[evaluation-metrics-guide]] — accuracy is misleading for imbalanced data (99% accuracy on 99% majority); use precision, recall, F1, AUROC, or AP over the minority class
- [[distributions-overview]] — class imbalance means the marginal distribution $p(y)$ is skewed; both loss reweighting and sampling methods correct for this prior shift
- [[data-augmentation]] — SMOTE synthesizes minority class samples in feature space by interpolating between nearest neighbors; it is a data-level alternative to loss reweighting
