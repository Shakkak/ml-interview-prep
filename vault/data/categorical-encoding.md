---
title: Categorical Encoding
tags: [categorical-encoding, one-hot, target-encoding, embeddings, feature-engineering]
aliases: [one-hot encoding, ordinal encoding, target encoding, categorical features, entity embeddings]
difficulty: 1
status: complete
related: [feature-preprocessing, gradient-boosting, logistic-regression, word-embeddings, data-leakage]
---

# Categorical Encoding

---

## Fundamental

### The Problem

Most ML models require numerical inputs, but real-world data contains categorical features: country, product category, user ID. Encoding these correctly affects model quality, memory, and training stability.

### One-Hot Encoding

Create a binary column for each category:

"Color" ∈ {red, green, blue} → columns `is_red`, `is_green`, `is_blue`

**Properties:**
- No ordinal assumption — each category treated independently
- Dimensionality: $C$ binary columns for $C$ categories
- Sparse: one 1, the rest 0
- Works well with linear models and neural networks
- **Problem:** high cardinality ($C$ = 10,000+ for country codes, user IDs) → sparse, high-dimensional, may overfit

**Dummy encoding:** drop one column to avoid perfect multicollinearity (for linear models). Not needed for tree models or neural networks.

### Ordinal Encoding

Map categories to integers 0, 1, 2, ..., C-1. Appropriate only when the ordering is meaningful (education level: high school=0, bachelor=1, master=2, PhD=3).

**Problem:** imposing arbitrary order (e.g., red=0, green=1, blue=2) tells the model colors have a magnitude relationship, which is wrong.

---

## Intermediate

### Target Encoding

Replace each category with the mean target value for that category:

$$\text{TargetEnc}(c) = \frac{\sum_{i: x_i = c} y_i}{|\{i: x_i = c\}|}$$

**Benefits:**
- Single numeric feature regardless of cardinality
- Directly informative for the target
- Works well with tree models and gradient boosting

**Leakage risk:** computing target means on training data includes the current row's target, causing data leakage (see [[data-leakage]]). 

**Leave-one-out (LOO) encoding:** exclude the current row when computing its category mean:
$$\text{TargetEnc}_\text{LOO}(x_i, c) = \frac{\sum_{j \neq i: x_j = c} y_j}{|\{j \neq i: x_j = c\}|}$$

**Smoothed target encoding:** blend the category mean with the global mean to handle rare categories:
$$\hat{y}_c = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}$$
where $n_c$ is the count for category $c$, $m$ is a smoothing factor (number of pseudo-observations at global mean), $\bar{y}$ is the global mean.

### CatBoost Encoding (Ordered Statistics)

CatBoost (see [[gradient-boosting]]) uses **ordered target statistics** — for each sample $i$ in a random permutation, compute the category statistic using only samples $j < i$ (earlier in the permutation). This prevents leakage by construction while still using target information.

### Frequency Encoding

Replace each category with its frequency (count or proportion) in the training set:
$$\text{FreqEnc}(c) = \frac{|\{i: x_i = c\}|}{N}$$

No leakage, no smoothing needed. Useful as a second feature alongside one-hot or target encoding. Captures "rarity" signal.

---

## Advanced

### Entity Embeddings

For high-cardinality categorical features (e.g., user ID with 1M users), train a learnable embedding lookup table $E \in \mathbb{R}^{C \times d}$:

$$\text{Embed}(c) = E[c] \in \mathbb{R}^d$$

The embeddings are learned end-to-end with the main model. Similar to word embeddings (see [[word-embeddings]]) — each category gets a dense representation capturing its semantic position relative to others.

**Practical:** work well with neural networks; also extracted post-training as features for other models. Used in recommendation systems (user/item embeddings), ad click prediction, tabular deep learning (TabNet, FT-Transformer).

### Hashing Trick

For very high cardinality (new categories at inference, streaming data), map categories to a fixed-size integer using a hash function:
$$\text{hash}(c) \bmod K$$

Collisions (two different categories mapped to the same bucket) are accepted as a necessary tradeoff for bounded dimensionality. Used in Vowpal Wabbit, online learning systems.

### Comparison Table

| Method | Cardinality | Leakage risk | Works with trees | Works with linear models |
|--------|-------------|-------------|-----------------|--------------------------|
| One-hot | Low (< 100) | None | Yes | Yes |
| Ordinal | Ordered | None | Yes | Sometimes |
| Target | Any | High if naïve | Excellent | Good |
| Frequency | Any | None | Good | Moderate |
| Entity embedding | Very high | None | Via extracted features | Via extracted features |
| Hashing | Any | None | Via features | Yes |

*See also: [[feature-preprocessing]] · [[gradient-boosting]] · [[word-embeddings]] · [[data-leakage]]*
