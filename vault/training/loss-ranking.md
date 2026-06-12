---
title: Ranking Losses
tags: [ranking-losses, lambdarank, ranknet, listwise, pairwise-ranking, learning-to-rank]
aliases: [ranking loss, LambdaRank, RankNet, ListNet, pairwise ranking, learning to rank, NDCG]
difficulty: 2
status: complete
related: [loss-triplet, loss-cross-entropy, loss-mse, contrastive-learning, evaluation-metrics-guide]
---

# Ranking Losses

---

## Fundamental

### The Ranking Problem

Given a query $q$ and a set of documents $\{d_1, \ldots, d_n\}$ with relevance labels $\{y_1, \ldots, y_n\}$, learn a scoring function $f(q, d_i)$ such that documents are ranked by relevance.

**Why not use regression/classification?** The ranking metric (NDCG, MRR, MAP) is non-differentiable — it depends on the relative order of scores, which involves argmax. Ranking losses approximate or bound these non-differentiable objectives.

**NDCG (Normalized Discounted Cumulative Gain):**
$$\text{NDCG}@k = \frac{1}{\text{IDCG}@k}\sum_{i=1}^k \frac{2^{y_{\pi(i)}} - 1}{\log_2(i+1)}$$
where $\pi(i)$ is the rank of position $i$. Not directly differentiable.

---

## Intermediate

### Pointwise, Pairwise, Listwise

**Pointwise:** treat each document independently — apply regression or classification loss to each $(q, d_i)$ pair independently. Ignores relative order. Simple but misses the ranking structure.

**Pairwise:** train on pairs $(d_i, d_j)$ where $y_i > y_j$. Optimize to score $d_i$ higher than $d_j$:

**RankNet (Burges et al., 2005):**
$$P_{ij} = \sigma(f(q, d_i) - f(q, d_j)) = \frac{1}{1 + e^{-(s_i - s_j)}}$$
$$\mathcal{L} = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij})\log(1 - P_{ij})$$
where $\bar{P}_{ij}$ is the observed preference (1 if $d_i$ is more relevant). Minimizing this pushes $s_i > s_j$ for relevant $d_i$.

The RankNet gradient: $\frac{\partial \mathcal{L}}{\partial s_i} = \sigma(s_i - s_j) - \bar{P}_{ij}$ — elegant form analogous to logistic regression.

**Listwise:** optimize a loss over the full ranked list.

**ListNet (Cao et al., 2007):** define a probability distribution over permutations based on scores (Plackett-Luce model) and minimize KL divergence from the target distribution. Differentiable approximation to directly optimizing ranking metrics.

### LambdaRank

**LambdaRank (Burges et al., 2006):** the key insight — you don't need a loss function to have gradients. Directly define the gradient:

$$\lambda_{ij} = \frac{-\partial \mathcal{L}}{\partial s_i}\bigg|_\text{RankNet} \cdot |\Delta \text{NDCG}_{ij}|$$

where $|\Delta \text{NDCG}_{ij}|$ is the change in NDCG if documents $i$ and $j$ were swapped. Multiplying the RankNet gradient by the NDCG change emphasizes pairs that matter most for the ranking metric.

**LambdaMART:** LambdaRank implemented with gradient boosted trees (see [[gradient-boosting]]). Uses $\lambda_{ij}$ as pseudo-residuals. State-of-the-art for web search ranking for many years.

---

## Advanced

### Approximating NDCG Directly

**SoftNDCG / ApproxNDCG (Qin et al., 2010):** replace the hard rank function (argmax) with a soft approximation:
$$\hat{\pi}(i) = 1 + \sum_{j \neq i} \sigma\!\left(\frac{s_j - s_i}{\tau}\right)$$

As $\tau \to 0$, $\hat{\pi}(i) \to$ true rank. Use the approximate rank in the NDCG formula for a differentiable surrogate. Temperature $\tau$ controls approximation quality vs gradient stability.

**NeuralNDCG (Pobrotyn & Bartczak, 2021):** uses the Sinkhorn operator to compute a differentiable soft permutation matrix that approximates the true ranking permutation.

### Contrastive Ranking in Retrieval

Modern dense retrieval (DPR, ColBERT) uses contrastive losses (in-batch negatives, see [[loss-triplet]]) rather than traditional ranking losses. Given a query, positive document, and $N-1$ in-batch negatives:

$$\mathcal{L} = -\log\frac{e^{s(q,d^+)/\tau}}{\sum_j e^{s(q,d_j)/\tau}}$$

This is equivalent to multiclass classification (predict which document is positive) — optimizes a smooth approximation of the ranking metric without requiring pairwise comparisons.

*See also: [[loss-triplet]] · [[contrastive-learning]] · [[loss-cross-entropy]] · [[gradient-boosting]] · [[evaluation-metrics-guide]]*
