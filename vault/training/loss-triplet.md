---
title: Triplet Loss and Contrastive Losses
tags: [triplet-loss, contrastive-loss, metric-learning, hard-negative-mining, siamese]
aliases: [triplet loss, margin loss, metric learning loss, hard negatives, semi-hard negatives]
difficulty: 2
status: complete
related: [contrastive-learning, loss-nt-xent, self-supervised-overview, word-embeddings, loss-cross-entropy]
depends_on: [contrastive-learning, loss-nt-xent, backpropagation]
---

# Triplet Loss and Contrastive Losses

---

## Fundamental

### Metric Learning Goal

**Metric learning** trains an embedding function $f_\theta: \mathcal{X} \to \mathbb{R}^d$ such that similar inputs are nearby and dissimilar inputs are far apart in the embedding space (under Euclidean or cosine distance).

Applications: face verification, person re-identification, semantic search, few-shot learning, image retrieval.

### Triplet Loss

Given a triplet $(a, p, n)$ — anchor $a$, positive $p$ (same class/similar), negative $n$ (different class/dissimilar) — the **triplet loss** is:

$$\mathcal{L} = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha)$$

where $\alpha > 0$ is the **margin**. The loss is zero when $d_{ap} + \alpha < d_{an}$ — the positive is closer than the negative by at least $\alpha$.

**Intuition:** push the anchor-positive pair closer while pushing the anchor-negative pair further, maintaining a minimum separation margin.

---

## Intermediate

### Mining Strategies

Triplet loss with random triplets saturates quickly — most random negatives are easy (very far away) and contribute zero loss. Effective training requires hard or semi-hard negatives.

**Definitions for anchor $a$:**
- **Easy negative:** $d_{an} > d_{ap} + \alpha$ — loss already zero
- **Semi-hard negative:** $d_{ap} < d_{an} < d_{ap} + \alpha$ — some loss, informative gradient
- **Hard negative:** $d_{an} < d_{ap}$ — closer than the positive, large loss

**Online hard negative mining (OHNM):** within each batch, for each anchor select the hardest valid triplet. Most common strategy in practice.

**Problem with pure hard negatives:** selecting only the hardest negatives can introduce noisy gradients (outliers, label errors) and cause training instability. Semi-hard mining is often more robust.

### Variants

**Contrastive loss (Chopra et al., 2005):** operates on pairs $(x_1, x_2)$ with label $y \in \{0,1\}$ (same/different):

$$\mathcal{L} = y \cdot d(x_1, x_2)^2 + (1-y) \cdot \max(0, \alpha - d(x_1, x_2))^2$$

where $y=1$ for same class pairs, $y=0$ for different class pairs, $d(x_1,x_2)$ = embedding distance, $\alpha$ = margin (dissimilar pairs must be pushed beyond this distance).

Pulls similar pairs together, pushes dissimilar pairs beyond margin $\alpha$.

**N-pair loss / Multi-similarity loss:** use all pairs in a batch rather than pre-selected triplets. The NT-Xent loss (see [[loss-nt-xent]]) used in SimCLR is a softmax-normalized form of this — treating one positive and $2(N-1)$ negatives simultaneously.

**ArcFace / CosFace (angular margin losses):** add a margin directly in the angle between embedding and class center on the unit hypersphere. Used in face recognition:
- CosFace: $\cos\theta_y - m$ (additive cosine margin)
- ArcFace: $\cos(\theta_y + m)$ (additive angular margin)

These are more geometrically principled than Euclidean triplet losses for normalized embeddings.

---

## Advanced

### Relationship to Cross-Entropy

Softmax cross-entropy over class prototypes is closely related to metric learning:
- Each class prototype acts as a "positive anchor"
- Softmax denominator effectively creates $K-1$ negatives per sample
- Fine-tuning with cross-entropy produces metric-like representations

**Prototypical networks** (Snell et al., 2017) for few-shot learning: compute class prototypes as mean embeddings of support set; classify by nearest prototype. Training with cross-entropy over episodes implicitly trains a metric space.

### Batch Construction for Metric Learning

**PK sampling:** sample $P$ classes, $K$ images per class per batch. Guarantees $P \cdot K (K-1)$ valid anchor-positive pairs and $P(P-1)K^2$ anchor-negative pairs per batch.

Larger batches → more diverse negatives → better mining → better representations. This motivates large-batch contrastive training (SimCLR uses batch sizes of 4096+).

## Links

- [[contrastive-learning]] — triplet loss is pairwise contrastive learning: anchor, positive, negative must satisfy $d(a,p) + m < d(a,n)$; NT-Xent (InfoNCE) uses all in-batch negatives instead of explicit triplets
- [[loss-nt-xent]] — NT-Xent is a multi-negative extension of triplet loss; at batch size $N$, each anchor has $2(N-1)$ negatives vs. one in triplet loss; more negatives = better signal
- [[backpropagation]] — triplet loss has a combinatorial mining problem: valid triplets satisfying $d^+ < d^-$ don't contribute gradients; hard negative mining selects triplets near the margin
- [[self-supervised-overview]] — triplet loss was the original metric learning objective for face recognition (FaceNet); it has been largely superseded by NT-Xent in SSL settings
- [[word-embeddings]] — word2vec uses a softmax or noise-contrastive objective equivalent to triplet loss; the word embedding geometry reflects semantic relationships learned from co-occurrence statistics
