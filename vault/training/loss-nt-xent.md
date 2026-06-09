---
title: NT-Xent Loss (Contrastive)
tags: [loss, contrastive, self-supervised, simclr]
aliases: [NT-Xent, InfoNCE, contrastive loss, normalized temperature cross-entropy]
difficulty: 2
status: complete
related: [loss-cross-entropy, loss-kl-divergence, contrastive-learning, entropy-mutual-info]
---

# NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy)

---

## Fundamental

NT-Xent is the training loss for [[contrastive-learning|SimCLR]] contrastive self-supervised learning. It trains an encoder to produce similar representations for two views of the same image and dissimilar representations for different images.

**Setup:** given a batch of $N$ images, augment each twice to get $2N$ views. Encode and project each: $z_i = g(f(x_i))$, normalized to unit sphere.

For views $i$ and $j$ from the same original image (positive pair):
$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}[k \neq i]\,\exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}(u,v) = u \cdot v$ (cosine similarity for unit vectors) and $\tau$ is temperature. Total loss: average over all $2N$ positive pairs.

**Connection to cross-entropy:** NT-Xent is exactly cross-entropy on a classification problem — "which of the $2N-1$ other views is the positive pair?" The model must score the true positive higher than all $2N-2$ negatives.

**Worked example** ($N=2$, $\tau=0.5$):
```
z₁ = [1.0, 0.0],  z₁'= [0.91, 0.41]  (positive pair A)
z₂ = [0.0, 1.0],  z₂'= [0.1, 0.99]   (positive pair B)
```
Cosine similarities: sim(z₁, z₁') = 0.91, sim(z₁, z₂) = 0.00, sim(z₁, z₂') = 0.10.

$$\ell_{1,1'} = -\log\frac{e^{1.82}}{e^{1.82} + e^{0.00} + e^{0.20}} = -\log(0.735) = 0.308$$

---

## Intermediate

**Role of temperature $\tau$:**

| $\tau$ | Effect |
|--------|--------|
| Small (0.05–0.1) | Sharp — dominates by hardest negatives; discriminative but can be unstable |
| Large (0.5–1.0) | Soft — all negatives contribute equally; stable but weak signal |
| SimCLR default | 0.07 — empirically optimal balance |

Temperature effectively controls the concentration of the representation: low temperature encourages tight clustering of positives, high temperature allows looser clusters. Too low causes gradient vanishing for easy negatives; too high weakens the learning signal.

**Why many negatives matter:** the denominator contains $2N-2$ negatives. With few negatives, most are "easy" (very dissimilar from different classes) and contribute near-zero gradient. With many diverse negatives (large batch = 4096 in SimCLR), semantically hard negatives (same scene, different object) appear and provide strong gradient signal. This is why SimCLR requires very large batch sizes to work well.

**Alignment and uniformity analysis** (Wang & Isola, 2020): NT-Xent implicitly optimizes two properties:
- **Alignment:** $\mathcal{L}_{align} = \mathbb{E}_{(x,x^+)\sim p_{pos}}[\|f(x) - f(x^+)\|^2]$ — positive pairs should be close.
- **Uniformity:** $\mathcal{L}_{unif} = \log \mathbb{E}_{(x,y)\sim p_{data}^2}[e^{-2\|f(x)-f(y)\|^2}]$ — representations should be spread uniformly on the hypersphere.

NT-Xent optimizes both simultaneously: the numerator drives alignment, the denominator drives uniformity. High-quality representations require both — aligning without uniformity leads to collapsed representations.

**MoCo vs SimCLR implementation:** SimCLR uses large in-batch negatives (requires batch ≥ 4096). MoCo (He et al., 2020) maintains a momentum encoder and a memory bank of $K$ recent representations as negatives (default $K=65536$), allowing contrastive training with small batches. The momentum encoder update $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ (with $m=0.999$) prevents the key encoder from changing too fast, maintaining representation consistency across the memory bank.

---

## Advanced

**NT-Xent as a lower bound on [[entropy-mutual-info|mutual information]]:** the InfoNCE bound (Oord et al., 2018) establishes:
$$I(z_i; z_j) \geq \log(N) - \mathcal{L}_{NT-Xent}$$

where $N$ is the number of negatives. Maximizing NT-Xent loss (making it less negative) lower-bounds the mutual information between the two views. This bound is tight when the optimal critic is used. However, the bound is inherently limited: with $N$ negatives, the maximum achievable bound is $\log N$ nats — larger batches allow tighter MI estimation. This theoretical insight motivates the empirical finding that larger batches systematically improve contrastive learning.

**Why the projection head matters (SimCLR, Chen et al., 2020):** representations used for downstream tasks should be taken from the encoder $f$, not the projection head $g$. The projection head learns to be invariant to augmentations, discarding information (like color, texture) that is useful for other tasks but redundant for the contrastive objective. Without the projection head, the encoder itself must be invariant, harming transferability. This explains a seemingly counterintuitive finding: using a non-linear projection head improves representation quality even though the head is discarded after pretraining.

**BYOL and the collapse problem without negatives** (Grill et al., 2020): BYOL removes negatives entirely by training an online network to predict the output of a momentum target network. Theoretically, without negatives the model could collapse to a constant output (zero gradient from the uniformity term). BYOL avoids collapse through the asymmetry of the online/target network pair: the stop-gradient on the target network and the predictor head on the online network create an implicit regularization that prevents collapse. Detailed analysis (Tian et al., 2021) showed batch normalization in the network is the actual mechanism preventing collapse in BYOL — not the momentum encoder or predictor.

**Supervised contrastive loss** (Khosla et al., 2020): extends NT-Xent to supervised learning by treating all samples of the same class as positives:
$$L_{sup} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $P(i)$ is the set of all positives for anchor $i$. This consistently outperforms [[loss-cross-entropy|cross-entropy training]] (0.5–1% top-1 on ImageNet with the same architecture) by learning representations with better inter-class margins and intra-class compactness.

---

*See also: [[contrastive-learning]] · [[loss-cross-entropy]] · [[entropy-mutual-info]]*
