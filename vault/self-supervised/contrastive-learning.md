---
title: Contrastive and Self-supervised Learning
tags: [self-supervised, contrastive, representation-learning, deep-learning]
aliases: [SimCLR, MoCo, BYOL, NT-Xent, InfoNCE]
difficulty: 2
status: complete
related: [bayesian-inference, backpropagation, attention-mechanism]
depends_on: [backpropagation, loss-nt-xent, normalization-layers]
---

# Contrastive and Self-supervised Learning

---

## Fundamental

Labeling data is expensive. ImageNet has 1.2M labeled images — tiny compared to the billions of unlabeled images available. Self-supervised learning defines a pretext task using only the structure of the data, without human labels.

**Goal:** learn representations $z = f(x)$ that capture semantic content, such that a linear classifier trained on top of $z$ achieves high accuracy on downstream labeled tasks.

**The pretext task idea:** early self-supervised approaches used handcrafted pretext tasks — predict rotation angle, predict relative position of patches, colorize grayscale images. These tasks are too easy to "cheat": a model can solve rotation prediction using texture statistics, not object identity.

**Contrastive learning solution:** use image identity as the supervision signal. Two augmented views of the same image should have similar representations; views from different images should not.

**Framework:**
1. Apply two independent random augmentations: $x^+ = \text{aug}_1(x)$, $x^{++} = \text{aug}_2(x)$
2. Encode both: $z^+ = f(x^+)$, $z^{++} = f(x^{++})$
3. $(z^+, z^{++})$ is a **positive pair**; all other images in the batch provide **negative pairs**

**NT-Xent Loss (SimCLR):** Normalized Temperature-scaled Cross-Entropy. For a batch of $N$ images, create $2N$ augmented views. For a positive pair $(i, j)$:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}[k \neq i]\, \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $z_i, z_j$ = projected representations of the two augmented views of the same image (a positive pair), $\text{sim}(u,v) = \frac{u^\top v}{\|u\|\|v\|}$ = cosine similarity (dot product of unit vectors, ranges from -1 to 1), $\tau$ = temperature hyperparameter (smaller = sharper distribution, focuses more on hard negatives), $N$ = number of images in the batch, $2N$ = total augmented views, and $\mathbb{1}[k \neq i]$ = indicator that excludes $z_i$ itself from the denominator sum. The loss is minimized when $z_i$ and $z_j$ are more similar to each other than to all $2(N-1)$ other views. The denominator sums over $2(N-1)$ negatives.

**Why augmentation design is critical:** augmentations define what "invariance" means. Too weak: the pretext task is trivial (solved by low-level features). Too strong: positives look like negatives. SimCLR's critical combination: random resized crop + color jitter + Gaussian blur + grayscale. Random crop is most important — different crops share semantic content but differ spatially. Color jitter destroys the shortcut of using identical color statistics between crops.

**Good representation goal:** $z^+$ and $z^{++}$ close in representation space; $z^+$ and any $z^-$ far apart. This forces the encoder to capture information preserved across augmentations (semantic content) while discarding what varies (lighting, crop, color).

---

## Intermediate

**Temperature effect:** low $\tau$ (< 0.1) concentrates loss on the hardest negatives — more discriminative representations but unstable training. High $\tau$ (> 0.5) distributes loss uniformly — smoother optimization but weaker signal. Typical: $\tau = 0.07$–$0.1$.

**Connection to InfoNCE and [[entropy-mutual-info|mutual information]]:**

$$\mathcal{I}(z^+; z^{++}) \geq \log K - \mathcal{L}_{NCE}$$

where $\mathcal{I}(z^+; z^{++})$ = mutual information between the two view representations (what information they share), $K$ = number of negative examples, and $\mathcal{L}_{NCE}$ = the NT-Xent / InfoNCE loss. As $K$ grows, the bound $\log K$ grows, making the lower bound tighter. Maximizing $\mathcal{I}$ = minimizing $\mathcal{L}_{NCE}$. NT-Xent is a softmax classification loss where the "class" is the identity of the positive pair among all candidates.

**Why large batches are needed:** with few negatives (small batch), many negatives are "easy" — very dissimilar images where the model already knows they differ. With large batches (SimCLR uses 4096–8192), the pool contains diverse, hard negatives that provide strong learning signal. This requirement makes SimCLR expensive (32+ GPUs).

**MoCo (He et al., 2020) — momentum contrast:** maintains a queue of past-encoded negatives, decoupling batch size from negative count.

```
Initialize: queue Q of size K=65536
For each batch (x_q, x_k):
  1. Encode queries: z_q = f_θ(x_q)
  2. Encode keys: z_k = f_ξ(x_k)  [with momentum encoder]
  3. Loss: z_q against z_k (positive) and Q (negatives)
  4. Enqueue z_k, dequeue oldest
  5. Update: θ ← Adam(θ, grad)
             ξ ← m·ξ + (1-m)·θ   [m ≈ 0.999]
```

The momentum encoder prevents inconsistency: if $f_\xi$ updated by gradient, old keys in the queue would be computed by very different parameters → incomparable negatives. EMA keeps $f_\xi$ slowly changing → queue is internally consistent.

**BYOL (Grill et al., 2020) — no negatives needed:**

Two networks:
- **Online network:** encoder $f_\theta$ + projector $g_\theta$ + predictor $q_\theta$
- **Target network:** encoder $f_\xi$ + projector $g_\xi$ (no predictor, EMA-updated)

Loss:
$$L = 2 - 2 \cdot \frac{q_\theta(g_\theta(f_\theta(x_1)))\, \cdot\, g_\xi(f_\xi(x_2))}{\|q_\theta(g_\theta(f_\theta(x_1)))\| \cdot \|g_\xi(f_\xi(x_2))\|}$$

Three mechanisms prevent collapse: (1) **Asymmetry** — only online has the predictor, which must actively track the moving target; (2) **Stop-gradient** on target, breaking the trivial optimization path; (3) **EMA** — target moves slowly, preventing instant collapse; (4) **Batch normalization** — implicitly uses batch statistics as a contrastive signal.

**MAE (He et al., 2022) — masked autoencoders:**
1. Divide image into $14\times14$ patches (ViT-Large, patch size 16)
2. Randomly mask 75% of patches
3. Encoder (large ViT): processes only the 25% visible patches
4. Decoder (small transformer): receives encoded visible + learned mask tokens, predicts pixel values for masked patches
5. Loss: MSE on masked patches only

Why 75% masking? Natural image patches are highly redundant — adjacent patches are strongly correlated. At 25% masking, the model can reconstruct by interpolating from neighbors. At 75%, local interpolation fails; the model must understand the global semantic content.

| | Contrastive (SimCLR/MoCo) | MAE |
|--|---|---|
| Architecture | CNN or ViT | ViT (patch-based required) |
| Linear probe accuracy | High | Lower |
| Fine-tune accuracy | Good | State-of-the-art |
| Scales with model size | Moderate | Excellent |
| Compute | Moderate | Fast (encode 25% of patches) |

---

## Advanced

**The projection head as information filter:** all modern contrastive methods add a projection head $g$ on top of the encoder $f$: $z = g(f(x))$. Training uses $z$; evaluation uses $h = f(x)$ (without $g$). The projection head learns to absorb invariances required by the contrastive task but harmful for downstream tasks. For example, aggressive color jitter makes $z$ invariant to color — but color is useful for downstream classification. Without the projection head, the encoder itself must become invariant, losing useful information. The projection is trained into uselessness and discarded.

**Barlow Twins (Zbontar et al., 2021) — redundancy reduction:**

$$\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2$$

where $C$ is the cross-correlation matrix between embeddings of the two views. First term: diagonal = 1 (same feature in both views → maximally correlated). Second term: off-diagonal = 0 (different features should be decorrelated). This is Barlow's (1961) efficient coding hypothesis from neuroscience: remove redundancy between neurons. Unlike contrastive methods, Barlow Twins works with small batches and requires no negative pairs or asymmetry tricks.

**VICReg (Bardes et al., 2022):** three explicit terms:
- **Variance:** each feature dimension should have non-zero variance across the batch (prevent per-dimension collapse)
- **Invariance:** same images' representations should be similar
- **Covariance:** different dimensions should be uncorrelated

VICReg makes the collapse-prevention mechanism explicit rather than implicit, enabling theoretical analysis.

**DINO (Caron et al., 2021) — self-distillation with [[vision-transformer|ViT]]:**
- Student: processes small crops (local view)
- Teacher: EMA of student, processes large crops (global view)
- Loss: [[loss-cross-entropy|cross-entropy]] between student and teacher output distributions

DINO ViT features show emergent segmentation properties without any supervision — the `[CLS]` token's [[attention-mechanism|attention maps]] segment semantically coherent objects. This wasn't explicitly trained; it emerges from the self-distillation objective forcing the student to predict global semantics from local views.

**DINOv2 (Oquab et al., 2023):** scales DINO to large curated datasets (LVD-142M) with a combination of SSL objectives. Produces universal visual features used as frozen backbones for many downstream tasks. Key finding: at sufficient scale with curated data, SSL features match or exceed supervised ImageNet features for dense prediction tasks (depth estimation, segmentation), not just classification.

**Why does contrastive learning with augmentation work better than pretext tasks?** Information-theoretic argument: augmentation invariance forces the encoder to retain mutual information $I(z; y)$ between representation and class label $y$, while discarding augmentation-specific noise. Since augmentations are chosen to preserve class identity, the encoder is biased toward class-relevant features. Handcrafted pretext tasks (rotation prediction) don't have this property — a model can achieve low rotation loss using features that are irrelevant to class identity.

---

## Links

- [[backpropagation]] — contrastive learning is trained by backpropagating the InfoNCE / NT-Xent loss; momentum encoders (MoCo) use an EMA of the online encoder to generate stable negatives
- [[loss-nt-xent]] — NT-Xent (InfoNCE) is the contrastive loss: pulls together positive pairs, pushes apart negatives; temperature $\tau$ balances uniformity and alignment
- [[normalization-layers]] — batch normalization in SimCLR prevents collapse and provides implicit negatives via batch statistics; BYOL uses layer norm to avoid this information leakage
- [[self-supervised-overview]] — contrastive learning is the dominant paradigm in visual SSL; SimCLR, MoCo, CLIP are all contrastive methods that learn by comparing augmented views
- [[attention-mechanism]] — ViT encoders in CLIP/DINO use self-attention; contrastive learning provides the pretraining signal; the attention maps emerge without segmentation labels
- [[entropy-mutual-info]] — InfoNCE is a lower bound on mutual information between two views; maximizing InfoNCE maximizes the shared information across augmentations
- [[vision-transformer]] — contrastive SSL with ViT (DINOv2, CLIP) produces strong visual features; CNNs and ViTs both benefit from contrastive pretraining
