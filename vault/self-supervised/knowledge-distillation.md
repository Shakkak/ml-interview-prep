---
title: Knowledge Distillation & Teacher-Student Learning
tags: [knowledge-distillation, teacher-student, model-compression, efficiency, soft-targets]
aliases: [knowledge distillation, teacher student, soft targets, KD]
difficulty: 2
status: complete
related: [normalization-layers, backpropagation, loss-cross-entropy, lora-quantization, self-supervised-overview]
---

# Knowledge Distillation & Teacher-Student Learning

---

## Fundamental

A large pretrained **teacher** model $T$ transfers knowledge to a smaller **student** model $S$ by providing richer training signal than hard one-hot labels.

**Why not just train the student from scratch?** Hard labels carry almost no information beyond the correct class — every training example is either 1 or 0. The teacher's output distribution (e.g., "70% cat, 25% tiger, 5% dog") carries inter-class similarity structure that helps the student generalise. This similarity information encodes the teacher's learned world model.

**Response-based distillation (Hinton et al., 2015) — soft targets:**

Given teacher logits $z_T$ and student logits $z_S$, compute soft probability distributions using temperature $\tau > 1$:

$$p_i^T = \frac{\exp(z_i^T / \tau)}{\sum_j \exp(z_j^T / \tau)}, \qquad p_i^S = \frac{\exp(z_i^S / \tau)}{\sum_j \exp(z_j^S / \tau)}$$

Higher $\tau$ flattens the distribution, making small probabilities (inter-class similarities) more prominent.

**Distillation loss:**

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_{CE}(y, p^S) + \alpha\,\tau^2\,D_{KL}(p^T \| p^S)$$

- First term: standard cross-entropy against the hard label $y$
- Second term: KL divergence against soft teacher targets
- $\tau^2$ factor: compensates for gradient magnitude reduction from temperature scaling
- $\alpha \in [0,1]$: balance between hard and soft loss; typically $\alpha=0.9$

**Numerical example — temperature effect:**

Teacher logits: $z^T = [4.0, 1.0, 0.5]$ (classes: cat, tiger, bus).

At $\tau=1$: $p^T = \text{softmax}([4.0, 1.0, 0.5]) = [0.920, 0.046, 0.034]$ — nearly hard labels.

At $\tau=4$: $z/\tau = [1.0, 0.25, 0.125]$, $p^T_\tau = [0.486, 0.297, 0.217]$. Tiger probability rises from 4.6% to 29.7%. The student now learns "cats and tigers are similar" — structural knowledge hard labels cannot convey.

---

## Intermediate

**Feature-based distillation:** instead of matching output distributions, match intermediate representations.

**FitNets (Romero et al., 2015):** match intermediate feature maps. A thin regressor $r$ projects student features to teacher feature dimensions:

$$\mathcal{L}_{hint} = \| r(F^S) - F^T \|_F^2$$

Allows the teacher to be deeper than the student — the student learns to match internal representations, not just outputs.

**Attention Transfer (Zagoruyko & Komodakis, 2017):** match spatial attention maps derived from intermediate features:

$$A^T = \text{normalize}\!\left(\sum_c |F^T_c|^2\right), \quad \mathcal{L}_{AT} = \| A^T - A^S \|_2^2$$

Forces the student to look at the same spatial locations as the teacher.

**Relation-based distillation:** match relationships between samples rather than individual features:

$$\mathcal{L}_{RKD} = \sum_{(i,j)} l\left(d(z_i^S, z_j^S),\, d(z_i^T, z_j^T)\right)$$

Captures structural properties of the embedding space rather than absolute positions. Particularly useful when student and teacher have different architectures (different embedding dimensions).

**DeiT — distillation token (Touvron et al., 2021):**

Brings knowledge distillation to Vision Transformers, making them train-efficient without ImageNet-21k pretraining. One extra learnable **distillation token** is appended to the patch token sequence alongside `[CLS]`:

```
Input patches → [CLS] + patch tokens + [DIST]
                  ↓         ↓           ↓
             class head  (attention)  distill head
                  ↓                     ↓
              hard label CE        soft/hard teacher loss
```

The `[DIST]` token is trained to match the teacher's prediction (a strong CNN like RegNet). At inference, predictions from `[CLS]` and `[DIST]` heads are averaged.

Key insight: the distillation token attends to different patches than `[CLS]` — they learn complementary representations. DeiT-B with distillation matches ResNet-152 accuracy while training 3× faster.

**When to use which method:**

| Scenario | Method |
|---|---|
| Compress large classifier to small one | Response distillation (Hinton) |
| Student must mimic teacher's internal representations | Feature distillation (FitNets) |
| Self-supervised pretraining | Self-distillation (DINO) |
| Train ViT without massive data | DeiT distillation token |
| Cross-modality knowledge transfer | Cross-modal distillation |
| Ensemble compression | Born-Again Networks |

---

## Advanced

**Self-distillation (Born-Again Networks, Furlanello et al., 2018):** distill from the same architecture — the student has identical capacity to the teacher. Counterintuitively, this improves performance. The student trained on soft targets from the first-generation model produces a better second-generation model, and averaging an ensemble of born-again networks outperforms the original. The explanation: the soft targets act as implicit data augmentation, smoothing the loss landscape and preventing overfitting to label noise.

**DINO (Caron et al., 2021) — self-supervised self-distillation:**

```
Online ("student"): updated by gradient descent
Momentum ("teacher"): EMA of student weights, processes larger crops

Loss: cross-entropy between student and teacher output distributions
      (on different crops of the same image)
      + stop-gradient on teacher + centering (subtract running mean)
```

DINO ViT features learn emergent segmentation properties — attention maps of `[CLS]` tokens attend to semantically coherent regions without any supervision. The centering operation (subtract running mean from teacher output before softmax) prevents mode collapse where the teacher always outputs the same token.

**Why does distillation work beyond just soft labels?** Dark knowledge hypothesis: the teacher encodes structured information about the relationship between incorrect classes. For a model trained on CIFAR-100, the probabilities assigned to wrong classes encode the hierarchical structure of categories (animals vs vehicles vs household objects). A student learning from these distributions implicitly learns this hierarchy even without explicit hierarchical labels.

**Contrastive Representation Distillation (CRD, Tian et al., 2020):** combines contrastive learning with feature distillation. Instead of matching individual features, student representations are contrasted against teacher representations as positives and random samples as negatives:

$$\mathcal{L}_{CRD} = -\log\frac{\exp(z_S \cdot z_T^+ / \tau)}{\exp(z_S \cdot z_T^+ / \tau) + \sum_k \exp(z_S \cdot z_T^{k-} / \tau)}$$

This transfers the structure of the teacher's representation space (not just individual activation magnitudes). CRD significantly outperforms FitNets on cross-architecture distillation where feature dimensions differ.

**Implicit distillation in large language models:** model merging (SLERP, TIES, DARE) can be interpreted as knowledge distillation without a teacher forward pass. By averaging parameters from multiple fine-tuned variants of the same base model, the merged model implicitly combines the knowledge each variant acquired during fine-tuning. This connection to distillation helps explain why model merging works surprisingly well despite its conceptual simplicity.

---

*See also: [[lora-quantization]] · [[loss-cross-entropy]] · [[self-supervised-overview]] · [[contrastive-learning]]*
