---
title: Self-supervised Learning — Unified Overview
tags: [self-supervised, contrastive, masked-prediction, representation-learning]
aliases: [SSL, pretraining, SimCLR, BYOL, MAE, DINO]
difficulty: 3
status: complete
related: [contrastive-learning, attention-mechanism, bayesian-inference]
depends_on: [backpropagation, distributions-overview]
---

# Self-supervised Learning — Unified Overview

---

## Fundamental

All self-supervised methods define a pretext task that forces the encoder to learn meaningful representations without labels. The three families differ in how they define "what makes two things similar":

| Family | Core Idea | Examples |
|--------|-----------|---------|
| **Contrastive** | Push positive pairs together, negative pairs apart | SimCLR, MoCo, CLIP |
| **Non-contrastive** | Push positive pairs together, prevent collapse | BYOL, SimSiam, Barlow Twins |
| **Masked Prediction** | Predict masked parts of the input | MAE, BEiT, [[bert-mlm\|BERT (text)]] |

**Why self-supervised learning works:** pretext tasks are designed so that solving them requires capturing semantic content. Two augmented views of the same image must share the object identity — an encoder that maps both to similar representations must have learned what the object is. The key insight is that the pretext task implicitly defines the invariances the representation should have.

**The projection head principle:** all modern [[contrastive-learning|contrastive]] and non-contrastive methods add a projection head $g$ on top of the encoder $f$: $z = g(f(x))$. Training minimizes loss in $z$-space; evaluation uses $h = f(x)$ (without $g$). The projection head absorbs augmentation-specific invariances that would be harmful for downstream tasks. The encoder retains richer information because it doesn't need to be augmentation-invariant itself.

**Practical selection guide:**

```
Do you have labeled data?
  └── No → Self-supervised pretraining
        ├── Using ViT? Large model?
        │     └── Yes → MAE / DINOv2
        ├── Need strong linear probe?
        │     └── Yes → SimCLR / MoCo / DINO
        ├── Limited GPU memory / small batch size?
        │     └── Yes → BYOL / SimSiam / Barlow Twins
        └── Multi-modal (text+image)?
              └── [[clip\|CLIP]] (contrastive across modalities)
```

---

## Intermediate

**Contrastive methods — information-theoretic view:**

All contrastive methods maximize a lower bound on mutual information between two views:

$$\mathcal{I}(v_1; v_2) \geq \mathcal{L}_{NCE} = \mathbb{E}\left[\log \frac{e^{f(v_1)^T f(v_2)/\tau}}{\frac{1}{K}\sum_{k=1}^K e^{f(v_1)^T f(v_k^-)/\tau}}\right]$$

where $v_1, v_2$ = two augmented views of the same image, $f(\cdot)$ = encoder, $\tau$ = temperature, $K$ = number of negative examples, $v_k^-$ = negative views (from different images), and $\mathcal{I}(v_1; v_2)$ = mutual information between the two view representations (how much they share).

Maximizing mutual information $\mathcal{I}(v_1; v_2)$ forces the encoder to capture shared information between views — the semantic content that survives augmentation. The **information bottleneck** principle applied via data augmentation: by choosing augmentations that preserve class identity but destroy irrelevant variation, the model learns to retain class-relevant information.

**Non-contrastive methods — preventing collapse:**

Without negative pairs, the network might collapse to $f(x) = c$ (constant) for all $x$ — trivial solution.

BYOL prevents this via: asymmetry (online network has predictor, target doesn't) + stop-gradient on target + EMA (target moves slowly).

Barlow Twins takes a different approach: minimize redundancy between representation dimensions:

$$\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2$$

where $C$ is the cross-correlation matrix between embeddings of two views. Diagonal = 1 (same feature maximally correlated across views); off-diagonal = 0 (different features decorrelated). Neuroscientific motivation: Barlow's (1961) efficient coding hypothesis.

**Masked prediction — denoising autoencoder perspective:**

Masked prediction is a special case of denoising autoencoders with a specific corruption: replacing tokens with a mask token.

[[bert-mlm|BERT]]'s masking (15% of text tokens): forces learning of language statistics necessary for prediction. MAE's masking (75% of image patches): forces learning of image statistics (object structure, spatial relationships) necessary for reconstruction.

**Why images need higher masking ratios than text:** text tokens are semantically dense — each word carries meaning. Masking 15% provides significant context removal. Image patches are spatially redundant — adjacent patches are highly correlated. Masking 15% of patches can be solved by local interpolation without understanding content. Need 75% to make local interpolation insufficient.

**Pixel vs feature prediction targets:**

- **MAE:** predicts raw pixel values of masked patches. Simple target, learns low-level and mid-level features.
- **BEiT (Bao et al., 2022):** predicts discrete visual tokens from a learned codebook (VQ-VAE). More abstract prediction target, learns more semantic features but requires a pre-trained tokenizer.
- **Data2Vec (Baevski et al., 2022):** predicts the top-$K$ layer representations of the full unmasked input. Self-distillation style: predict abstract features, not raw pixels. Works across modalities (vision, text, audio) with the same framework.

**Unified comparison:**

```
                    Contrastive          Non-contrastive       Masked Prediction
                    ───────────          ───────────────       ─────────────────
Negative pairs:     Required             Not needed            Not needed
Batch size:         Large (SimCLR)       Small ok (BYOL)       Small ok
Architecture:       CNN or ViT           CNN or ViT            ViT (patch needed)
Linear probe:       ★★★★★               ★★★★☆               ★★★☆☆
Fine-tune:          ★★★★☆               ★★★★☆               ★★★★★
Scale with size:    ★★★☆☆               ★★★☆☆               ★★★★★
Compute:            Medium               Medium                Low (25% encoding)
```

---

## Advanced

**DINO — emergent properties from self-distillation:**

DINO (Caron et al., 2021) trains a student [[vision-transformer|ViT]] on small crops to predict the teacher ViT's output on large crops (teacher = EMA of student). The emergent property: DINO ViT representations without any supervision learn to segment objects — the `[CLS]` [[attention-mechanism|attention maps]] attend to semantically coherent regions. This wasn't explicitly trained; it arises because the self-distillation objective forces the student to predict global (large-crop) semantics from local (small-crop) features, requiring understanding of object identity.

DINOv2 scales this to large datasets with curated training and produces excellent general-purpose visual features used as frozen backbones. Key finding: at sufficient scale with curated data, SSL features match or exceed supervised features for dense prediction tasks (depth, segmentation), not just classification.

**The representation learning landscape — what each method actually captures:**

Contrastive methods trained with aggressive augmentation produce highly augmentation-invariant features — excellent for classification where color, pose, and crop don't change the label. But this invariance is a liability for tasks that need those features (e.g., counting requires position sensitivity; medical imaging requires color sensitivity).

MAE representations are richer in low-level spatial information — the reconstruction objective requires capturing fine-grained details. This explains why MAE features are better for fine-tuning (all information is preserved; fine-tuning can select what to use) but worse for linear probing (the semantic signal is mixed with spatial noise).

**Theoretical connection between SSL and supervised learning:**

Saunshi et al. (2022) prove that minimizing contrastive loss on augmented views upper-bounds the loss of a linear classifier on the true labels, under the assumption that augmentations preserve labels. Formally: if $P(\text{same class} | \text{same image}) = 1$, then

$$\mathcal{L}_{\text{contrastive}} \geq \frac{1}{\tau} \cdot \mathcal{L}_{\text{downstream}}$$

This provides the first theoretical justification for why contrastive SSL representations transfer to supervised downstream tasks.

**The collapse taxonomy:** different methods prevent collapse through different mechanisms:

| Method | Collapse prevention mechanism |
|--------|-------------------------------|
| SimCLR | Explicit negatives in loss |
| MoCo | Negatives from queue |
| BYOL | Predictor asymmetry + stop-gradient |
| SimSiam | Stop-gradient only (no EMA) |
| Barlow Twins | Cross-correlation decorrelation |
| VICReg | Explicit variance regularization |
| DINO | Centering (subtract teacher mean) + momentum |

**Foundation models and SSL at scale:** large language models (GPT, Llama) are autoregressive self-supervised models trained on next-token prediction — a form of masked prediction with 100% masking of future context. BERT's MLM and MAE's masked patches are the same principle applied to different modalities. The unification: SSL learns to compress input distributions into representations that enable accurate prediction of held-out information, whether that information is future tokens, masked patches, or mismatched captions.

---

## Links

- [[backpropagation]] — all SSL methods are trained by backpropagation; the key difference is the pretext task that defines the self-supervised loss function
- [[distributions-overview]] — SSL methods implicitly model the data distribution: contrastive methods learn a metric, generative methods learn the density; the choice shapes downstream representations
- [[contrastive-learning]] — contrastive SSL (SimCLR, MoCo, CLIP) pulls positive pairs together and pushes negatives apart; it learns an embedding metric without explicit density modeling
- [[bert-mlm]] — masked prediction SSL (BERT, MAE) predicts masked tokens/patches from context; it learns bidirectional contextual representations
- [[knowledge-distillation]] — self-distillation (BYOL, DINO) is a hybrid: the teacher is the student's own EMA; no negatives, no explicit reconstruction — just match your past self
- [[clip]] — multimodal SSL (CLIP) aligns representations across modalities; it learns a shared embedding space for images and text using paired data
- [[vision-transformer]] — ViT is the dominant backbone for visual SSL; its patch tokenization enables both contrastive (DINO) and reconstruction (MAE) pretraining without CNN inductive biases
