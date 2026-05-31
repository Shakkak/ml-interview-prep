---
title: CLIP (Contrastive Language-Image Pretraining)
tags: [clip, vision-language, contrastive-learning, multimodal, zero-shot]
aliases: [CLIP, contrastive language-image pretraining, OpenAI CLIP]
difficulty: 2
status: complete
related: [contrastive-learning, attention-mechanism, self-supervised-overview, generative-adversarial-networks]
---

# CLIP — Contrastive Language-Image Pretraining

---

## Fundamental

CLIP (Radford et al., OpenAI 2021) learns visual representations by predicting which caption goes with which image from a dataset of 400M image-text pairs scraped from the internet. There are **no manually labelled categories** — supervision comes entirely from natural language.

**Dual-encoder architecture:**

```
Image x  →  Image Encoder (ViT or ResNet)  →  image embedding e_I  (L2-normalised)
Text  t  →  Text  Encoder (Transformer)   →  text  embedding e_T  (L2-normalised)
```

Similarity is cosine similarity: $s(x,t) = e_I \cdot e_T$. A learnable temperature parameter $\tau$ scales similarities before softmax (initialised to $1/0.07 \approx 14.3$).

**Training objective — symmetric InfoNCE:**

Given a batch of $N$ (image, text) pairs, construct an $N \times N$ similarity matrix. Diagonal entries are correct pairs; off-diagonal entries are negatives.

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^N \left[\log\frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)} + \log\frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)}\right]$$

The first term is the image-to-text direction; the second is text-to-image. The loss is averaged across both directions.

**Zero-shot classification:**

```
1. For each class c, create a text prompt: "a photo of a {class}"
2. Encode all prompts → text embeddings {e_T^c}
3. Encode test image → image embedding e_I
4. Predict: argmax_c (e_I · e_T^c)
```

The same image features that let the model match a caption also let it classify images zero-shot by comparing to text labels. No retraining or fine-tuning needed.

**Worked numerical example — similarity computation:**

Batch of $N=3$ pairs. Embeddings (2D, already normalised):

| i | $e_I^i$ | $e_T^i$ |
|---|---------|---------|
| 1 | [1, 0] | [0.9, 0.44] |
| 2 | [0, 1] | [0.1, 0.99] |
| 3 | [0.71, 0.71] | [0.71, 0.71] |

Similarity matrix $S_{ij} = e_I^i \cdot e_T^j$, $\tau=1$:

$$S = \begin{bmatrix} 0.90 & 0.10 & 0.71 \\ 0.44 & 0.99 & 0.71 \\ 0.88 & 0.78 & 1.00 \end{bmatrix}$$

Row 1 image-to-text loss: $-\log\frac{e^{0.90}}{e^{0.90}+e^{0.10}+e^{0.71}} = -\log(0.44) \approx 0.82$.

---

## Intermediate

**Why large batch size is critical:** with $N=32768$, each correct pair competes against 32767 negatives. The model must learn very precise representations to succeed. Small batches mean easy negatives — images that are clearly dissimilar. Large batches include hard negatives that share similar low-level features but different semantics, providing strong learning signal. CLIP used 32,768 pairs per batch across 256 GPUs.

**Prompt engineering matters:** "a photo of a {class}" works better than just "{class}" because it matches the style of training captions. Context matters for the text encoder.

**Prompt ensembling:** average embeddings from multiple prompts ("a photo of a {class}", "a blurry photo of a {class}", "a photo of many {class}s") to reduce sensitivity to any single phrasing. OpenAI found this consistently improves zero-shot performance.

**Temperature $\tau$ effect:** small $\tau$ (sharp softmax) concentrates the loss on the hardest negatives — faster convergence but unstable training. Large $\tau$ (soft softmax) spreads the loss across all negatives — more stable but weaker signal. CLIP learns $\tau$ as a parameter, starting at $\approx 0.07$.

**Strengths and weaknesses:**

Strengths:
- Zero-shot transfer to new datasets without retraining
- Robust to distribution shift (trained on internet diversity)
- Flexible: same model for classification, retrieval, semantic similarity
- Scales well: larger models + more data → better transfer

Weaknesses:
- Weak on fine-grained tasks: counting, spatial relationships, OCR
- Expensive to train from scratch (~400M pairs, large batch, hundreds of GPUs)
- Retrieval is symmetric — doesn't model compositionality well (e.g., "a red ball on a blue box" vs "a blue ball on a red box" are hard to distinguish)
- Biases in internet-scraped training data transfer to representations

**Comparison with BLIP and BLIP-2:**

| Property | CLIP | BLIP | BLIP-2 |
|---|---|---|---|
| Text generation | ✗ | ✓ | ✓ via LLM |
| Training cost | Very high | High | Low (Q-Former only) |
| Retrieval speed | $O(1)$ dot product | Cascade (ITC + ITM) | Cascade |
| VQA quality | ✗ | ✓ | ✓ strong |

---

## Advanced

**InfoNCE as a mutual information lower bound:**

$$\mathcal{L}_{\text{InfoNCE}} \ge \log N - I(X;T)$$

Minimising the InfoNCE loss maximises a lower bound on mutual information $I(X;T)$ between images and text. CLIP learns representations that capture all information shared between modalities — which is exactly semantic content. However, the bound is loose: maximising $I(X;T)$ can also be achieved by encoding nuisance information shared between image and caption (e.g., photographic style), not just semantics.

**The compositionality failure:** CLIP embeddings are order-insensitive to a surprising degree. Thrush et al. (2022) showed CLIP confuses "dog biting man" with "man biting dog" — both produce similar cosine similarities because the embeddings mix the token representations into a global average that lacks relational structure. This is a fundamental limitation of bag-of-words-style embedding approaches, not a data issue.

**ALIGN (Jia et al., Google 2021):** scales CLIP's approach to 1.8 billion noisy image-text pairs with minimal filtering. Finding: scale compensates for noise. A simple dual-encoder on noisy but enormous data matches or beats CLIP trained on 400M more carefully filtered pairs. This challenges the assumption that data quality dominates quantity at scale.

**SigLIP (Zhai et al., Google 2023):** replaces the softmax InfoNCE loss with a sigmoid binary cross-entropy loss applied independently to each pair:

$$\mathcal{L} = -\sum_{i,j} \left[y_{ij} \log \sigma(s_{ij}/\tau + b) + (1-y_{ij}) \log(1 - \sigma(s_{ij}/\tau + b))\right]$$

where $y_{ij} = 1$ iff pair $(i,j)$ is matched. Unlike InfoNCE, this does not require a softmax over all negatives in the batch — each pair is treated independently. This removes the large-batch requirement (works with batch 256) while improving performance. The key difference: InfoNCE competes each positive against all batch negatives; SigLIP treats every pair as an independent binary classification.

**EVA-CLIP (Fang et al., 2023):** scales to 18B parameters by combining CLIP's contrastive objective with masked image modeling as a secondary pretraining task. The dual objective forces the encoder to simultaneously learn discriminative cross-modal features (from contrastive) and dense reconstructive features (from masking), producing representations that excel at both retrieval and dense prediction tasks.

**Why CLIP features transfer to detection and segmentation better than supervised ImageNet features:** CLIP's text supervision creates semantically meaningful feature spaces that align with category boundaries humans care about. ImageNet supervision creates feature spaces optimized for 1000 fixed categories. When transferred to detection tasks with different category sets, CLIP features generalize better because they encode concept-level semantics from natural language descriptions rather than fixed category indices.

---

*See also: [[contrastive-learning]] · [[attention-mechanism]] · [[self-supervised-overview]] · [[blip]]*
