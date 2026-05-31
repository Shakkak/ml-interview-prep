---
title: BLIP and BLIP-2 (Bootstrapping Language-Image Pretraining)
tags: [blip, vision-language, multimodal, image-captioning, visual-qa, zero-shot]
aliases: [BLIP, BLIP-2, bootstrapping language-image pretraining, Q-Former]
difficulty: 2
status: complete
related: [clip, attention-mechanism, self-supervised-overview, contrastive-learning]
---

# BLIP / BLIP-2 — Bootstrapping Language-Image Pretraining

---

## Fundamental

CLIP excels at retrieval and zero-shot classification but has two fundamental gaps:
1. **No text generation.** CLIP is a dual encoder — it can score image-text pairs but cannot generate captions or answers.
2. **Noisy web data.** Internet image-text pairs often have misaligned or irrelevant captions. CLIP learns to tolerate this noise.

BLIP (Li et al., Salesforce 2022) addresses both with a unified architecture and a data bootstrapping strategy.

**BLIP architecture — three components, one model:**

BLIP uses a single model with three operating modes depending on which components are activated:

```
Image → Visual Encoder (ViT)
           ↓
    ┌──────────────────────────────────────┐
    │ Image-Text Contrastive (ITC)         │  ← CLIP-like alignment
    │ Image-Text Matching     (ITM)        │  ← binary: does image match text?
    │ Language Modelling      (LM)         │  ← caption/answer generation
    └──────────────────────────────────────┘
```

**ITC (contrastive):** aligns image and text embeddings — same InfoNCE loss as CLIP. Pulls matched pairs together in embedding space.

**ITM (matching):** a binary classifier that takes fused image+text features and predicts whether the pair is genuinely matched. Uses hard negative mining: the closest non-matching pair is used as the negative.

**LM (generation):** a causal language model decoder that generates text conditioned on the image. Enables captioning and visual question answering.

The text encoder/decoder shares weights — different attention masking is applied for each task (bidirectional for understanding, causal for generation).

**Comparison with CLIP:**

| Property | CLIP | BLIP |
|---|---|---|
| Text generation | ✗ | ✓ (captioning, VQA) |
| Training data | 400M web pairs | Web + CapFilt cleaning |
| Architecture | Dual encoder | Unified enc-dec |
| Zero-shot classification | ✓ excellent | ✓ good |
| Fine-grained VQA | ✗ | ✓ |
| Noisy data handling | Learns to tolerate | Filters and replaces |

---

## Intermediate

**CapFilt — bootstrapping clean data:**

The key BLIP innovation is cleaning noisy web data using the model itself:

```
Step 1 — Captioner: fine-tune LM component on clean COCO captions
         → generates synthetic captions for web images

Step 2 — Filter:    fine-tune ITM component on clean COCO captions
         → removes noisy web captions AND poor synthetic captions

Step 3 — Retrain BLIP on the cleaned dataset
```

Why this works: synthetic captions are image-grounded (generated from the actual image); the filter removes both original noisy captions and bad synthetic ones. The result is a cleaner, higher-quality training set from the same web data. This bootstrapping loop consistently improves performance over training on raw web data — even with a smaller cleaned dataset.

**Hard negative mining in ITM:** pairing each image with the text that scores highest in ITC (but is wrong) makes the discriminator task harder and representations more precise. Standard random negatives would be too easy to distinguish.

**BLIP-2 — efficient multimodal LLM alignment:**

BLIP-2 (Li et al., Salesforce 2023) addresses a new problem: how to connect a frozen vision encoder and a frozen LLM without expensive end-to-end training.

The **Q-Former (Querying Transformer):** a lightweight transformer bridging the frozen image encoder and frozen LLM:

```
Image → Frozen ViT → Q-Former → Frozen LLM
```

The Q-Former has $N = 32$ learnable **query tokens** that interact with image features via cross-attention. These 32 tokens extract the most task-relevant visual information and pass it to the LLM as soft visual prompts. Only the Q-Former is trained.

**Why freeze everything?** Training a 7B+ LLM end-to-end with image data is prohibitively expensive and often catastrophically forgets language capabilities. Freezing preserves both encoders and forces the Q-Former to learn the alignment — a tractable, parameter-efficient objective.

**Two-stage Q-Former training:**

Stage 1 — Vision-language representation learning (frozen ViT, trainable Q-Former):
- ITC: align query tokens with text
- ITM: match/mismatch classification
- Image-grounded text generation (ITG): generate caption from query tokens

Stage 2 — Vision-to-language generative learning (frozen ViT + frozen LLM, trainable Q-Former):
- Query tokens → linear projection → fed as prefix to LLM
- LLM learns to read visual tokens like soft prompts
- LLM's language capabilities remain intact

**32 query tokens:** enough to capture image semantics but far fewer than full ViT patch tokens (~196 for ViT-B/16 on 224×224). This compresses visual information to what language generation needs, reducing the cross-modal alignment burden.

---

## Advanced

**Information bottleneck perspective on Q-Former:** the 32 query tokens act as a learned bottleneck between $\sim$196 image patch features and the LLM. The Q-Former is forced to extract the most language-relevant visual information, discarding visual details irrelevant to language generation. This is analogous to the projection head in contrastive learning — the bottleneck protects the frozen LLM from low-level visual noise.

**Shared weights between encoder and decoder in BLIP:** the text encoder (bidirectional attention) and decoder (causal attention) share all transformer parameters. Only the attention mask differs. This is parameter-efficient but creates a subtle training challenge: the same parameters must work for both attention patterns simultaneously. The model converges because bidirectional and causal modes access different attention score regions — bidirectional attends future tokens, causal does not. This weight-sharing trick was also used in UNITER and ViLBERT but BLIP systematizes it across three objectives.

**CapFilt as self-training with distribution control:** CapFilt is an instance of iterative pseudo-labeling, but with an important difference from naive self-training. Standard self-training uses the model's own high-confidence predictions as pseudo-labels, risking confirmation bias (wrong confident predictions reinforce errors). CapFilt avoids this by: (1) fine-tuning the captioner on clean COCO captions (not web data), ensuring the synthetic captions follow a clean distribution; (2) using ITM as a separate filtering signal rather than raw caption confidence. This two-stage filtering breaks the self-reinforcing error loop.

**InstructBLIP (Dai et al., 2023):** extends BLIP-2 by instruction-tuning the Q-Former. Rather than training the Q-Former on fixed image-caption tasks, InstructBLIP trains it on 26 diverse vision-language datasets formulated as instructions. The Q-Former is taught to extract task-relevant features conditional on the instruction — different queries for "describe the scene" vs "count the objects" vs "translate the text." This instruction-aware feature extraction substantially improves zero-shot generalization compared to BLIP-2.

**Why BLIP outperforms CLIP on generation but CLIP remains competitive on retrieval:** CLIP's dual-encoder architecture allows $O(1)$ retrieval — image and text embeddings are precomputed and similarity is a dot product. BLIP's cross-encoder ITM head requires a forward pass for every candidate pair — $O(n)$ for a corpus of $n$ texts. BLIP uses ITC for coarse retrieval (same $O(1)$ as CLIP) then re-ranks with ITM. This cascade design trades retrieval speed for precision, particularly on compositional queries where CLIP's independent embeddings miss cross-modal interactions.

---

*See also: [[clip]] · [[attention-mechanism]] · [[self-supervised-overview]] · [[contrastive-learning]]*
