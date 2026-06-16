---
title: "Vision-Language Model Architectures"
tags: [vlm, llava, flamingo, instructblip, multimodal, vision-language, dense-prediction, visual-instruction-tuning]
aliases: [VLM Architectures, Large Vision-Language Models, Multimodal LLMs, LLaVA, Flamingo, BLIP-2, InstructBLIP, Q-Former]
difficulty: 3
status: complete
depends_on: [vision-transformer, clip, attention-mechanism]
related: [open-vocabulary-detection, lora-quantization, activation-gelu-swish, cross-attention]
---

# Vision-Language Model Architectures

---

## Fundamental

**The problem:** a language model can reason about text but cannot see images; a vision encoder can extract image features but cannot generate text. Vision-Language Models (VLMs) bridge these two worlds so the system can answer questions about images, describe scenes, and reason across modalities.

**Intuition:** the core challenge is the *bridge* — visual patch tokens live in a different embedding space than text tokens. If you can map visual tokens into the LLM's token space, the LLM's existing language reasoning works on them unchanged. The question is how much expressiveness the bridge needs.

Every modern VLM shares three components:

```
Image
  ↓
[Visual Encoder]     ← extracts image features (ViT, CLIP encoder)
  ↓
[Bridge / Connector] ← maps visual tokens to LLM's embedding space
  ↓
[Language Model]     ← generates text autoregressively
  ↓
Text output
```

### LLaVA: Minimal Bridge

**LLaVA (Liu et al., 2023)** uses the simplest possible bridge — a single linear projection:

```
Image → CLIP ViT-L encoder → Z_v ∈ ℝ^{N×D_v}
  → Linear W ∈ ℝ^{D_v × D_l} → H_v ∈ ℝ^{N×D_l}
  → Concatenate with text tokens → LLaMA decoder → answer
```

where $Z_v$ = visual patch tokens output by CLIP (e.g., $256 \times 1024$), $W$ = the only new parameter, $D_l$ = LLM embedding dimension (e.g., 4096), and $H_v$ = projected visual tokens now in the LLM's embedding space.

**Why it works:** CLIP already aligned image patches with language concepts during pretraining. The linear projection just needs to rescale the space — a surprisingly small job. LLaVA-1.5 upgrades to a 2-layer MLP with [[activation-gelu-swish|GELU]] and higher resolution (336px), adding ~4 points on benchmarks.

**Training data insight:** LLaVA uses GPT-4 to generate 150K instruction-following (image, question, answer) examples from existing image captions — no human annotation. Synthetic instruction data is what enables open-ended question answering.

---

## Intermediate

### Flamingo: Few-Shot with Gated Cross-Attention

**Flamingo (Alayrac et al., 2022, DeepMind)** targets few-shot learning — given a handful of image-text examples, answer questions about a new image.

**Perceiver Resampler:** the visual encoder outputs variable-length patch tokens (e.g., 400 tokens for different image sizes). Flamingo maps these to a fixed 64 learned query tokens via [[cross-attention|cross-attention]]:

```
Visual features (variable length) → Cross-attention with 64 learned queries → 64 fixed visual tokens
```

**Gated cross-attention layers:** inserted at fixed intervals in the frozen LLM. Text tokens attend to visual tokens, with the output scaled by a $\tanh$ gate initialized to zero — so at training start, Flamingo outputs exactly what the frozen LLM would. The gate opens gradually, enabling stable fine-tuning.

**Interleaved image-text:** Flamingo processes arbitrary sequences of images and text — the same architecture handles image captioning, VQA, and few-shot visual reasoning with the same weights.

### BLIP-2 and Q-Former: Learned Query Extraction

**BLIP-2 (Li et al., 2023)** introduces the Q-Former — a more powerful bridge than LLaVA's linear projection:

```
Image → Frozen CLIP encoder → visual features F_v
         ↑
Q-Former: 32 learnable query tokens
  - Self-attention among queries
  - Cross-attention: queries attend to F_v
  - Text self-attention (for language-conditioned querying)
         ↓
32 query tokens → linear projection → LLM input
```

The Q-Former compresses $N$ patch tokens into just 32 tokens, extracting the most task-relevant information. Efficient, but can lose fine-grained spatial detail needed for localization tasks.

**Two-stage training:** Stage 1 trains the Q-Former against the frozen visual encoder (ITC, ITM, ITG objectives). Stage 2 plugs the Q-Former into a frozen LLM and trains the whole bridge on language modeling. The visual encoder and LLM are never co-trained.

**InstructBLIP** extends BLIP-2 by feeding the instruction text into the Q-Former during fine-tuning — so the visual features extracted are instruction-conditioned. "What color is the flower?" elicits different features from the same image than "What species of plant is this?"

---

## Advanced

### VLMs for Dense Prediction

Standard VLMs pool spatial information into patch tokens (16×16 pixels per token at 224px). Dense prediction (segmentation, depth, detection) needs per-pixel outputs — a fundamental mismatch.

**Approaches:**
1. **Prompt-and-parse:** ask the VLM to output bounding box coordinates as text tokens. Works for simple detection; fails for pixel-level segmentation.
2. **High-resolution tiling:** LLaVA-HR encodes the image in overlapping tiles at 448–672px, providing more spatial tokens.
3. **Hybrid architecture:** use a dense backbone (FPN, U-Net) for spatial features + VLM for semantic reasoning. Used in PixelLM, LISA.
4. **SAM + VLM pipeline:** SAM generates masks → VLM classifies each mask. Best for semantic segmentation without re-training.

### Adapting VLMs to Domain-Specific Imagery

Overhead imagery (drone, satellite) differs from natural photos: no color in SAR, non-RGB channels in multispectral, top-down perspective, and much smaller objects.

| Adaptation | Cost | Risk | When to use |
|---|---|---|---|
| Zero-shot (frozen CLIP) | None | Poor fine-grained accuracy | Coarse categories |
| Fine-tune bridge only | Low | Limited capacity | <10K images |
| [[lora-quantization\|LoRA]] on visual encoder + LLM | Medium | Slight forgetting | 10K–100K images |
| Full fine-tuning | High | Catastrophic forgetting | >100K images + data mixing |

**Key warning:** full fine-tuning of a CLIP encoder on domain-specific data rapidly degrades zero-shot generalization. If you need both domain accuracy and zero-shot capability, fine-tune only the projection or use LoRA adapters — keep the visual encoder frozen unless your dataset is very large.

### LLaVA-1.5 Forward Pass (Step by Step)

Input: 336×336 image + text question.

1. Split into 14×14 patches of 24×24 pixels → 576 patches; each → 1024-dim CLIP token
2. Apply 2-layer MLP (1024 → 4096, GELU, 4096 → 4096) → 576 × 4096 visual tokens
3. Concatenate: `[BOS] + [576 visual tokens] + [text tokens]`
4. LLaMA causal self-attention over the full sequence — text tokens attend to visual tokens via the unified token sequence (no separate cross-attention needed)
5. Autoregressive decoding of answer

Visual tokens occupy 576 of a 4096-token context window — a significant fraction. This is why higher-resolution VLMs face a context length problem.

| Model | Bridge | Visual Tokens | Strength |
|-------|--------|:---:|------|
| LLaVA | Linear projection | 256 | Simple, effective |
| LLaVA-1.5 | 2-layer MLP | 576 | State-of-art simple VLM |
| Flamingo | Perceiver + cross-attn | 64 | Few-shot, interleaved |
| BLIP-2 | Q-Former | 32 | Efficient, strong zero-shot |
| InstructBLIP | Q-Former (instruction-conditioned) | 32 | Task-adaptive visual features |

---

## Links

- [[vision-transformer]] — the image encoder in most VLMs is a pretrained ViT (CLIP ViT-L); patch tokens are extracted and fed to the LLM
- [[clip]] — CLIP-pretrained image encoders are the standard VLM visual backbone; CLIP's shared image-text space enables zero-shot classification
- [[attention-mechanism]] — cross-attention VLMs (Flamingo) inject image features via cross-attention layers inserted into the frozen LLM
- [[cross-attention]] — Flamingo's Perceiver Resampler and BLIP-2's Q-Former both use cross-attention to map variable-length visual features to fixed-length query tokens
- [[open-vocabulary-detection]] — VLMs extended to spatial tasks combine the VLM semantic reasoning with detection-specific localization heads
- [[lora-quantization]] — VLMs are fine-tuned efficiently with LoRA; QLoRA enables training 13B–34B VLMs on single consumer GPUs
- [[activation-gelu-swish]] — the FFN sublayers inside both the ViT encoder and LLM decoder use GELU or SiLU; LLaVA-1.5's MLP bridge also uses GELU
