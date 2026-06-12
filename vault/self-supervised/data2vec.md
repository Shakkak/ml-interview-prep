---
title: data2vec
tags: [data2vec, multimodal-ssl, masked-prediction, latent-targets, self-supervised]
aliases: [data2vec, unified self-supervised learning, latent target prediction]
difficulty: 2
status: complete
related: [masked-autoencoders, bert-mlm, byol-simsiam, self-supervised-overview, vision-transformer]
---

# data2vec

---

## Fundamental

### Unified Self-Supervised Learning

**data2vec (Baevski et al., 2022)** proposes a single framework for self-supervised learning across modalities (text, vision, speech/audio) using the same fundamental approach:

**Core idea:** predict **latent representations** of the full input from a **partially masked input**, rather than predicting raw targets (pixels, tokens, waveforms).

This contrasts with:
- MAE: predicts raw pixels (perceptual targets)
- BERT: predicts discrete tokens (symbolic targets)
- data2vec: predicts continuous latent representations (semantic targets)

---

## Intermediate

### Architecture

**Teacher-student structure:**
1. **Teacher:** encode the full (unmasked) input with a standard transformer encoder. Top-$k$ layer activations are averaged and normalized to form the target representation.
2. **Student:** encode only the masked input (masking strategy matches each modality — image patches, text tokens, audio frames). Predict the teacher's target representations at the masked positions.

$$\mathcal{L} = \sum_{t \in M} \|p_\theta(x)_t - y_t\|_\beta^2$$

where $y_t$ is the teacher representation at masked position $t$, $p_\theta$ is the student prediction, and $\|\cdot\|_\beta$ is the smooth-$\ell_1$ (Huber) loss.

The **teacher is the EMA** of the student (like BYOL): $\xi_\text{teacher} \leftarrow \tau \xi_\text{teacher} + (1-\tau) \theta_\text{student}$.

**No discrete tokenizer:** unlike BERT (WordPiece) or BEiT (DALL-E tokenizer), data2vec predicts continuous representations directly — no bottleneck from discretization artifacts.

### Modality-Specific Details

**Vision:** same masking as MAE (75% mask ratio, random patches). Teacher processes full image.

**Text:** same masking as BERT (15%, [MASK] token replacement). Teacher processes unmasked text.

**Speech:** contiguous segments masked (BPE-like masking of audio frames). Teacher processes full waveform.

The encoder architecture (transformer) and training objective (predict teacher latents) are the same across all three modalities.

---

## Advanced

### data2vec 2.0

**data2vec 2.0 (Baevski et al., 2023):** dramatically improves efficiency by:
1. Only computing teacher representations for masked regions (not full sequence) during training
2. Using multi-mask training — multiple independent masks per sample, all sharing the same teacher representations
3. Inverse block masking — masked regions are more structured (inverted block masks rather than random patches)

**Speed:** 16× faster training than original data2vec with better or equal performance.

### Comparison to MAE and BERT

| Property | BERT | MAE | data2vec |
|----------|------|-----|----------|
| Target type | Discrete tokens | Raw pixels | Continuous latents |
| Teacher needed | No | No | Yes (EMA) |
| Modality-specific | Text only | Vision only | Any (same framework) |
| Feature quality | Strong (frozen) | Better (fine-tuned) | Competitive |
| Collapse prevention | Via softmax | Via reconstruction | Via EMA + layer avg |

### Why Latent Targets Work

Predicting latent representations instead of raw inputs encourages the student to learn **semantic** features:
- The teacher's top layers have already processed the full context, encoding semantic meaning
- Raw pixels/tokens contain a lot of low-level variation (lighting, noise) irrelevant to semantics
- The student's representations become aligned with the teacher's semantic space, not the raw input space

This makes data2vec features competitive with both contrastive (DINO) and reconstruction-based (MAE) approaches.

*See also: [[masked-autoencoders]] · [[bert-mlm]] · [[byol-simsiam]] · [[self-supervised-overview]] · [[vision-transformer]]*
