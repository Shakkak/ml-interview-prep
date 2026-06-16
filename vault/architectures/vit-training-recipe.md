---
title: ViT Training and Fine-Tuning Recipe
tags: [vision-transformer, training, stochastic-depth, droppath, llrd, layer-wise-lr, adamw, deit, augmentation, fine-tuning]
aliases: [ViT training recipe, stochastic depth, DropPath, layer-wise learning rate decay, LLRD, DeiT recipe, ViT fine-tuning]
difficulty: 2
status: complete
depends_on: [vision-transformer, optimizer-adam, normalization-layers]
related: [vision-transformer, optimizer-adam, optimizer-lr-schedules, data-augmentation, regularization-dropout, arch-positional-encoding, normalization-layers, knowledge-distillation]
---

# ViT Training and Fine-Tuning Recipe

---

## Fundamental

CNNs ship with three regularizers baked in — translation equivariance, local connectivity, and weight sharing across positions. These act as a prior that restricts the function space to image-plausible solutions, reducing the amount of data and regularization needed. ViT has none of these. Training ViT from scratch on small datasets without compensating for this requires explicit regularization that CNNs simply don't need.

**Why SGD fails for ViT:** the standard ResNet training recipe (SGD + momentum + weight decay) consistently underperforms for ViT. The core issue is that ViT's attention mechanism has flat curvature directions — many directions in weight space leave the loss almost unchanged. SGD treats all directions equally. AdamW normalizes by per-parameter gradient variance, effectively pre-conditioning each dimension:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\,\hat{m}_t - \eta\lambda\,\theta_t$$

The $\sqrt{\hat{v}_t}$ denominator shrinks updates in directions with high gradient variance (noisy, unreliable) and amplifies updates in low-variance directions (consistent, trustworthy). This is critical for attention weight matrices where meaningful updates coexist with many flat directions.

**Core recipe components:** AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\lambda=0.05$) + linear warmup (typically 5–10% of total steps) + cosine decay + stochastic depth + strong augmentation.

### Stochastic Depth (DropPath)

Stochastic depth (Huang et al., 2016) randomly **drops entire residual blocks** during training. For a residual block $F_l$ at layer $l$ of $L$ total layers:

$$\text{output}_l = x + b_l \cdot F_l(x), \quad b_l \sim \text{Bernoulli}(p_l)$$

The survival probability $p_l$ is linearly scheduled from $1.0$ (first layer always kept) to $1 - d_{\text{rate}}$ (last layer dropped most often):

$$p_l = 1 - \frac{l}{L} \cdot d_{\text{rate}}$$

At test time all blocks are kept and each block's output is scaled by its survival probability $p_l$ (analogous to dropout's inference scaling). Typical rates: $d_{\text{rate}} = 0.1$ for ViT-S, $0.2$ for ViT-B, $0.3$–$0.5$ for ViT-L/H.

> [!tip] Why early layers survive more often ([[regularization-dropout]])
> Early ViT layers learn general low-frequency spatial patterns — they are the most data-efficient and hardest to re-learn if randomly dropped.
> Later layers learn task-specific high-level representations that are more redundant.
> The linear schedule from $p=1$ to $p=(1-d)$ is a soft version of the intuition that "deeper = more redundant" — the model trains as a random ensemble of shallower sub-networks, each capable of producing a useful gradient signal.
> This is why stochastic depth acts as both regularization (prevents over-reliance on any specific layer) and an implicit ensemble method.

---

## Intermediate

### DeiT: Training ViT-B/16 Without JFT-21k

The original ViT paper required JFT-21k (300M images) to train ViT-B/16 to match ResNets on ImageNet-1k. DeiT (Touvron et al., 2021) showed that aggressive augmentation + [[knowledge-distillation|distillation]] closes this gap with ImageNet-1k alone (1.28M images).

**DeiT augmentation stack:**

| Technique | Effect |
|-----------|--------|
| RandAugment ($N=2$, $M=9$) | Random photometric transforms |
| CutMix ($\alpha=1.0$) | Paste rectangular patches across images, blend labels |
| Mixup ($\alpha=0.8$) | Linear blend of two images and their labels |
| Label smoothing ($\varepsilon=0.1$) | Prevent overconfident softmax; acts as entropy regularizer |
| Repeated augmentation | Sample same image twice per batch; increases effective diversity |
| Random erasing | Randomly zero out rectangular regions (like dropout on pixels) |

Each technique independently provides modest gains; their combination is multiplicative. The key insight: in the absence of CNN's inductive bias, the model has a larger effective hypothesis space — the augmentation stack constrains it by forcing consistency under a wider variety of transforms.

**Distillation token:** DeiT adds a **distillation token** alongside the [CLS] token. It attends to all patches and is trained separately to match a CNN teacher's hard prediction (argmax, not soft probabilities):

$$\mathcal{L}_{\text{DeiT}} = \frac{1}{2}\mathcal{L}_{\text{CE}}(y_{\text{CLS}}, y_{\text{true}}) + \frac{1}{2}\mathcal{L}_{\text{CE}}(y_{\text{dist}}, y_{\text{teacher}})$$

At inference, predictions from [CLS] and distillation tokens are averaged. The distillation token learns a "CNN-like" representation that is complementary to the global [CLS] representation — empirically this adds ~1–2% accuracy.

### Resolution Fine-Tuning

ViT learns positional embeddings for $N_{\text{train}}$ patches during pretraining (e.g., $N=196$ at 224×224 with patch 16). Fine-tuning at higher resolution (384×384 gives $N=576$) requires position embeddings for unseen positions.

**Standard approach:** bicubic 2D interpolation of the $N_{\text{train}}$ learned embeddings to $N_{\text{finetune}}$ positions. The interpolation works because the model learns smooth position embeddings — nearby positions have similar embeddings, so intermediate positions can be reliably interpolated.

```python
# Interpolate from 14×14 to 24×24 grid (224→384 at patch 16)
pe_2d = pe[1:].reshape(14, 14, d)          # exclude [CLS] token
pe_interp = F.interpolate(pe_2d.permute(2,0,1).unsqueeze(0),
                          size=(24, 24), mode='bicubic')
pe_new = pe_interp.squeeze(0).permute(1,2,0).reshape(576, d)
pe_final = torch.cat([pe[:1], pe_new], dim=0)  # prepend [CLS] embedding
```

Fine-tuning at the higher resolution for a short number of steps (e.g., 30 epochs) then recovers and surpasses pretraining accuracy. ViT-B/16 pretrained at 224 + fine-tuned at 384 improves from 86.0% to 86.9% on ImageNet.

---

## Advanced

### Layer-Wise Learning Rate Decay (LLRD)

When fine-tuning a pretrained ViT on a downstream task, applying the same learning rate to all layers is suboptimal. Early layers encode general low-level features that transfer across all tasks; they should barely change. Later layers encode task-specific representations that need to adapt more. **LLRD** implements this by assigning exponentially decreasing learning rates from the last layer to the first:

$$\eta_l = \eta_{\text{base}} \times \alpha^{L - l}, \quad \alpha \in [0.65, 0.85]$$

where $L$ is the total number of layers and $l$ is the current layer index (0-indexed from bottom). With $\alpha = 0.75$ and $L = 12$ (ViT-B), the first layer receives $\eta_0 = \eta_{\text{base}} \times 0.75^{12} \approx 0.032 \times \eta_{\text{base}}$.

**Why it works:** the pretrained representations in early layers have been optimized across millions of images to capture universally useful features (edges, textures, spatial relationships). Fine-tuning data is orders of magnitude smaller — updating early layers aggressively on limited data causes catastrophic forgetting of these general features. LLRD is a lightweight alternative to full layer freezing that allows early layers to adapt slightly while preserving most of their generality.

**Practical settings for ViT fine-tuning:**

| Hyperparameter | ViT-B/16 | ViT-L/16 |
|---|---|---|
| Base LR ($\eta_{\text{base}}$) | $5 \times 10^{-4}$ | $2 \times 10^{-4}$ |
| LLRD decay ($\alpha$) | 0.75 | 0.65 |
| Weight decay | 0.05 | 0.05 |
| Stochastic depth | 0.1 | 0.2 |
| Warmup epochs | 5 | 5 |
| Total epochs | 100 | 50 |
| Batch size | 1024 | 512 |

### MAE Fine-Tuning vs. CLIP Fine-Tuning

MAE and CLIP pretrained ViTs need different fine-tuning approaches because the pretraining objective shapes what the encoder has learned.

**MAE encoder** has learned to reconstruct fine-grained spatial detail. Fine-tuning transfers this to discriminative tasks by aggressively replacing the pretraining signal with the downstream task. LLRD is essential — the early MAE layers encode low-level pixel statistics that change dramatically when moving to a high-level classification task.

**CLIP encoder** has been directly optimized for semantic discrimination via contrastive language-image training. The features are already well-organized in a semantically meaningful space. Fine-tuning risks destroying the zero-shot generalization that makes CLIP valuable. Best practice: fine-tune only the final few layers, or use LoRA adapters ([[lora-quantization]]), rather than full LLRD fine-tuning.

> [!warning] CLIP fine-tuning destroys zero-shot capability
> Full fine-tuning of a CLIP encoder on a domain-specific dataset (e.g., medical images) rapidly degrades performance on the original image-text zero-shot tasks — the kind of catastrophic forgetting that makes the fine-tuned model useless for anything outside its fine-tuning distribution.
> If you need both domain-specific accuracy and zero-shot generalization, fine-tune only the projection head or use parameter-efficient methods (LoRA, adapter layers). Keep the visual encoder frozen unless your domain dataset is very large (>1M images).

### ViT-G and Extreme Scale

At ViT-G scale (1.8B–2B params), the standard recipes need adjustment:
- **Batch size:** 65,536–262,144 with gradient accumulation — required to provide enough negative diversity in self-supervised training.
- **Training duration:** 3–10 epochs over JFT-3B or similar — longer than smaller models relative to parameters because the optimization landscape is flatter.
- **Gradient clipping** (max norm 1.0) becomes essential — gradient spikes that are survivable for ViT-B can destabilize ViT-G due to sheer depth.
- **Stochastic depth** rate increases to 0.4–0.5 — with 48 transformer blocks, dropping 40% of late blocks per step still leaves 28+ effective blocks per forward pass.

---

## Links

- [[vision-transformer]] — this file documents the training details needed to actually reproduce ViT results; the architecture file describes structure, this one describes the recipe
- [[optimizer-adam]] — AdamW (Adam with decoupled weight decay) is the standard optimizer for ViT training; the weight decay rate and $\beta_2$ differ from CNN recipes
- [[normalization-layers]] — pre-norm ViTs (LayerNorm before attention/FFN) are more stable than post-norm; this choice significantly affects learning rate sensitivity
- [[optimizer-lr-schedules]] — cosine schedule with linear warmup is the standard ViT LR schedule; LLRD (layer-wise LR decay) improves fine-tuning by using lower LRs in earlier layers
- [[data-augmentation]] — strong augmentation (RandAugment, Mixup, CutMix) is essential for ViT training from scratch; ViTs have fewer inductive biases than CNNs
- [[regularization-dropout]] — DropPath (stochastic depth) randomly drops entire residual branches during training; it is the primary regularization for ViT, replacing dropout
- [[knowledge-distillation]] — DeiT trains ViT-S/B from scratch using a CNN teacher via a distillation token; knowledge distillation compensates for ViT's lack of local inductive biases
- [[lora-quantization]] — LoRA fine-tuning of ViT uses lower ranks for earlier layers and higher ranks for later ones, following the LLRD intuition
- [[arch-positional-encoding]] — 2D learned position embeddings must be interpolated when fine-tuning at higher resolution than pretraining; bilinear interpolation is the standard approach
