---
title: Transfer Learning and Fine-Tuning
tags: [transfer-learning, fine-tuning, computer-vision, deep-learning, domain-adaptation]
aliases: [transfer learning, fine-tuning, pretrained models, domain adaptation, feature extraction]
difficulty: 2
status: complete
related: [cnn-architectures-guide, normalization-layers, regularization-dropout, lora-quantization, self-supervised-overview]
---

# Transfer Learning and Fine-Tuning

---

## Fundamental

Instead of training from random initialization, start from weights pretrained on a large dataset (e.g., ImageNet, LAION). The pretrained model has already learned useful low-level features that transfer to most visual tasks.

**Why it works:** early CNN layers learn general features (Gabor-like edge detectors, color blobs) useful for virtually any image recognition task. Only later layers encode task-specific information.

| Layer depth | What it learns | Transferability |
|-------------|---------------|----------------|
| Early (conv1–conv2) | Edges, colors, oriented patterns | Universal |
| Middle (conv3–conv4) | Textures, object parts | General |
| Late (conv5+, fc) | Task-specific features | Task-specific — often replaced |
| Head (classifier) | Class logits | Always replaced |

### The Two Core Strategies

**Feature Extraction (Freeze All):** replace the classification head; freeze all pretrained weights; train only the new head. Use when: dataset is very small (<1k images) and domain is similar to pretraining.

**Fine-Tuning (Unfreeze Some/All):** replace the head, then progressively unfreeze top layers with a small learning rate. Use when: dataset is moderate-to-large or the target domain differs from pretraining.

---

## Intermediate

### How Much to Freeze: Decision Framework

| Dataset size | Domain similarity | Recommendation |
|-------------|-----------------|----------------|
| Small | Similar | Feature extraction only |
| Small | Different | Fine-tune top layers with caution |
| Large | Similar | Fine-tune top 1–2 blocks |
| Large | Different | Full fine-tuning or train from scratch |

Domain gap examples: ImageNet→CIFAR-10 (low), ImageNet→satellite (medium), ImageNet→X-ray (high), ImageNet→audio spectrograms (extreme).

### Discriminative Learning Rates

Use lower learning rates for earlier layers, higher for later layers:

$$\eta_\ell = \frac{\eta_\text{head}}{k^{L - \ell}}$$

where $k$ is the decay factor (commonly 2–10) and $L$ is the number of layer groups. **Rationale:** early layers already have good features — high LR destroys them; later layers are task-specific and need larger updates.

### Gradual Unfreezing

1. Freeze all pretrained layers. Train head for $N_1$ epochs.
2. Unfreeze last block. Train for $N_2$ epochs with discriminative LR.
3. Unfreeze next-to-last block. Train for $N_3$ epochs.
4. Continue until all layers are unfrozen.

**Why:** prevents catastrophic forgetting — earlier layers are exposed to gradient updates only once the head has stabilized, so gradients are more informative. (ULMFiT, Howard & Ruder 2018.)

---

## Advanced

### Catastrophic Forgetting

When fine-tuning overwrites pretrained features with task-specific ones, losing the general knowledge. Signs: performance on ImageNet drops after fine-tuning; model fails to generalize to slightly different examples.

**Mitigations:**
- **Low LR for earlier layers** (discriminative rates)
- **Elastic Weight Consolidation (EWC):** add penalty $\lambda \sum_i F_i (\theta_i - \theta^*_i)^2$ where $F_i$ is Fisher information — penalizes changing weights important to the original task
- **Replay:** mix in a fraction of pretraining data during fine-tuning

### Parameter-Efficient Fine-Tuning (PEFT)

For very large pretrained models (ViT-Large, CLIP, LLaMA), full fine-tuning is expensive. PEFT methods add a small number of trainable parameters while keeping pretrained weights frozen:

- **LoRA:** decompose weight updates as $\Delta W = BA$ where $r \ll \min(d,k)$. Only $A$ and $B$ are trained. See [[lora-quantization]].
- **Adapter layers:** small trainable MLP modules inside each transformer block.
- **Prompt tuning:** prepend learnable soft token embeddings to the input.

### Self-Supervised Pretraining

Modern vision models are pretrained without ImageNet labels:

| Method | Pretext Task | Key Idea |
|--------|------------|---------|
| DINO / DINOv2 | Self-distillation | Student-teacher with no labels |
| MAE | Masked autoencoding | Reconstruct masked patches |
| CLIP | Image-text contrastive | Match images to captions |
| SimCLR / MoCo | Contrastive | Augmented views should be similar |

Advantage: pretraining on much larger unlabeled datasets (web-scale). Often outperforms ImageNet-supervised pretraining on downstream tasks.

### When NOT to Use Transfer Learning

1. **Very different modalities:** ImageNet weights for audio, molecular graphs, or tabular data provide little benefit.
2. **Enough task-specific data:** with millions of examples, pretraining may not help.
3. **Strict privacy:** pretrained models may have memorized training data (membership inference risk).
4. **Real-time constraints:** a smaller model trained from scratch may be faster.

---

*See also: [[cnn-architectures-guide]] · [[self-supervised-overview]] · [[lora-quantization]] · [[normalization-layers]]*
