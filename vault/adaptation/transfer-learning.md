---
title: Transfer Learning and Fine-Tuning
tags: [transfer-learning, fine-tuning, computer-vision, deep-learning, domain-adaptation]
aliases: [transfer learning, fine-tuning, pretrained models, domain adaptation, feature extraction]
difficulty: 2
status: complete
depends_on: [backpropagation, normalization-layers, regularization-dropout]
related: [cnn-architectures-guide, normalization-layers, regularization-dropout, lora-quantization, self-supervised-overview]
---

# Transfer Learning and Fine-Tuning

---

## Fundamental

**The problem:** training a deep neural network on a small target dataset (a few thousand images) from random initialization leads to overfitting — the model has too many parameters relative to data. Yet large labeled datasets are expensive and slow to collect for specialized domains.

**The solution:** start from a model already trained on millions of examples. Feature learning is the expensive part; most of what a vision model needs to know (how to detect edges, textures, shapes) is domain-agnostic. Transfer this general knowledge and fine-tune only what is task-specific.

**Intuition:** think of the pretrained model as an expert who already understands visual structure. Fine-tuning teaches that expert your specific vocabulary (classes, output format) rather than making them re-learn how to see. The deeper the layer, the more task-specific the knowledge — early layers rarely need retraining.

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

where:
- $\eta_\ell$ — the learning rate for layer group $\ell$
- $\eta_\text{head}$ — the learning rate for the top (head) layer group (the highest value used)
- $k$ — decay factor (commonly 2–10): each layer group gets $k$ times lower LR than the group above it
- $L$ — total number of layer groups
- $\ell$ — index of the current layer group; when $\ell = L$ (head): $\eta_L = \eta_\text{head}$ (full LR); when $\ell = 0$ (first layer): $\eta_0 = \eta_\text{head}/k^L$ (smallest LR)

**Rationale:** early layers already have good features — high LR destroys them; later layers are task-specific and need larger updates.

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
- **Elastic Weight Consolidation (EWC):** add penalty $\lambda \sum_i F_i (\theta_i - \theta^*_i)^2$ to the loss, where $\lambda$ is the regularization strength, $F_i$ is [[fisher-information|Fisher information]] (measures how sensitive the original task's loss is to parameter $i$), $\theta_i$ is the current parameter value, and $\theta^*_i$ is the frozen snapshot from before fine-tuning — penalizes changing weights important to the original task
- **Replay:** mix in a fraction of pretraining data during fine-tuning

### Parameter-Efficient Fine-Tuning (PEFT)

For very large pretrained models ([[vision-transformer|ViT-Large]], [[clip|CLIP]], LLaMA), full fine-tuning is expensive. PEFT methods add a small number of trainable parameters while keeping pretrained weights frozen:

- **LoRA:** decompose weight updates as $\Delta W = BA$ where $r \ll \min(d,k)$. Only $A$ and $B$ are trained. See [[lora-quantization]].
- **Adapter layers:** small trainable MLP modules inside each transformer block.
- **Prompt tuning:** prepend learnable soft token embeddings to the input.

### Self-Supervised Pretraining

Modern vision models are pretrained without ImageNet labels:

| Method | Pretext Task | Key Idea |
|--------|------------|---------|
| DINO / DINOv2 | [[knowledge-distillation\|Self-distillation]] | Student-teacher with no labels |
| MAE | Masked autoencoding | Reconstruct masked patches |
| [[clip\|CLIP]] | Image-text contrastive | Match images to captions |
| [[contrastive-learning\|SimCLR / MoCo]] | Contrastive | Augmented views should be similar |

Advantage: pretraining on much larger unlabeled datasets (web-scale). Often outperforms ImageNet-supervised pretraining on downstream tasks.

### When NOT to Use Transfer Learning

1. **Very different modalities:** ImageNet weights for audio, molecular graphs, or tabular data provide little benefit.
2. **Enough task-specific data:** with millions of examples, pretraining may not help.
3. **Strict privacy:** pretrained models may have memorized training data (membership inference risk).
4. **Real-time constraints:** a smaller model trained from scratch may be faster.

---

## Links

- [[backpropagation]] — fine-tuning propagates gradients through the pretrained weights; freezing layers stops gradient flow to preserve pretrained representations
- [[normalization-layers]] — batch norm statistics are domain-specific; re-estimating them on the target domain (or using layer norm) is often the first fine-tuning fix
- [[regularization-dropout]] — dropout during fine-tuning prevents overfitting when the target dataset is small; lower dropout rates than pretraining are typical
- [[cnn-architectures-guide]] — CNNs pretrained on ImageNet are the canonical transfer learning backbone for vision; lower layers capture edges, higher layers capture semantics
- [[self-supervised-overview]] — self-supervised pretraining (MAE, DINO, SimCLR) produces transferable features without labeled data
- [[lora-quantization]] — LoRA is the standard PEFT method for transfer learning in LLMs; it adds trainable rank-$r$ matrices without touching frozen weights
- [[fisher-information]] — EWC (Elastic Weight Consolidation) uses the Fisher diagonal to regularize important weights during fine-tuning, preventing catastrophic forgetting
- [[clip]] — CLIP enables zero-shot transfer via text prompts; its visual encoder transfers to downstream tasks through linear probing
- [[contrastive-learning]] — contrastively pretrained encoders (SimCLR, MoCo) transfer better than fully supervised models on out-of-distribution tasks
- [[vision-transformer]] — ViTs fine-tuned from pretrained checkpoints (DeiT, DINO) show stronger transfer than CNNs on many vision benchmarks
- [[knowledge-distillation]] — distillation is a form of transfer: the student learns from the teacher's soft outputs rather than hard labels
