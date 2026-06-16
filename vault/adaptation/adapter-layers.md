---
title: Adapter Layers
tags: [adapter-layers, parameter-efficient, fine-tuning, bottleneck-adapter, peft]
aliases: [adapters, bottleneck adapters, Houlsby adapters, AdapterFusion, PEFT adapters]
difficulty: 1
status: complete
depends_on: [transfer-learning, arch-bottleneck-1x1, backpropagation]
related: [lora-quantization, prefix-tuning, instruction-tuning, arch-bottleneck-1x1, transfer-learning]
---

# Adapter Layers

---

## Fundamental

### The Problem

Fine-tuning a large pretrained model on a new task means updating every parameter — hundreds of millions of weights. This is expensive, requires separate model copies per task, and risks overwriting general knowledge that the model already has.

Adapters solve this by keeping the pretrained model completely frozen and inserting small trainable modules inside it. Only the adapter parameters are updated during fine-tuning.

### Why Adapters Were Invented

Houlsby et al. (2019) asked: what is the minimum number of parameters needed to adapt a pretrained model to a new task without degrading its general representations? Their answer was a bottleneck module — compress the representation to a low-dimensional space, apply a task-specific transformation, then project back. The pretrained weights supply the rich representations; the adapter learns only what needs to change.

### Intuition

Think of the pretrained model as a general-purpose feature extractor. Adapters are small "translator" modules that rephrase those features in a task-specific way. The bottleneck ($r \ll d$) forces the adapter to learn a compressed task representation — it cannot memorize individual examples, only extract what is consistently different about the new task.

### Architecture

```
Input x (dim d)
   → Linear down-project: d → r     # r << d, typically r = 16–64
   → Nonlinearity (ReLU or GeLU)
   → Linear up-project: r → d
   → + x  (residual connection)
Output (dim d)
```

where $d$ is the model hidden dimension and $r$ is the bottleneck rank (a hyperparameter).

**Parameter count per adapter:** $2 \times d \times r$ (two linear layers, no bias).

Example: BERT-base ($d = 768$, $r = 64$) — 98K parameters per adapter. With 12 layers × 2 adapters per layer, the total is ~2.4M parameters, about 1.9% of BERT's 125M.

**Placement:** Houlsby et al. insert adapters after the attention sublayer *and* after the FFN sublayer in each transformer block. A cheaper single-adapter variant (after FFN only) is more common in practice.

---

## Intermediate

### Why the Bottleneck Works

The frozen pretrained weights already encode rich representations. The bottleneck forces the adapter to learn a low-rank update — only the directions in representation space that matter for the new task. This is analogous to the intuition behind [[lora-quantization|LoRA]]: fine-tuning updates tend to live in a low-dimensional subspace of parameter space.

Universal approximation still holds: with sufficient rank $r$, any linear transformation from the pretrained space can be approximated. The nonlinearity lets the adapter learn non-linear task adaptations.

### Adapter vs LoRA

| Property | Adapters | LoRA |
|----------|----------|------|
| Where applied | Sequential (adds layers) | Parallel (modifies weight matrices) |
| Inference cost | Extra forward pass per adapter | Zero (weights merged at inference) |
| Composability | Cascade adapters | Approximately sum adapters |
| Architecture change | Yes (new modules) | No (merged into existing weights) |
| Historical era | BERT-era (2019–2021) | LLM era (2022–present) |

The key distinction: adapters add **sequential** computation — the representation must pass through the adapter before continuing. LoRA applies a **parallel** low-rank correction to the weight matrix, which can be merged into the original weights for zero inference overhead.

### AdapterFusion

**AdapterFusion (Pfeiffer et al., 2020):** train task-specific adapters for $K$ tasks independently, then learn a weighted combination.

At inference for a new task, a learned attention mechanism over all $K$ adapter outputs selects which task knowledge to use. Adapters are frozen during fusion training — only the fusion weights are updated. This enables multi-task and cross-task transfer without catastrophic forgetting.

---

## Advanced

### Adapter Variants

**Parallel Adapter:** instead of inserting adapters sequentially inside the residual stream, add them in parallel — the adapter output is added directly to the transformer block output. Reduces latency by running adapter computation in parallel with the main block.

**Compacter (Karimi Mahabadi et al., 2021):** parameterize the adapter weight matrices as a sum of [[kronecker-products|Kronecker products]]:

$$W = \sum_{i=1}^n A_i \otimes B_i$$

where $A_i \in \mathbb{R}^{a \times a}$ and $B_i \in \mathbb{R}^{b \times b}$ are small factor matrices, and $\otimes$ denotes the Kronecker product (the block matrix where each entry of $A_i$ multiplies all of $B_i$). This reduces the parameter count further — a rank-$n$ Kronecker parameterization needs $n(a^2 + b^2)$ parameters instead of $(ab)^2$ — at similar quality to standard adapters.

**Sparse Adapters:** add sparse (non-dense) bottleneck modules — only a small fraction of neurons per adapter are updated. Reduces parameter count at the cost of hardware efficiency.

### Multi-Task Serving

A frozen base model + task-specific adapters is a clean architecture for serving many fine-tuned variants:
- $K$ tasks require $K$ adapter sets (megabytes each)
- One base model loaded once (gigabytes)
- Adapters are swapped between requests at negligible cost

This pattern (popularized by Hugging Face PEFT) allows a single GPU to serve hundreds of task-specific variants of a large model simultaneously.

## Links

- [[transfer-learning]] — adapters are inserted into pretrained models to enable parameter-efficient fine-tuning without updating the frozen backbone
- [[arch-bottleneck-1x1]] — adapter modules use the same bottleneck structure: down-project to $r \ll d$, apply nonlinearity, up-project back to $d$
- [[backpropagation]] — during fine-tuning only adapter parameters receive gradients; backbone weights are frozen, reducing compute and memory
- [[lora-quantization]] — LoRA is the dominant PEFT alternative to adapters; both add trainable parameters but LoRA modifies weight matrices while adapters add sequential modules
- [[prefix-tuning]] — prefix tuning prepends soft tokens to the input instead of inserting modules; complementary PEFT approach with different trade-offs
- [[instruction-tuning]] — adapters are commonly used to fine-tune pretrained models for instruction following at much lower cost than full fine-tuning
- [[kronecker-products]] — Compacter parameterizes adapter weight matrices as sums of Kronecker products, reducing parameter count while preserving expressivity
