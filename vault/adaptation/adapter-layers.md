---
title: Adapter Layers
tags: [adapter-layers, parameter-efficient, fine-tuning, bottleneck-adapter, peft]
aliases: [adapters, bottleneck adapters, Houlsby adapters, AdapterFusion, PEFT adapters]
difficulty: 1
status: complete
related: [lora-quantization, prefix-tuning, instruction-tuning, arch-bottleneck-1x1, transfer-learning]
---

# Adapter Layers

---

## Fundamental

### Adapter Architecture

**Adapters (Houlsby et al., 2019)** insert small bottleneck modules inside each transformer block. The pretrained model is frozen; only adapters are trained.

**Bottleneck adapter structure:**
```
Input x (dim d)
   → Linear down-project (d → r)  # r << d, e.g., r = 64
   → Nonlinearity (ReLU / GeLU)
   → Linear up-project (r → d)
   → + x  (residual connection)
Output (dim d)
```

Parameters per adapter: $2 \times d \times r$ (two linear layers). With $r = 64$, $d = 1024$: 131K parameters per adapter. A BERT-base with 12 layers and 2 adapters per layer: ~3M parameters = 2.5% of BERT's 125M.

**Placement:** typically after the attention sublayer and after the FFN sublayer in each transformer block (Houlsby configuration). Single adapter placement (after FFN only) is more common in practice.

---

## Intermediate

### Why Adapters Work

Adapters decompose the fine-tuning problem: the frozen pretrained weights capture general knowledge, while the small adapters learn task-specific transformations on top. The bottleneck structure ($r \ll d$) forces a compact, compressed task representation.

**Universal approximation in the adapter:** with sufficient rank $r$, any linear transformation from the pretrained representation can be approximated. Non-linearity allows adapting the function class, not just linear projection.

### Adapter vs LoRA Comparison

| Property | Adapters | LoRA |
|----------|----------|------|
| Where applied | Sequential (adds layers) | Parallel (to weight matrices) |
| Inference cost | Extra forward pass through adapter | None if weights merged |
| Composability | Cascade adapters | Sum adapters (approximately) |
| Architecture change | Yes (extra modules) | No (merged into weights) |
| Historical use | BERT-era | LLM era |

**Key difference:** adapters add **sequential** computation — input must pass through the adapter. LoRA applies a **parallel** low-rank update to the weight matrix, which can be merged for zero inference overhead.

### AdapterFusion

**AdapterFusion (Pfeiffer et al., 2020):** separately train task-specific adapters for $K$ tasks, then learn a weighted combination:

At inference for a new task, the model attends over all $K$ adapters using a learned attention mechanism over the adapter outputs. This enables multi-task and cross-task transfer without forgetting — adapters are frozen, only the fusion weights are trained.

---

## Advanced

### Adapter Variants

**Parallel Adapter (PA):** instead of inserting adapters sequentially in the residual path, add them in parallel — the adapter output is added directly to the transformer block output rather than applied inside the block. Reduces inference latency (parallel computation).

**Compacter (Karimi Mahabadi et al., 2021):** use Kronecker product to parameterize the adapter matrices: $W = \sum_i A_i \otimes B_i$ where $A_i, B_i$ are small matrices. More parameter-efficient than standard adapters at similar quality.

**Sparse Adapters:** add sparse (non-dense) bottleneck modules — only update a small fraction of neurons per layer.

### Multi-Task with Adapters

A single frozen base model + task-specific adapters is a clean architecture for multi-task serving:
- $K$ tasks require $K$ adapter sets (small)
- Base model loaded once (large)
- Swap adapters between requests — fast (adapter weights are small)

This "hot-swap" multi-task serving was popularized by Hugging Face PEFT and is used in production systems serving many fine-tuned variants of a base model.

*See also: [[lora-quantization]] · [[prefix-tuning]] · [[instruction-tuning]] · [[transfer-learning]] · [[arch-bottleneck-1x1]]*
