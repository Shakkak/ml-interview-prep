---
title: Prefix Tuning and Prompt Tuning
tags: [prefix-tuning, prompt-tuning, soft-prompts, parameter-efficient, fine-tuning]
aliases: [prefix tuning, prompt tuning, soft prompts, continuous prompts, P-tuning]
difficulty: 1
status: complete
related: [lora-quantization, instruction-tuning, prompt-engineering, adapter-layers, attention-mechanism, arch-kv-cache]
---

# Prefix Tuning and Prompt Tuning

---

## Fundamental

### Motivation

Full fine-tuning updates all model parameters — expensive storage (one copy per task) and prone to catastrophic forgetting. PEFT (Parameter-Efficient Fine-Tuning) methods update only a small number of parameters while keeping the pretrained model frozen.

**Prefix tuning** and **prompt tuning** prepend learnable "virtual tokens" to the input, conditioning the frozen model's behavior without touching its weights.

### Prompt Tuning

**Prompt Tuning (Lester et al., 2021):** prepend $k$ trainable embedding vectors ("soft prompts") to the input sequence before the first transformer layer:

$$\text{input} = [\underbrace{P_1, P_2, \ldots, P_k}_{\text{learnable}}, x_1, x_2, \ldots, x_n]$$

Only $P_i \in \mathbb{R}^d$ are trained (parameters = $k \times d$). The frozen model processes this augmented sequence normally.

**Result:** at model scale ≥ 11B, prompt tuning matches full fine-tuning quality. At smaller scales (< 1B), the gap is significant. The soft prompts "reprogram" the frozen model's attention patterns.

---

## Intermediate

### Prefix Tuning

**Prefix Tuning (Li & Liang, 2021):** more powerful — prepend learnable KV pairs to **every** transformer layer's attention, not just the input embeddings:

For each layer $l$, the attention key/value matrices are prepended with trainable prefix matrices:
$$K_l = [P^K_l; X W^K_l], \quad V_l = [P^V_l; X W^V_l]$$

The prefix $P^K_l, P^V_l \in \mathbb{R}^{k \times d_\text{head}}$ is prepended and the full token sequence attends over both prefix and input keys/values.

**Implementation note:** prefix KV pairs are stored directly in the KV cache format — inference is efficient (no extra forward passes).

**Parameters:** $k \times d_\text{model} \times 2 \times L$ for $L$ layers (key and value for each layer). For prefix length 100, ViT-L (1024 dim, 24 layers): ~5M parameters — 0.1% of model parameters.

**Prefix vs Prompt Tuning:** prefix conditions every layer (more control, more expressive), prompt tuning only conditions the first layer. Prefix tuning generally outperforms prompt tuning.

### P-Tuning v2

**P-Tuning v2 (Liu et al., 2022):** brings prefix tuning to NLU tasks, showing it matches or exceeds LoRA at similar parameter counts. Key additions:
- Apply prefix to all layers (as in prefix tuning)
- Use prefix length 100 (not short)
- Works for understanding and generation tasks

Comparison at 0.1% parameter budget: P-Tuning v2 ≈ LoRA ≈ fine-tuning on most NLP tasks.

---

## Advanced

### Tradeoffs vs LoRA

| Property | Prefix/Prompt Tuning | LoRA |
|----------|---------------------|------|
| Parameter count | $O(kLd)$ | $O(rLd)$ |
| Inference overhead | Prefix in KV cache | None (weights merged) |
| Task switching | Swap prefix KV cache | Swap LoRA weights |
| Expressiveness | Conditions all attn | Modifies weight matrices |
| Interpretability | Harder (virtual tokens) | Cleaner (weight deltas) |

**Key difference:** LoRA adapters can be merged into model weights before deployment ($W' = W + AB$), so inference is exactly the same cost as the base model. Prefix tuning always has $k$ extra tokens in the KV cache, adding $O(k/N)$ overhead for sequence length $N$.

### Prefix Tuning for Task Composition

Multiple prefixes can be composed — attend over multiple prefix sets simultaneously. This enables **multi-task inference** with a single frozen model:
- Prefix $A$ for style transfer
- Prefix $B$ for sentiment control
- Compose both for styled + controlled generation

This is harder with LoRA (weight matrices don't compose easily) and natural with prefix tuning (KV caches can be concatenated).

*See also: [[lora-quantization]] · [[instruction-tuning]] · [[adapter-layers]] · [[arch-kv-cache]] · [[prompt-engineering]]*
