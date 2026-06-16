---
title: Prefix Tuning and Prompt Tuning
tags: [prefix-tuning, prompt-tuning, soft-prompts, parameter-efficient, fine-tuning]
aliases: [prefix tuning, prompt tuning, soft prompts, continuous prompts, P-tuning]
difficulty: 1
status: complete
depends_on: [attention-mechanism, arch-kv-cache, instruction-tuning]
related: [lora-quantization, instruction-tuning, prompt-engineering, adapter-layers, attention-mechanism, arch-kv-cache]
---

# Prefix Tuning and Prompt Tuning

---

## Fundamental

### Motivation

Full fine-tuning updates all model parameters — expensive storage (one copy per task) and prone to catastrophic forgetting. PEFT (Parameter-Efficient Fine-Tuning) methods update only a small number of parameters while keeping the pretrained model frozen.

**Prefix tuning** and **prompt tuning** prepend learnable "virtual tokens" to the input, conditioning the frozen model's behavior without touching its weights.

**Intuition:** think of the soft prefix as a learned "instruction" in the model's native language — not natural language words, but continuous embedding vectors that steer the frozen model's attention patterns. Instead of finding the right words to put in a prompt, gradient descent finds the right vectors directly, optimizing for task performance end-to-end.

### Prompt Tuning

**Prompt Tuning (Lester et al., 2021):** prepend $k$ trainable embedding vectors ("soft prompts") to the input sequence before the first transformer layer:

$$\text{input} = [\underbrace{P_1, P_2, \ldots, P_k}_{\text{learnable}}, x_1, x_2, \ldots, x_n]$$

where:
- $P_1, \ldots, P_k$ — learnable soft prompt vectors; each $P_i \in \mathbb{R}^d$ is a $d$-dimensional embedding vector (not real words — continuous vectors in embedding space, optimized by gradient descent)
- $k$ — number of prefix tokens (typically 10–100; more gives more "space" to condition the model)
- $d$ — the model's embedding dimension (same size as real token embeddings)
- $x_1, \ldots, x_n$ — the actual input tokens (real word embeddings, fixed)
- $[\cdot, \cdot]$ — concatenation along the sequence dimension

Only $P_i \in \mathbb{R}^d$ are trained (parameters = $k \times d$). The frozen model processes this augmented sequence normally.

**Result:** at model scale ≥ 11B, prompt tuning matches full fine-tuning quality. At smaller scales (< 1B), the gap is significant. The soft prompts "reprogram" the frozen model's attention patterns.

---

## Intermediate

### Prefix Tuning

**Prefix Tuning (Li & Liang, 2021):** more powerful — prepend learnable KV pairs to **every** transformer layer's attention, not just the input embeddings:

For each layer $l$, the attention key/value matrices are prepended with trainable prefix matrices:
$$K_l = [P^K_l; X W^K_l], \quad V_l = [P^V_l; X W^V_l]$$

where:
- $K_l, V_l$ — the key and value matrices at layer $l$ (the full set that attention queries will attend over)
- $P^K_l, P^V_l \in \mathbb{R}^{k \times d_\text{head}}$ — the trainable prefix key/value matrices at layer $l$ ($k$ = prefix length, $d_\text{head}$ = dimension per attention head)
- $X$ — the input token representations at layer $l$ (frozen)
- $W^K_l, W^V_l$ — the frozen pretrained key/value projection matrices at layer $l$
- $[;]$ — vertical concatenation (stacking along the sequence dimension)

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

## Links

- [[attention-mechanism]] — prefix tokens are prepended to the key and value sequences in each layer; attention computes over both prefix and actual tokens
- [[arch-kv-cache]] — prefix KV states can be precomputed and cached; prefix tuning is analogous to extending the KV cache with soft learned tokens
- [[instruction-tuning]] — prefix tuning is a parameter-efficient alternative to full SFT; only prefix vectors are trained, not backbone weights
- [[lora-quantization]] — LoRA is a more popular PEFT alternative; LoRA modifies weight matrices while prefix tuning prepends continuous embeddings
- [[adapter-layers]] — adapter layers insert modules in the residual path; prefix tuning inserts soft tokens in the input space — different intervention points
- [[prompt-engineering]] — prefix tuning automates prompt search: instead of manual discrete prompts, it learns continuous soft prompts end-to-end
