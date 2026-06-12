---
title: Speculative Decoding
tags: [speculative-decoding, draft-model, inference-acceleration, autoregressive, token-generation]
aliases: [speculative sampling, draft-verify, assisted decoding, speculative execution]
difficulty: 2
status: complete
related: [arch-kv-cache, autoregressive-models, attention-mechanism, flash-attention, flash-decoding]
---

# Speculative Decoding

---

## Fundamental

### The Autoregressive Bottleneck

Autoregressive generation is sequential: to generate token $t+1$, the model must have computed token $t$. Each step requires a full forward pass through a large model (e.g., 70B parameters), making inference memory-bandwidth bound. Even with batching, each individual sequence is decoded serially.

**The insight:** the model spends the same compute verifying an easy prediction ("the" after "I am") as a hard one. If we could propose likely tokens cheaply, then verify multiple at once, we'd save wall-clock time.

### Draft-Then-Verify

**Speculative decoding (Leviathan et al., 2023; Chen et al., 2023):**

1. **Draft phase:** run a small, fast **draft model** $M_q$ autoregressively to generate $\gamma$ candidate tokens $\tilde{x}_1, \ldots, \tilde{x}_\gamma$
2. **Verify phase:** run the large target model $M_p$ once on the $\gamma$-token draft sequence (one forward pass, processes all $\gamma$ tokens in parallel thanks to teacher forcing)
3. **Accept or reject:** accept token $i$ with probability $\min(1, p(\tilde{x}_i) / q(\tilde{x}_i))$; reject at the first mismatch; sample a correction token from $p - q$ (adjusted distribution)

**Key guarantee:** the output distribution is identical to sampling directly from $M_p$. No quality loss.

---

## Intermediate

### Acceptance Rate and Speedup

Let $\alpha$ = average acceptance rate per draft token. Expected tokens generated per target forward pass:

$$\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

For $\gamma = 4$ draft tokens and $\alpha = 0.8$: expected $\approx 3.36$ tokens per target call, vs 1 without speculation.

**Wall-clock speedup** depends on:
- $\alpha$ (acceptance rate) — driven by how well draft distribution matches target
- Draft model latency vs target model latency
- Batch size and KV cache management

Typical reported speedups: **2–3× on greedy/low-temperature**, lower for high-temperature (more diverse outputs → lower $\alpha$).

### Draft Model Options

| Type | Example | Tradeoff |
|------|---------|---------|
| Smaller same-family model | LLaMA-7B drafts for LLaMA-70B | Best $\alpha$, requires extra memory |
| Quantized version of target | INT4 target drafts for FP16 target | No extra parameters; lower $\alpha$ |
| n-gram / retrieval | Database lookups | Zero model cost; $\alpha$ depends on repetition |
| Self-speculative (Medusa) | Extra heads on target predict $k$ steps ahead | No separate model; heads share backbone |

**Medusa (Cai et al., 2024):** adds $\gamma$ extra decoding heads on top of the frozen target model, each predicting position $+1, +2, \ldots, +\gamma$. Trained with a tree-attention structure and verified with the same rejection sampling guarantee.

---

## Advanced

### Tree Attention for Speculative Decoding

Evaluating multiple draft continuations at once requires a **tree structure**: the draft model proposes not just one sequence but a tree of possible continuations (e.g., top-2 tokens at each step = $2^\gamma$ candidates). Tree attention computes all paths in the tree in a single target forward pass using a custom attention mask.

**SpecInfer (Miao et al., 2023):** formally combines tree-based speculation with exact matching, showing that tree drafts achieve higher acceptance than beam-style drafts.

### Limitations

- **Hardware efficiency:** the verification pass runs with batch size 1 (just the draft sequence). This is inefficient on high-throughput serving systems where larger batches amortize compute better. At large batch sizes, speculative decoding provides less speedup.
- **Memory overhead:** storing the draft model (even small) competes with the KV cache for memory
- **Continuous batching:** integrating speculative decoding with continuous batching (vLLM) requires careful synchronization — draft sequences have variable accept lengths, disrupting fixed-length batches

### Self-Consistency and Speculative Decoding

Speculative decoding is strictly equivalent to target model sampling — it does not change the model's outputs, only the speed. This is important: methods that change the output distribution (e.g., lookahead decoding, contrastive decoding) are separate from speculative decoding and do not preserve the target distribution.

*See also: [[arch-kv-cache]] · [[autoregressive-models]] · [[flash-decoding]] · [[attention-mechanism]]*
