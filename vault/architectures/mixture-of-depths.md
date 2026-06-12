---
title: Mixture of Depths (MoD)
tags: [mixture-of-depths, mod, token-routing, adaptive-compute, conditional-computation]
aliases: [Mixture of Depths, MoD, token skipping, adaptive depth, conditional computation]
difficulty: 2
status: complete
related: [mixture-of-experts, attention-mechanism, autoregressive-models, arch-residual-block, flash-attention]
---

# Mixture of Depths (MoD)

---

## Fundamental

### Motivation

In a standard transformer, every token passes through every layer with equal compute. But not all tokens need equal processing:
- Common, predictable tokens ("the", "a", punctuation) need less compute
- Rare, semantically dense tokens need more

**Mixture of Depths (Raposo et al., 2024)** routes tokens to skip certain transformer layers entirely — each layer processes only a fraction of tokens rather than all of them.

### Algorithm

For each layer $l$, a learned router assigns a scalar score to each token and selects the top-$k$ tokens (by capacity fraction $C$, e.g., 12.5% of tokens per layer):

1. Compute router scores: $r_t = w^\top x_t$ (scalar per token)
2. Select top $\lfloor C \cdot T \rfloor$ tokens by score
3. Selected tokens pass through the layer (attention + FFN)
4. Skipped tokens bypass via residual connection: $x_t \leftarrow x_t$ (unchanged)
5. Router weight is multiplied into selected tokens' outputs (weighted combination)

This is in contrast to MoE (see [[mixture-of-experts]]) which routes tokens to different **expert FFNs** at the same layer — MoD routes tokens to skip entire **layers**.

---

## Intermediate

### Compute Savings

If capacity $C = 0.125$ (12.5% of tokens processed per layer) across all layers:
- Each layer processes $12.5\%$ of tokens → $87.5\%$ savings per layer
- But attention still requires all tokens as context for KV computation (depending on implementation)

**Practical savings:** for isoFLOP comparison (same total compute budget), MoD models can be larger (more parameters, more layers) than dense models, achieving better quality at the same compute.

From the paper: a MoD transformer matches a dense transformer at 50% of the FLOPs when properly sized.

### Autoregressive Sampling Complication

During training: easy — process top-$C$ tokens in parallel, skip the rest.

During autoregressive decoding: the router must decide whether the current token should be processed by each layer before processing it. This creates a dependency: the router decision requires the input representation, but we want to skip computation based on that decision.

**Solutions:**
1. **Separate auxiliary router:** train a lightweight predictor of the main router's decision — used only at inference
2. **Fixed-depth sampling:** fall back to processing every token at every layer (lose compute savings at inference, keep quality)
3. **Top-$p$ threshold:** use a fixed threshold instead of top-$k$ so the router can be applied per-token independently

---

## Advanced

### MoD vs MoE

| Property | MoD | MoE |
|----------|-----|-----|
| What is routed | Tokens (skip layers) | Tokens (different FFN experts) |
| Capacity constraint | Per-layer token budget | Per-expert token budget |
| Parameter count | Same as dense | $E \times$ more than dense |
| Active compute | Proportional to $C$ | Proportional to $k/E$ |
| KV cache | Full sequence needed | Full sequence needed |
| Inference complexity | Variable depth | Variable FFN |

**Hybrid MoD+MoE:** combine both — route some tokens to skip the layer entirely (MoD), and for tokens that do process, route to different experts (MoE). Provides both depth-adaptive and width-adaptive compute allocation.

### Connection to Early Exit

**Early exit** methods (also called adaptive depth inference) allow the network to exit after any layer if the current representation is confident enough. MoD is a per-layer version: instead of exiting globally, individual tokens exit specific layers.

Both methods share the insight that uniform depth is wasteful. The difference: early exit exits the entire sequence at the same layer; MoD exits individual tokens independently.

*See also: [[mixture-of-experts]] · [[attention-mechanism]] · [[arch-residual-block]] · [[autoregressive-models]]*
