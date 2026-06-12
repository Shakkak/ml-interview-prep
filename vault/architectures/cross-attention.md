---
title: Cross-Attention
tags: [cross-attention, encoder-decoder, attention, transformers, multimodal]
aliases: [cross-attention, encoder-decoder attention, cross-modal attention]
difficulty: 1
status: complete
related: [attention-mechanism, arch-kv-cache, vision-transformer, vlm-architectures, rnn-lstm, arch-positional-encoding]
---

# Cross-Attention

---

## Fundamental

### Self-Attention vs Cross-Attention

In **self-attention**, queries, keys, and values all come from the same sequence: $Q = K = V = X W_Q, X W_K, X W_V$.

In **cross-attention**, queries come from one sequence (the **target/decoder**) and keys/values come from a different sequence (the **source/encoder**):

$$Q = X_\text{dec} W_Q, \quad K = X_\text{enc} W_K, \quad V = X_\text{enc} W_V$$

$$\text{CrossAttention}(X_\text{dec}, X_\text{enc}) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Each decoder position attends over the full encoder output — the decoder "looks up" relevant information from the encoded source.

### Where Cross-Attention Appears

**Encoder-decoder transformers (T5, BART, original Transformer):** each decoder layer has two attention sublayers:
1. Causal self-attention over the decoded sequence so far
2. Cross-attention over the encoder's output

**Diffusion model U-Nets (DALL-E 2, Stable Diffusion):** the U-Net denoiser uses cross-attention to condition on text embeddings — each spatial position in the image attends over the text token representations.

**Multimodal models (Flamingo, BLIP-2):** cross-attention between visual tokens and text tokens enables visual question answering and image captioning.

---

## Intermediate

### Complexity

Cross-attention between a target sequence of length $T$ and source sequence of length $S$:
- Attention matrix: $T \times S$
- Compute: $O(TSd)$
- Memory: $O(TS)$ for the attention matrix

For machine translation with $T, S \approx 512$: $O(512 \times 512) = O(262K)$ entries — manageable.

For image-to-text with $T = 512$ text tokens, $S = 196$ patch tokens: $O(512 \times 196) = O(100K)$ — also fine.

Cross-attention scales better than self-attention when $S \ll T$ (small source, long target) but worse when both are long.

### KV Caching in Cross-Attention

In encoder-decoder architectures at inference time, the encoder runs once and the encoder output $X_\text{enc}$ is fixed. The cross-attention keys $K = X_\text{enc} W_K$ and values $V = X_\text{enc} W_V$ are computed once and cached for all decoder steps — no recomputation needed.

This is a free efficiency gain: unlike self-attention KV cache (which grows with decoded length), the cross-attention KV cache has fixed size $S \times d$ regardless of how many tokens are decoded.

### Cross-Attention in Perceiver / Perceiver IO

**Perceiver (Jaegle et al., 2021)** uses cross-attention to map a high-dimensional input (e.g., $S = 65536$ image pixels) to a small latent array ($T = 512$ latents). Cross-attention from latents to inputs collapses the large input into a fixed-size representation efficiently: $O(T \times S) = O(512 \times 65536)$ vs self-attention on inputs $O(S^2) = O(65536^2)$.

---

## Advanced

### Cross-Attention as Retrieval

Cross-attention can be interpreted as **soft retrieval**: the query vector selects which key-value pairs to attend to, retrieving a weighted combination of values. This is the basis for:

**Memory-augmented networks:** decoder attends over a large external memory bank via cross-attention.

**Retrieval-augmented generation (RAG):** retrieved document chunks are encoded and placed in cross-attention memory; the decoder generates conditioned on retrieved content.

**Diffusion transformers (DiT):** condition on class embeddings or text via cross-attention in each transformer block, rather than concatenating conditioning to the input.

### Grouped Cross-Attention

In very large encoder-decoder models, cross-attention heads can use grouped/multi-query attention (see [[grouped-query-attention]]) to reduce the KV cache size. Since the cross-attention KV cache is fixed and shared across decode steps, reducing its size reduces memory proportionally.

*See also: [[attention-mechanism]] · [[arch-kv-cache]] · [[vlm-architectures]] · [[vision-transformer]] · [[arch-positional-encoding]]*
