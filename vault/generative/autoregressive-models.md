---
title: Autoregressive Models (GPT-style)
tags: [autoregressive, gpt, causal-lm, next-token-prediction, tokenization, bpe, kv-cache, sampling-strategies]
aliases: [autoregressive model, GPT, causal language model, next token prediction, BPE, nucleus sampling, Chinchilla]
difficulty: 2
status: complete
related: [attention-mechanism, arch-positional-encoding, arch-kv-cache, bert-mlm, rlhf, lora-quantization, activation-gelu-swish]
---

# Autoregressive Models (GPT-style)

---

## Fundamental

An autoregressive model applies the **chain rule of probability** to factor the joint distribution over a sequence:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_1, \ldots, x_{t-1})$$

This is always exact — no approximation. Each token is predicted from all preceding tokens.

**Training objective** ([[loss-cross-entropy|next-token prediction]], causal language modeling):
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

This is fully **self-supervised** — labels are the tokens themselves, shifted by one position. Any text corpus is a valid training set. This is the key reason LLMs scale: internet-scale text provides almost unlimited self-supervised training signal.

### GPT Architecture

GPT is a stack of transformer decoder blocks with **causal (masked) [[attention-mechanism|self-attention]]**. Position $t$ cannot attend to positions $> t$:

$$A_{ij} = \begin{cases}\text{softmax}(QK^T/\sqrt{d_k})_{ij} & i \geq j \\ 0 & i < j\end{cases}$$

**Key design choices in GPT:**
- **Pre-norm:** LayerNorm before attention/FFN (more stable gradients at scale)
- **[[activation-gelu-swish|GELU/SwiGLU]]** in FFN layers (smoother than ReLU)
- **Learned [[arch-positional-encoding|positional embeddings]]** (GPT-2) or **RoPE** (LLaMA, Mistral, GPT-4)
- **Weight tying:** input embedding matrix = output unembedding matrix (reduces parameters, works well)
- **No bias** in attention/FFN weight matrices (marginal efficiency improvement)

**Training efficiency:** the causal mask allows all $T$ next-token predictions to be computed in a single forward pass (teacher forcing). This is what makes GPT training GPU-efficient despite being sequential at inference.

---

## Intermediate

### Tokenization: BPE

**Byte Pair Encoding (BPE):** start with byte-level vocabulary (256 tokens). Repeatedly find the most frequent adjacent symbol pair in the corpus and merge it into a new token. Repeat until vocabulary size is reached (GPT-2: 50,257 tokens; GPT-4: ~100k).

Common subwords get their own token; rare words are split into subword pieces. Byte-level BPE (GPT-2+) is lossless — any text is representable, never produces `<UNK>`.

**Tokenization effects:** English text ≈ 4 characters/token ≈ 0.75 words/token. Mathematical notation expands (each symbol = one token). Code varies widely.

### Sampling Strategies

| Strategy | Description | Effect |
|---|---|---|
| Greedy | Always argmax | Deterministic, repetitive |
| Temperature $\tau$ | Divide logits by $\tau$ before softmax | $\tau < 1$: sharp; $\tau > 1$: flat |
| Top-$k$ | Keep only top-$k$ tokens, renormalize | Prevents very unlikely tokens |
| Nucleus (top-$p$) | Keep smallest set with cumulative prob $\geq p$ | Adaptive: fewer tokens when distribution is sharp |

**Nucleus sampling** is more principled than top-$k$: when the model is confident, top-$p=0.9$ keeps just a few tokens; when the model is uncertain, it keeps many. In practice, temperature + top-$p$ is the standard combination for chat models.

### KV Cache Growth During Generation

Without a cache: generating token $t$ requires processing the entire prefix $x_1, \ldots, x_{t-1}$ from scratch — $O(T^2)$ total cost for a sequence of length $T$.

With [[arch-kv-cache|KV cache]]: keys and values from previous positions are stored. Each new token requires only $O(d^2)$ per new token. Memory cost: $O(L \cdot T \cdot d)$ for $L$ layers.

At scale: LLaMA-65B generating 2048 tokens uses ~8 GB for the KV cache in FP16. This memory cost — not compute — often limits maximum batch size and context length during inference.

---

## Advanced

### Scaling Laws and Chinchilla Optimality

GPT-style models follow empirical scaling laws (Kaplan et al., 2020):

$$\mathcal{L}(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \mathcal{L}_\infty$$

where $N$ = parameters, $D$ = training tokens, $\alpha_N \approx \alpha_D \approx 0.05$.

**Chinchilla (Hoffmann et al., 2022) correction:** for a fixed compute budget $C = 6ND$ (approximately), the optimal allocation is $D \approx 20 \times N$. GPT-3 (175B params, 300B tokens) was compute-suboptimal — it should have been trained with fewer parameters on more data. LLaMA follows Chinchilla scaling: LLaMA-7B trains on 1T tokens.

**Practical implication:** for a given inference cost budget (fixed parameter count), train on as many tokens as compute allows. Smaller, better-trained models outperform larger undertrained models at equal inference FLOPs.

### In-Context Learning as Implicit Gradient Descent

GPT-style models exhibit **in-context learning (ICL)**: they solve new tasks from a few examples in the prompt without gradient updates. Akyürek et al. (2022) showed that transformers with linear attention can exactly implement one step of gradient descent on the in-context examples. This suggests training on diverse tasks teaches the model to learn a fast in-weights optimization algorithm — not to memorize all tasks.

Implications: (1) larger models with more parameters have higher capacity for the implicit learning algorithm; (2) prompt engineering is essentially defining the task for the implicit optimizer; (3) ICL and fine-tuning lie on a continuum — fine-tuning is multiple gradient steps, ICL is zero explicit gradient steps.

### SwiGLU: Modern FFN Activation

Modern LLMs (LLaMA, PaLM, Mistral) replace the standard FFN with SwiGLU:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2), \quad \text{followed by } xW_3$$

The hidden dimension must be set to $8d/3$ (not the standard $4d$) to keep parameter count equal. SwiGLU empirically outperforms GELU+MLP by ~1 perplexity point at equivalent parameter budgets (Noam Shazeer, 2020).

---

*See also: [[attention-mechanism]] · [[arch-kv-cache]] · [[arch-positional-encoding]] · [[bert-mlm]] · [[rlhf]] · [[loss-cross-entropy]] · [[activation-gelu-swish]]*
