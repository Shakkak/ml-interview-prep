---
title: Mixture of Experts
tags: [mixture-of-experts, moe, sparse-moe, routing, switch-transformer, mixtral]
aliases: [MoE, sparse MoE, expert routing, top-k gating, Switch Transformer, Mixtral]
difficulty: 3
status: complete
depends_on: [attention-mechanism, normalization-layers, linear-algebra-fundamentals]
related: [attention-mechanism, arch-kv-cache, flash-attention, autoregressive-models, normalization-layers, lora-quantization]
---

# Mixture of Experts

---

## Fundamental

### Core Idea

**Mixture of Experts (MoE)** replaces a single dense feed-forward network with $E$ parallel "expert" networks, and a **router** that selects which experts process each token. Only $k \ll E$ experts activate per token (sparse activation), so total computation is proportional to $k \cdot \text{expert\_size}$ rather than $E \cdot \text{expert\_size}$.

**Sparsity = scale without proportional compute:** a 47B-parameter Mixtral 8×7B model has the inference cost of a ~13B dense model (only 2 of 8 experts active), but the capacity (memory) of 47B.

### Architecture

In a transformer, the MoE layer replaces the FFN sublayer:

```
Token embedding → Router → [top-k expert indices + scores]
                         → Expert_1(x) · score_1
                         + Expert_2(x) · score_2
                         → combined output
```

**Router:** a learned linear layer $W_r \in \mathbb{R}^{d \times E}$ computing logits for each expert, then softmax + top-$k$ selection:

$$g(x) = \text{softmax}(\text{TopK}(W_r x, k))$$

where $x \in \mathbb{R}^d$ = the token's hidden state, $W_r \in \mathbb{R}^{d \times E}$ = the router weight matrix mapping token representations to per-expert logits, $E$ = total number of experts, $\text{TopK}(\cdot, k)$ = sets all but the $k$ largest logits to $-\infty$ before softmax (so only $k$ experts get non-zero weights), and $g(x) \in \mathbb{R}^E$ = the sparse gating vector (at most $k$ non-zero entries). Tokens routed to the same expert are batched together for efficiency.

---

## Intermediate

### Load Balancing

Naive routing collapses — the router learns to always use the same $k$ experts (popularity bias). This leaves most experts undertrained.

**Auxiliary load balancing loss** (Switch Transformer, Fedus et al., 2022):

$$\mathcal{L}_\text{aux} = \alpha \cdot E \sum_{i=1}^E f_i \cdot p_i$$

where $f_i$ = fraction of tokens routed to expert $i$ (non-differentiable), $p_i$ = average router probability for expert $i$ (differentiable). Minimizing $\mathcal{L}_\text{aux}$ encourages uniform routing while keeping gradients flowing.

**Expert capacity:** each expert has a fixed token budget per batch (capacity factor $C$). Tokens that overflow an expert are dropped or passed through a residual. $C = 1.25$ is typical — some slack above uniform assignment.

**Z-loss (GLaM, ST-MoE):** $\mathcal{L}_z = \beta \cdot \frac{1}{B}\sum_i (\log \sum_j e^{x_j})^2$ penalizes large router logits, improving stability.

### Top-1 vs Top-2 Routing

**Top-1 (Switch Transformer):** each token goes to exactly 1 expert. Maximum sparsity, simplest routing, but highest variance — tokens can get very different processing depending on the single expert chosen.

**Top-2 (GShard, Mixtral):** each token uses 2 experts, weighted by router scores. More stable gradients, better quality, 2× compute vs top-1.

| Model | Experts | $k$ | Parameters |
|-------|---------|-----|-----------|
| Switch Transformer | 2048 | 1 | 1.6T (only ~7B active) |
| GShard | 2048 | 2 | 600B |
| Mixtral 8×7B | 8 | 2 | 47B (13B active) |
| Mixtral 8×22B | 8 | 2 | 141B (39B active) |
| GPT-4 (estimated) | ~16 | 2 | ~1.8T |

---

## Advanced

### Expert Parallelism

MoE models need a new parallelism dimension: **expert parallelism** places different experts on different devices. During a forward pass:
1. Each device computes router decisions for its local tokens
2. All-to-all communication: send tokens to the devices hosting their assigned experts
3. Each device runs its local experts
4. All-to-all again: return processed tokens to original devices

Combined with tensor and pipeline parallelism, this allows training trillion-parameter MoE models. The all-to-all is the main communication bottleneck.

### Fine-Grained vs Coarse-Grained Experts

**Coarse-grained (standard):** few large experts (e.g., 8 experts of size 7B). Experts specialize strongly — routing becomes semantically meaningful.

**Fine-grained:** many small experts (e.g., 64 experts of size 500M), higher $k$. Better load balancing, less routing variance, but experts specialize less.

**DeepSeekMoE:** uses a mix of shared experts (always active) + fine-grained experts (routed), avoiding expert collapse on common knowledge while routing only specialized information.

### When MoE Outperforms Dense Models

MoE wins when:
- Parameter count (capacity) matters more than compute budget
- Memory bandwidth is cheap relative to arithmetic throughput
- Training steps are limited (MoE learns faster per step than same-compute dense)

MoE loses when:
- Serving latency is critical (all experts must be loaded into memory)
- Training is communication-bottlenecked
- Very small batch sizes (routing overhead dominates)

## Links

- [[attention-mechanism]] — MoE replaces the dense FFN sublayer while keeping attention unchanged; the attention mechanism sees all tokens, FFN experts see only routed subsets
- [[normalization-layers]] — load balancing in MoE is sensitive to activation scale; layer normalization before routing gates stabilizes the routing distribution
- [[linear-algebra-fundamentals]] — the router is a learned linear projection to $E$ experts; top-$k$ selection is an argmax operation over the dot products
- [[autoregressive-models]] — Mixtral, Switch Transformer, and Gemini MoE use sparse MoE layers; they achieve GPT-scale quality at 2–4× lower FLOPs per token
- [[arch-kv-cache]] — MoE increases model capacity without increasing FLOPs, but it does increase memory: all expert weights must reside in GPU VRAM or be offloaded
- [[lora-quantization]] — LoRA fine-tuning of MoE models can target just the router or the expert FFNs; expert-specific LoRA adapters specialize individual experts
