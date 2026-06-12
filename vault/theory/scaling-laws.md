---
title: Scaling Laws
tags: [scaling-laws, chinchilla, compute-optimal, power-laws, llm-training]
aliases: [Chinchilla scaling, compute-optimal training, neural scaling laws, Kaplan scaling]
difficulty: 2
status: complete
related: [autoregressive-models, bias-variance-double-descent, generalization-bounds, large-batch-training, loss-cross-entropy, emergent-abilities]
---

# Scaling Laws

---

## Fundamental

### What Scaling Laws Describe

**Neural scaling laws** describe how model performance (loss on a held-out set) changes as we scale:
- Number of parameters $N$
- Training dataset size $D$ (tokens)
- Compute budget $C$ (FLOPs)

Kaplan et al. (2020, OpenAI) found that for language models, loss follows **power laws** in each variable when the others are held large:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}$$

with $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$ (NLL on text). The constants $N_c, D_c$ depend on the architecture and data distribution.

**Combined law:** with compute $C \approx 6ND$ FLOPs for a dense transformer:

$$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

---

## Intermediate

### Chinchilla: Compute-Optimal Training

**Hoffman et al. (2022, DeepMind)** reran scaling experiments more carefully and found:

For a given compute budget $C$, the loss-minimizing allocation uses roughly equal scaling of $N$ and $D$:

$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

i.e., **tokens $\approx 20 \times$ parameters** is compute-optimal. Kaplan had underestimated the importance of data and recommended over-allocating parameters.

**Practical consequence:** GPT-3 (175B params, 300B tokens) was significantly undertrained. The Chinchilla-optimal model for the same compute (Chinchilla, 70B params, 1.4T tokens) outperformed GPT-3 on almost all benchmarks with 4× fewer parameters — far cheaper to serve.

| Model | Params | Tokens | Compute-optimal? |
|-------|--------|--------|-----------------|
| GPT-3 | 175B | 300B | No — over-parameterized |
| Chinchilla | 70B | 1.4T | Yes (at time) |
| LLaMA-1 70B | 70B | 1T | Close |
| LLaMA-3 8B | 8B | 15T | Over-trained (for inference efficiency) |

### Over-Training for Inference Efficiency

Chinchilla optimal minimizes loss per training FLOPs. But **inference cost** is proportional to model size. If a model will serve millions of users, it's worth over-training (more tokens, fewer parameters) to achieve a smaller deployed model with equal quality.

**LLaMA philosophy:** train a 7B model to Chinchilla-optimal quality for a 70B model by training on 10× more tokens. The resulting 7B model is "inference-optimal" — same loss, 10× cheaper to serve.

---

## Advanced

### Scaling Laws for Hyperparameters

Scaling laws also predict optimal hyperparameters:
- **Batch size:** optimal $B^* \propto L^{-1/\alpha_B}$ — larger batches optimal at larger scale
- **Learning rate:** optimal LR scales mildly with model size; larger models use slightly smaller LR
- **Architecture:** depth/width ratio and attention heads matter far less than $N$, $D$, $C$

The **critical batch size** (where linear LR scaling holds) grows with model size, explaining why large-model training uses massive global batch sizes (see [[large-batch-training]]).

### Beyond Language Models

Scaling laws hold empirically across:
- **Vision:** image generation and classification loss scale as power laws with compute
- **Code:** AlphaCode, Codex — similar exponents to language
- **Multimodal:** contrastive vision-language models (CLIP)
- **Protein folding, chess, math** — power laws observed in each domain

The exponents $\alpha$ vary by domain — domains with richer structure (code, math) tend to have more favorable (steeper) scaling exponents.

### Emergent Abilities and the Limits of Power Laws

Scaling laws are smooth and predictable in terms of loss. However, certain discrete capabilities appear suddenly at scale thresholds and cannot be extrapolated from small-scale experiments. See [[emergent-abilities]] for the debate around whether this is a genuine phase transition or a metric artifact.

*See also: [[autoregressive-models]] · [[large-batch-training]] · [[bias-variance-double-descent]] · [[emergent-abilities]] · [[loss-cross-entropy]]*
