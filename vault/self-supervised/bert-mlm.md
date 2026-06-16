---
title: BERT and Masked Language Modeling
tags: [bert, masked-language-modeling, mlm, nsp, bidirectional-transformer, pretraining, fine-tuning]
aliases: [BERT, masked language modeling, MLM, NSP, next sentence prediction, bidirectional encoder]
difficulty: 2
status: complete
related: [attention-mechanism, self-supervised-overview, contrastive-learning, rlhf, lora-quantization, arch-positional-encoding]
depends_on: [attention-mechanism, loss-cross-entropy, arch-positional-encoding]
---

# BERT and Masked Language Modeling

---

## Fundamental

Before BERT (Devlin et al., 2019), language [[self-supervised-overview|pretraining]] used **causal (left-to-right) language models** (GPT). These have a fundamental limitation for understanding tasks: when processing token $t$, only tokens $1, \ldots, t-1$ are visible. For understanding tasks — classification, NER, QA — full bidirectional context is more informative. "The bank was steep" means something different from "The bank approved the loan," and only seeing later context resolves the ambiguity.

**BERT's core insight:** use a masked prediction objective that allows bidirectional [[attention-mechanism|attention]] while still providing a non-trivial learning signal.

**The MLM objective:** randomly mask some tokens in the input; train the model to predict the original tokens from the bidirectional context.

**The 15% masking rule:** select 15% of tokens at random. Of those selected:
- **80%** are replaced with the special `[MASK]` token.
- **10%** are replaced with a random token from the vocabulary.
- **10%** are left unchanged (the original token).

**Why not just `[MASK]` for all selected tokens?** If the model only ever sees `[MASK]` at prediction positions, it learns to attend differently to masked vs unmasked tokens. At fine-tuning time, there are no `[MASK]` tokens — a train/inference mismatch. The 10%/10% rule forces the model to maintain useful representations for every token even when it is not masked.

**Loss:** [[loss-cross-entropy|cross-entropy]] over the 15% masked positions only:
$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid \tilde{x}; \theta)$$

where $x_i$ = the original token at masked position $i$, $\tilde{x}$ = the corrupted input sequence (with some tokens replaced by `[MASK]`, random tokens, or left unchanged), and $\theta$ = model parameters. The sum is over only the 15% selected positions, not all tokens.

**Architecture:** BERT is a standard Transformer encoder (no decoder). Every token attends to every other token — no causal mask. Special tokens: `[CLS]` at position 0 (classification tasks), `[SEP]` to separate sentences.

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

**WordPiece tokenization:** subword algorithm — common words get their own token; rare words are split ("unhappiness" → ["un", "##happy", "##ness"]). Balances vocabulary size (30,000 tokens) and sequence length.

**Fine-tuning heads** are added per task:
- Sequence classification: feed `[CLS]` output to a linear layer: $y = W \cdot h_{\text{[CLS]}} + b$
- Token classification (NER): feed each token's output to a linear layer independently
- Extractive QA (SQuAD): predict start and end span positions: $P_{\text{start}}(i) = \text{softmax}(W_s \cdot h_i)$
- Sentence pair tasks: encode both with `[SEP]`, feed `[CLS]` to a classifier

Fine-tuning typically takes 2–4 epochs, learning rate ~2e-5, with pretrained weights updated (not frozen).

---

## Intermediate

**Next Sentence Prediction (NSP):** BERT was also trained on a binary task: given sentences A and B, predict whether B follows A in the corpus (50%) or is random (50%). Purpose: learn inter-sentence relationships for QA and NLI.

**Why NSP was dropped:** RoBERTa (Liu et al., 2019) showed NSP slightly hurts performance. The task is too easy — random negatives differ in topic and can be detected from surface statistics without learning semantic coherence. RoBERTa removed NSP, trained longer with larger batches and dynamic masking (mask changes each epoch), and improved most benchmarks. ALBERT replaced NSP with sentence order prediction (SOP), a harder same-topic pairing task.

**Bidirectional vs Causal tradeoffs:**

| | Bidirectional (BERT) | Causal (GPT) |
|--|---------------------|-------------|
| Training objective | Masked token prediction (MLM) | Next token prediction (LM) |
| Context | Both left and right | Left only |
| Best for | Understanding (classification, NER, QA) | Generation (text, code, chat) |
| Fine-tuning | Task-specific head on [CLS] or tokens | Continue autoregressive generation |
| Example models | BERT, RoBERTa, ALBERT, DeBERTa | GPT-2/3/4, LLaMA, Mistral |

The encoder-decoder architecture (T5, BART) combines both: bidirectional encoder for understanding, causal decoder for generation.

**Key variants:**

**RoBERTa (2019):** removes NSP, trains on more data with larger batches and longer sequences, uses dynamic masking. Outperforms BERT on most benchmarks.

**ALBERT (2019):** parameter-efficient BERT — factorizes the embedding matrix ($V \times H$ becomes $V \times E + E \times H$ with $E \ll H$) and shares parameters across layers. 18× fewer parameters than BERT-large with comparable performance.

**DistilBERT:** [[knowledge-distillation|knowledge distillation]] of BERT-base — 40% smaller, 60% faster, retains 97% of BERT's performance.

**Practical fine-tuning details:** learning rate warmup avoids early catastrophic updates; layer-wise learning rate decay (lower LR for early layers) often helps because early representations change less during fine-tuning. For short-text tasks, pooling strategies other than `[CLS]` (mean pooling over all tokens) can yield better sentence embeddings.

---

## Advanced

**DeBERTa (He et al., 2020):** disentangled attention separates content and [[arch-positional-encoding|position embeddings]] entirely. Each token is represented by two vectors: one for content, one for relative position. The attention score between tokens $i$ and $j$ is computed as:

$$A_{i,j} = \langle \tilde{H}_i, \tilde{H}_j \rangle + \langle \tilde{H}_i, \tilde{P}_{i|j} \rangle + \langle \tilde{P}_{j|i}, \tilde{H}_j \rangle$$

where $\tilde{P}_{i|j}$ encodes the relative position of $j$ from $i$. This separates "what two tokens are" from "how far apart they are," producing state-of-the-art results on understanding benchmarks.

**The train-inference mismatch in depth:** the 80/10/10 rule is a pragmatic fix for a fundamental tension. At pretraining, `[MASK]` creates an artificial token not seen at fine-tuning time, causing a distribution mismatch. The 10% random token replacement forces the model to maintain contextual representations for all positions — if token $i$ could be replaced with a random token, the model cannot ignore token $i$ during encoding. This is analogous to denoising autoencoders: corrupting inputs forces robust internal representations.

**Why MLM learns better representations than causal LM for understanding:** mutual information theory perspective — MLM maximizes $I(x_i ; \tilde{x})$ where $\tilde{x}$ is the masked context. Since both left and right context is available, the model captures richer conditional distributions. For a word like "bank" in "The bank was steep," bidirectional context resolves the ambiguity that causal LM cannot, since the causal model sees only "The bank" before the target.

**SimCSE (Gao et al., 2021):** uses BERT as backbone for sentence-level [[contrastive-learning]]. The positive pair is obtained by feeding the same sentence twice through BERT with different dropout masks — the two forward passes produce different representations due to dropout noise. This simple approach achieves state-of-the-art on semantic textual similarity tasks, demonstrating that BERT's representations benefit substantially from contrastive fine-tuning without any new data.

**SpanBERT (Joshi et al., 2020):** replaces token-level masking with span masking (mask contiguous spans of 1–10 tokens). Additionally adds a span boundary objective: predict each masked token from the boundary token representations alone, not internal span context. This forces the model to encode span-level information, substantially improving on extractive QA and coreference resolution tasks.

**Scaling and efficiency research:** BERT's $O(n^2)$ attention limits sequence length. Longformer (Beltagy et al., 2020) uses sliding window attention (local) plus global attention on special tokens, enabling sequences of 4096+ tokens with $O(n)$ complexity. BigBird (Zaheer et al., 2020) proves theoretically that sparse attention is a universal approximator of full attention, justifying these approximations formally.

---

## Links

- [[attention-mechanism]] — BERT uses bidirectional self-attention: every token attends to all others simultaneously; this is the key difference from autoregressive (causal) transformers
- [[loss-cross-entropy]] — MLM is trained with cross-entropy on the 15% masked tokens; NSP uses binary cross-entropy on the sentence-pair classification head
- [[arch-positional-encoding]] — BERT uses learned absolute position embeddings; without them the transformer is permutation-invariant and cannot model word order
- [[self-supervised-overview]] — BERT pioneered masked-prediction SSL for NLP; the masking strategy (15% tokens, 80% [MASK] / 10% random / 10% original) reduces train-test mismatch
- [[contrastive-learning]] — BERT is a reconstruction-based SSL method, not contrastive; it predicts masked tokens rather than contrasting positive/negative pairs
- [[rlhf]] — RLHF fine-tunes autoregressive models (not BERT); BERT-style bidirectional encoders are used as reward models in RLHF pipelines
- [[knowledge-distillation]] — DistilBERT distills BERT into a 40% smaller model with 97% of performance; the soft labels from BERT's logits are the distillation targets
