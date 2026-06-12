---
title: Pretraining Objectives Beyond MLM
tags: [pretraining-objectives, span-masking, prefix-lm, ul2, causal-lm, denoising]
aliases: [span masking, T5 pretraining, prefix LM, UL2, denoising objectives, masked span prediction]
difficulty: 2
status: complete
related: [bert-mlm, autoregressive-models, instruction-tuning, self-supervised-overview, attention-mechanism]
---

# Pretraining Objectives Beyond MLM

---

## Fundamental

### BERT-Style vs GPT-Style

The two dominant pretraining paradigms:

**BERT (MLM):** mask 15% of tokens, predict masked tokens bidirectionally. Good for understanding tasks (classification, NER, QA). Cannot generate text autoregressively.

**GPT (CLM — Causal Language Modeling):** predict the next token given all previous tokens. Naturally generative. Works with left-to-right attention (causal mask).

More nuanced objectives exist that combine benefits of both or introduce different inductive biases.

---

## Intermediate

### Span Masking (T5, SpanBERT)

**T5 (Raffel et al., 2020):** instead of masking individual tokens, mask contiguous **spans** of text and predict them as a sequence:

- Sample spans of mean length $L = 3$ tokens (geometric distribution)
- Mask 15% of tokens total
- Replace each masked span with a sentinel token `<extra_id_0>`, `<extra_id_1>`, ...
- Decoder predicts: `<extra_id_0> span1_tokens <extra_id_1> span2_tokens ...`

This encourages the model to generate coherent multi-token spans, better suited for downstream generation tasks than token-by-token MLM. T5 frames all NLP tasks as text-to-text (input text → output text).

**SpanBERT:** mask spans in BERT (no sentence-order prediction), add a span boundary objective. Better for span-level tasks (question answering, coreference).

### Prefix LM

**Prefix LM:** split each sequence into a prefix (fully visible, bidirectional attention) and a completion (autoregressive, causal attention):

```
[prefix tokens] [completion tokens]
     ↑                ↑
  bidirectional    causal (left-to-right)
  attention        attention
```

The model can use full context for the prefix (like BERT) but must generate the completion autoregressively (like GPT). Useful for few-shot learning (prefix = examples + instruction, completion = answer).

Used in: T5 (when used as a prefix LM), PaLM, Gemini.

### UL2: Unified Language Learner

**UL2 (Tay et al., 2022)** mixes three denoising objectives during pretraining:

1. **R-Denoiser (regular):** BERT-like short span masking (T5 default)
2. **S-Denoiser (sequential):** T5 masking but always at the end of the sequence (prefix LM style)
3. **X-Denoiser (extreme):** high masking ratio (50%) over long spans — forces full generative reconstruction

A mode token (`[R]`, `[S]`, `[X]`) prefixes each input to signal which denoising mode is active. At inference, prepend `[S2S]` for seq2seq tasks, `[NLG]` for generation, `[NLU]` for classification.

UL2 matches or outperforms T5 on understanding tasks and significantly improves generation quality — combining the best of both paradigms.

---

## Advanced

### ELECTRA: Replaced Token Detection

**ELECTRA (Clark et al., 2020):** instead of predicting masked tokens (hard, many valid choices), predict whether each token has been **replaced** by a generator:

1. Small generator (MLM): fills masked positions with plausible but wrong tokens
2. Large discriminator: predicts for **every** token whether it is original or replaced (binary classification)

**Efficiency advantage:** the discriminator trains on every token (binary classification everywhere), not just the 15% masked positions. ~4× more supervision signal per example → trains faster to the same performance as BERT.

ELECTRA-large matches RoBERTa-large quality with 1/4 the compute.

### Causal Masking Variants

**CLM with document boundaries:** concatenate documents with `<eos>` separators; do not allow attention across document boundaries. Standard for LLM pretraining.

**Fill-in-the-Middle (FIM, Bavarian et al., 2022):** rearrange spans so the model must predict a middle span given prefix and suffix:
```
<prefix> P <suffix> S <middle> M
```
25% of sequences are reformatted this way during GPT pretraining. Enables code completion models to infill missing code given surrounding context.

*See also: [[bert-mlm]] · [[autoregressive-models]] · [[instruction-tuning]] · [[self-supervised-overview]] · [[attention-mechanism]]*
