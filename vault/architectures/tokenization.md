---
title: Tokenization
tags: [tokenization, bpe, wordpiece, sentencepiece, subword, vocabulary]
aliases: [BPE, byte pair encoding, WordPiece, SentencePiece, subword tokenization, tokenizer]
difficulty: 1
status: complete
related: [autoregressive-models, bert-mlm, word-embeddings, arch-positional-encoding, instruction-tuning, attention-mechanism]
---

# Tokenization

---

## Fundamental

### What Tokenization Does

A language model processes discrete **tokens** — integer IDs drawn from a fixed vocabulary. Tokenization converts raw text into token sequences.

**Three granularities:**

| Approach | Example: "unhappy" | Vocabulary size | OOV handling |
|----------|-------------------|-----------------|--------------|
| Word-level | ["unhappy"] | 50K–1M | UNK token for rare words |
| Character-level | ["u","n","h","a","p","p","y"] | ~256 | None (all chars known) |
| Subword (BPE etc.) | ["un", "happy"] | 32K–128K | Rare words split into pieces |

**Why subword wins:** character models have very long sequences (slow attention); word models fail on rare/new words and require enormous vocabularies. Subword tokenization balances both — common words stay whole, rare words split into meaningful parts.

### Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2016) starts from characters and iteratively merges the most frequent pair:

**Algorithm:**
1. Initialize vocabulary = all individual characters
2. Count all adjacent pair frequencies in training corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until vocabulary reaches target size $V$

**Example:** corpus contains "low", "lower", "newest", "widest"

After many merges: "low" stays whole (frequent), "est" becomes a token (appears in "newest", "widest"), "er" becomes a token ("lower"). The result: common words = single tokens, rare words = decomposed into common subwords.

**GPT-2 BPE:** starts from UTF-8 bytes (not characters), handles all Unicode without UNK. Vocabulary: 50,257 tokens. Never produces an unknown token.

---

## Intermediate

### WordPiece and SentencePiece

**WordPiece** (Schuster & Nakamura, 2012; used in BERT): similar to BPE but maximizes language model likelihood rather than merge frequency. Non-initial subwords are prefixed with `##` to mark continuation:

"playing" → ["play", "##ing"] — `##ing` signals "continuation of previous token."

BERT vocabulary: 30,522 tokens.

**SentencePiece** (Kudo & Richardson, 2018; used in T5, LLaMA): language-independent, treats spaces as regular characters (whitespace → `▁` token). Does not pre-tokenize on spaces, enabling languages like Chinese/Japanese without word boundaries.

**Unigram Language Model tokenization** (alternative to BPE): start with a large vocabulary, iteratively remove tokens that minimally reduce corpus log-likelihood. Probabilistic — can return the top-$k$ tokenizations with different probabilities. Used in SentencePiece with LLMs.

### Vocabulary Size Tradeoffs

| Aspect | Smaller vocabulary ($V = 8K$) | Larger vocabulary ($V = 128K$) |
|--------|------------------------------|-------------------------------|
| Sequence length | Longer (more tokens per sentence) | Shorter |
| Embedding matrix | Smaller | Larger (ties up parameters) |
| Rare word coverage | Poor | Better |
| Context window efficiency | Wasted on fragments | Better used |
| Multilingual coverage | Poor | Better |

**Typical choices:**
- GPT-2: 50K — optimized for English
- LLaMA 2: 32K — too small for multilingual, updated in LLaMA 3 to 128K
- GPT-4o: ~100K
- mT5: 250K — massively multilingual

The trend is toward larger vocabularies as models grow, because the embedding matrix cost ($V \times d$) is amortized over more tokens.

### Tokenization Artifacts and Pathologies

**Number handling:** "12345" → ["12", "345"] or ["1", "2", "3", "4", "5"] depending on tokenizer. Arithmetic is hard because digits are fragmented inconsistently. Newer tokenizers (GPT-4) group digits more uniformly.

**Trailing spaces:** " hello" and "hello" often produce different tokens. Space is attached to the following word (like `▁hello`), so tokenization depends on context before the word.

**Repeated characters:** "aaaaaa" in GPT-2 → many 1-character tokens (rare pattern in training). Causes surprisingly high token counts and poor generation for repeated-character strings.

**Case sensitivity:** BPE is usually case-sensitive, so "Cat" and "cat" may be different tokens. Case-insensitive tokenization loses information; case-sensitive doubles the effective vocabulary for capitalized variants.

**The "tokenization tax" on model capability:** models learn worse on languages with worse tokenization. Agglutinative languages (Finnish, Turkish) fragment into many tokens per word. A concept that takes 1 token in English may take 5+ tokens in another language — the context window holds proportionally less information. LLaMA 3's 128K vocabulary helped non-English performance significantly.

---

## Advanced

### Tokenization-Free and Byte-Level Models

**ByT5 (Xue et al., 2022):** operates directly on UTF-8 bytes, no tokenizer needed. Input "hello" → 5 byte tokens. Handles any language/script, robust to misspellings and noise. Tradeoff: very long sequences (5× longer than subword).

**MegaByte (Yu et al., 2023):** two-level transformer — a "global" model over patches of bytes and a "local" model within each patch. Achieves subword-model quality at byte level.

**Why byte-level matters:** tokenizers introduce biases (certain BPE merges make problems easier or harder), are language-specific, and require careful maintenance. Byte-level models are simpler and more uniform but computationally expensive.

### Tokenization and the LLM Training Pipeline

Tokenization is done **offline** before training:
1. Train tokenizer on a representative corpus (often separate from LLM training data)
2. Apply tokenizer to create integer-token sequences
3. Pack sequences into fixed-length chunks (e.g., 2048 tokens), concatenated with `<eos>` boundaries
4. Train language model on the packed token sequences

**Tokenizer-model mismatch:** when fine-tuning a model with a different tokenizer than pretraining, performance degrades. Extending vocabularies (e.g., adding domain-specific tokens) requires careful initialization of the new embedding rows (use average of similar tokens, not random).

**Token count ≠ word count:** pricing for API calls uses tokens. English text averages ~0.75 words per token (GPT-4 tokenizer). Code averages more tokens per logical unit. Chinese/Japanese text: highly compressed per character (~1–2 bytes per character vs ~3–5 UTF-8 bytes).

---

*See also: [[autoregressive-models]] · [[bert-mlm]] · [[word-embeddings]] · [[arch-positional-encoding]] · [[attention-mechanism]] · [[instruction-tuning]]*
