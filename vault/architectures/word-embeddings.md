---
title: Word Embeddings
tags: [word-embeddings, word2vec, glove, fasttext, embeddings, nlp]
aliases: [word2vec, GloVe, FastText, word vectors, distributed representations, skip-gram, CBOW]
difficulty: 1
status: complete
related: [tokenization, self-supervised-overview, contrastive-learning, attention-mechanism, bert-mlm, optimal-transport, loss-nt-xent]
---

# Word Embeddings

---

## Fundamental

### The Distributional Hypothesis

**"You shall know a word by the company it keeps"** (Firth, 1957). Words with similar meanings appear in similar contexts — "dog" and "cat" appear near "pet", "food", "vet". This means word meaning can be captured from distributional statistics.

**From one-hot to dense vectors:** a vocabulary of $V = 50,000$ words with one-hot encoding gives sparse $V$-dimensional vectors with no notion of similarity. Word embeddings map words to dense vectors in $\mathbb{R}^d$ ($d = 100$–$300$) where similar words map to nearby vectors. This captures semantic and syntactic relationships in the geometry.

### Word2Vec

Word2Vec (Mikolov et al., 2013) has two architectures:

**Skip-gram:** given a center word, predict surrounding context words. Maximizes:
$$\mathcal{L} = \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$$

For a context window $c$ and $T$ total words. The predicted probability uses a softmax over all $V$ words — expensive.

**CBOW (Continuous Bag of Words):** given context words, predict the center word. Averages context embeddings and predicts the target. Faster than skip-gram; skip-gram is better for infrequent words.

**Negative sampling** makes training tractable: replace the softmax over $V$ words with a binary classifier that distinguishes true context pairs from $k$ random (negative) samples:

$$\mathcal{L}_\text{neg} = \log\sigma(\mathbf{w}_o^\top \mathbf{w}_c) + \sum_{k=1}^K \log\sigma(-\mathbf{w}_k^\top \mathbf{w}_c)$$

$k = 5$–$20$ negatives is sufficient. This makes training linear in the context window size.

**The famous analogy:** $\text{vec}(\text{"king"}) - \text{vec}(\text{"man"}) + \text{vec}(\text{"woman"}) \approx \text{vec}(\text{"queen"})$. Linear structure in embedding space captures semantic relationships.

---

## Intermediate

### GloVe: Global Co-occurrence Vectors

GloVe (Pennington et al., 2014) uses **global word co-occurrence statistics** directly rather than local context windows. Build a co-occurrence matrix $X$ where $X_{ij}$ = how often word $j$ appears in the context of word $i$.

**Objective:** learn vectors $\mathbf{w}_i, \tilde{\mathbf{w}}_j$ and biases $b_i, \tilde{b}_j$ such that:
$$\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log X_{ij}$$

Weighted by $f(X_{ij})$ — a function that reduces weight for very frequent pairs (which are less informative). 

**Key insight:** the ratio $\log(X_{ik}/X_{jk})$ encodes the relative association of words $i$ and $j$ with word $k$. GloVe's objective makes $\mathbf{w}_i - \mathbf{w}_j$ capture this ratio — directly encoding relational meaning as vector differences.

**GloVe vs Word2Vec:**
- GloVe uses global statistics; Word2Vec uses local windows
- Both produce similar quality embeddings in practice
- GloVe is easier to parallelize (precompute $X$); Word2Vec needs online training
- Skip-gram with negative sampling and GloVe are closely related mathematically — both factorize variants of the shifted PMI matrix

### FastText: Subword Embeddings

FastText (Bojanowski et al., 2017) represents each word as the sum of its character $n$-gram embeddings:

"eating" → embeddings of {"eat", "ati", "tin", "ing", "<ea", "ng>"} (character trigrams + edge markers)

The word vector = sum of all subword vectors. **Benefits:**
- Handles rare and out-of-vocabulary words — "untrainable" can be represented via subwords even if never seen in training
- Morphologically rich languages — Finnish, Turkish, Arabic benefit greatly since inflected forms share subwords
- Better embeddings for morphological variants ("run", "runs", "running" share subword structure)

---

## Advanced

### Static vs Contextual Embeddings

Word2Vec/GloVe produce **static** embeddings — one vector per word regardless of context. "bank" has one embedding even though it has different meanings in "river bank" vs "bank account."

**Contextual embeddings** from BERT (see [[bert-mlm]]) and GPT produce different vectors for each token depending on its full context. The representation of "bank" in "river bank" is geometrically distinct from "bank account." ELMo (Peters et al., 2018) was the first large contextual model — it extracts embeddings from a bidirectional LSTM.

**Why static embeddings still matter:**
1. Much smaller ($V \times d$ parameters vs billions) — deployable anywhere
2. Precomputable at corpus level — $O(1)$ per word at inference
3. Interpretable geometry (analogy structure, clustering)
4. Strong baselines for classification and retrieval

### Evaluation: Intrinsic vs Extrinsic

**Intrinsic:** word similarity/analogy benchmarks
- Word similarity: cosine similarity vs human judgments (WordSim-353, SimLex-999)
- Analogy: "man:woman::king:?" nearest neighbor = "queen" (word2vec's famous demo)
- Directly measures the geometric properties of the embedding space

**Extrinsic:** downstream task performance (NER, sentiment, QA)
- Plugging embeddings into a classifier and measuring F1/accuracy
- More meaningful for applications; correlates poorly with intrinsic metrics in practice

**Geometric properties explored:**
- PCA/UMAP visualization: language families, semantic clusters, syntactic categories separate
- Gender bias: $\text{vec}(\text{doctor}) - \text{vec}(\text{nurse}) \approx \text{vec}(\text{man}) - \text{vec}(\text{woman})$ — problematic bias encoded in training data statistics
- Debiasing: project out the gender direction or train with fairness constraints

### Sentence and Document Embeddings

**Sentence-BERT (SBERT):** fine-tune BERT with a siamese/triplet network on NLI data to produce sentence embeddings where semantic similarity = cosine similarity. Used for semantic search and clustering (see [[contrastive-learning]]).

**Optimal transport distances** (see [[optimal-transport]]): Word Mover's Distance (WMD) measures the minimum "cost" of moving the word embedding distribution of one sentence to match another — a more expressive similarity metric than cosine on averaged embeddings.

---

*See also: [[tokenization]] · [[bert-mlm]] · [[contrastive-learning]] · [[attention-mechanism]] · [[self-supervised-overview]] · [[optimal-transport]]*
