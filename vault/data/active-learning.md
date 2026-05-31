---
title: Active Learning
tags: [active-learning, uncertainty-sampling, query-by-committee, annotation-efficiency, cold-start, bald]
aliases: [active learning, uncertainty sampling, query by committee, core-set selection, batch active learning]
difficulty: 2
status: complete
related: [model-calibration, regularization-dropout, anomaly-detection, evaluation-metrics-guide, bayesian-inference]
---

# Active Learning

---

## Fundamental

Labels are expensive; unlabeled data is cheap. A model trained on 1,000 carefully chosen labeled examples regularly outperforms one trained on 10,000 randomly chosen ones. **Active learning** closes this gap by selecting which examples to label next.

**The pool-based active learning loop:**
```
1. Train model on small labeled set L
2. Query: score all examples in unlabeled pool U by informativeness
3. Annotator labels the top-b selected examples
4. Add to L; retrain (or fine-tune) model
5. Repeat until annotation budget exhausted
```

### Uncertainty Sampling Strategies

All three strategies below select the example the current model is most uncertain about:

**Least confident (LC):** select the example where the top predicted class has lowest probability:
$$x^* = \arg\max_{x \in U}\left(1 - \max_c p(y=c \mid x)\right)$$

**Margin sampling:** select where the gap between the top two classes is smallest:
$$x^* = \arg\min_{x \in U}\left(p(y=c_1 \mid x) - p(y=c_2 \mid x)\right)$$

**Entropy sampling:** select the example with highest predicted entropy:
$$x^* = \arg\max_{x \in U} -\sum_c p(y=c \mid x)\log p(y=c \mid x)$$

For binary classification all three are equivalent. For many classes, entropy is preferred — it uses the full predicted distribution rather than just the top one or two classes.

---

## Intermediate

### Query-by-Committee (QBC)

Train $K$ diverse models on the current labeled set. Select examples where the committee most disagrees:

**Vote entropy:**
$$x^* = \arg\max_{x \in U} -\sum_c \frac{V(c,x)}{K}\log\frac{V(c,x)}{K}$$

where $V(c,x)$ = number of committee members predicting class $c$ for $x$.

**Why diversity matters:** if all members agree, the example is already well-understood regardless of confidence. Diversity is achieved via different random seeds, bagging (train each model on a bootstrap sample), or MC Dropout (each stochastic forward pass = one committee member).

**Advantage over single-model uncertainty:** a committee is less sensitive to mislabeled examples — a single model can be confused by a noisy label and focus on it repeatedly; a committee is likely to disagree on genuinely ambiguous examples, not on noise.

### Core-Set Selection

Uncertainty-based methods select hard examples but may leave large "easy" regions of input space unrepresented. **Core-set** (Sener & Savarese, 2018) selects examples that cover the full unlabeled pool:

$$S^* = \arg\min_{S:\,|S|=b} \max_{x \in U}\min_{x' \in S} d(x, x')$$

This is a **$k$-center problem** — minimize the maximum distance from any unlabeled point to its nearest labeled neighbor. Solved greedily: at each step, add the unlabeled point farthest from any currently selected point (approximately optimal).

Core-set is computed in **embedding space** from a pretrained encoder, not in raw input space. Distances in raw pixels are not meaningful for informativeness.

### Cold Start and Stopping

**Cold start:** at round 1, no labeled data exists to train a model. Solutions: (1) random seed of 50–200 examples (usually sufficient), (2) cluster unlabeled data in feature space and pick one centroid per cluster (ensures diverse initialization).

**Stopping criteria:** maximum annotation budget; validation performance plateau (change < $\epsilon$ for $k$ consecutive cycles); or when mean pool entropy drops below a threshold (the model has become confident about the entire pool).

---

## Advanced

### BALD: Bayesian Active Learning by Disagreement

Uncertainty sampling (entropy) selects examples that are uncertain for the model's current parameters. But some uncertain examples are uncertain because of noise — labeling them will not help. **BALD** (Houlsby et al., 2011) selects examples that maximize the **mutual information** between the prediction and the model's parameters:

$$x^* = \arg\max_{x \in U} I(y; \theta \mid x, \mathcal{D}) = H[y \mid x, \mathcal{D}] - \mathbb{E}_{p(\theta \mid \mathcal{D})}\left[H[y \mid x, \theta]\right]$$

The first term is the predictive entropy (total uncertainty). The second is the expected entropy under the posterior (aleatoric/irreducible uncertainty). Their difference is **epistemic uncertainty** — uncertainty that can be reduced by observing labels.

BALD implemented with MC Dropout: run $K$ stochastic forward passes, compute sample mean entropy (first term) minus mean of per-sample entropies (second term). Selects examples where the model is globally uncertain but individual samples disagree — the hallmark of epistemic uncertainty.

### Batch Active Learning and Diversity

Querying the top-$b$ uncertain examples independently is suboptimal: they cluster in the same uncertain region. Three fixes:

1. **K-means on uncertain examples:** take the top-$kb$ uncertain examples, run k-means clustering in embedding space, return the $b$ cluster centers.
2. **Core-set restricted to uncertain examples:** run the k-center greedy algorithm on the top uncertain subset — diverse + uncertain.
3. **BADGE (Ash et al., 2020):** embed each example by its gradient with respect to the last-layer parameters:
   $$g_x = \nabla_\theta \mathcal{L}(\hat{y}_x, f_\theta(x))$$
   Run $k$-means++ on $\{g_x\}_{x \in U}$. The magnitude of $g_x$ encodes uncertainty; the direction encodes which part of the model would change. Diverse gradient directions → diverse model updates.

### Connection to Semi-Supervised and Domain Adaptation

Active learning selects *which* unlabeled examples to label; semi-supervised learning uses *all* unlabeled examples as signal (consistency regularization, pseudo-labels). They are complementary — active selection + semi-supervised training is strictly stronger than either alone.

In **domain adaptation**, the target domain has cheap unlabeled data but expensive labels. Active learning on the target domain selects the most informative $b$ target examples to label per cycle, rather than labeling a random subset. This is the standard protocol for practical domain adaptation and reduces annotation cost by 5–20× compared to random sampling.

---

*See also: [[model-calibration]] · [[regularization-dropout]] · [[bayesian-inference]] · [[evaluation-metrics-guide]]*
