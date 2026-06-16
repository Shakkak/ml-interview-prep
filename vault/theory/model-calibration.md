---
title: Model Calibration
tags: [calibration, probability, evaluation, statistics, uncertainty]
aliases: [calibration, reliability diagram, ECE, temperature scaling, overconfidence]
difficulty: 2
status: complete
related: [evaluation-metrics-guide, statistical-inference-mle, activation-softmax, bayesian-inference, regularization-label-smoothing]
depends_on: [activation-softmax, statistical-inference-mle, bayesian-inference]
---

# Model Calibration

---

## Fundamental

A model is **perfectly calibrated** if, among all examples where the model predicts probability $p$, exactly a fraction $p$ of them actually belong to that class:

$$P(y = 1 \mid \hat{p}(x) = p) = p \quad \text{for all } p \in [0,1]$$

**Worked example.** A model says "80% confident" for 1,000 examples. A perfectly calibrated model is correct on ~800 of them. If it's only right on 600, it's **overconfident**. If it's right on 900, it's **underconfident**.

**Reliability diagram (calibration plot):**
1. Bin predictions by confidence: $[0, 0.1), [0.1, 0.2), \ldots, [0.9, 1.0]$
2. For each bin $b$: compute mean confidence $\bar{p}_b$ and actual accuracy $\text{acc}_b$
3. Plot $\bar{p}_b$ (x-axis) vs $\text{acc}_b$ (y-axis)

Perfect calibration = points on the diagonal $y = x$. Overconfident model (common for deep nets) = curve below the diagonal. Underconfident = curve above.

**Expected Calibration Error (ECE):**
$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{n} \left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

where $B$ = number of confidence bins, $B_b$ = set of examples in bin $b$, $|B_b|$ = bin count, $n$ = total examples, $\text{acc}(B_b)$ = fraction correct in bin $b$, $\text{conf}(B_b)$ = mean predicted confidence in bin $b$.

Weighted average absolute gap between confidence and accuracy across bins. Lower = better; 0 = perfect.

**Numerical example.** Predictions $[0.9, 0.8, 0.7, 0.6, 0.5]$, true labels $[1, 1, 0, 1, 0]$.

Bin $[0.8,1.0]$: 2 examples, mean conf 0.85, accuracy $2/2 = 1.0$, gap $= |1.0 - 0.85| = 0.15$.
Bin $[0.5,0.8)$: 3 examples, mean conf 0.6, accuracy $1/3 = 0.33$, gap $= |0.33 - 0.6| = 0.27$.

$\text{ECE} = \frac{2}{5}(0.15) + \frac{3}{5}(0.27) = 0.06 + 0.162 = 0.22$.

---

## Intermediate

### Why Deep Networks Are Overconfident

1. **[[activation-softmax|Softmax]] amplifies logit differences:** $\text{softmax}(c \cdot z) \to \text{one-hot}(\arg\max z)$ as $c \to \infty$. Large logits → overconfident probabilities.
2. **[[loss-cross-entropy|Cross-entropy]] incentivizes high confidence:** minimizing $-\log p_y$ pushes $p_y \to 1$.
3. **Modern architectures:** Guo et al. (2017) showed that BatchNorm, residual connections, and weight decay improve accuracy but worsen calibration — the gap between training-time confidence and test-time accuracy widens.
4. **Overfitting:** a model that memorizes training labels with high confidence extrapolates that confidence beyond what test performance warrants.

### Post-Hoc Calibration Methods

**Temperature scaling** — the simplest and most effective fix:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $z_i$ = raw logit for class $i$, $T > 1$ = temperature (learned scalar), and dividing by $T$ softens the distribution — larger $T$ → flatter probabilities → less overconfident.

Find $T > 1$ by minimizing NLL on a held-out validation set with $T$ as the only free parameter. Does not change accuracy (argmax is invariant to scaling). $T=1$ is the original softmax; $T>1$ flattens the distribution.

**Platt scaling:** train a logistic regression on model logits to predict true labels. Equivalent to temperature scaling with an additional bias term for binary classification.

**Isotonic regression:** fit a monotonic function from model scores to calibrated probabilities. Non-parametric — more flexible than Platt scaling but requires more validation data (typically ~1000+ examples).

**Histogram binning:** replace each score with the empirical accuracy of its bin. Simplest but coarsest — bins can be empty or sparse.

### Training-Time Calibration

**Label smoothing:** replace hard targets with soft targets $\tilde{y}_i = (1-\epsilon)y_i + \epsilon/K$. Prevents logits from diverging to $\pm\infty$, improving calibration. See [[regularization-label-smoothing]].

**Mixup:** train on convex combinations of examples and labels. Encourages linear interpolation between training examples → better calibrated confidence in the interpolated region.

**When calibration matters:**

| Application | Why |
|---|---|
| Medical diagnosis | Confidence level determines urgency of follow-up |
| Autonomous driving | Model must quantify its own uncertainty |
| Fraud detection | Threshold tuning requires reliable probabilities |
| [[bayesian-inference|Bayesian]] pipelines | Downstream inference requires calibrated likelihoods |

---

## Advanced

### Calibration vs Sharpness and the ECE Limitations

**Proper scoring rules** unify calibration and sharpness. The **Brier score** $\mathbb{E}[(p - y)^2]$ decomposes as:

$$\text{Brier} = \underbrace{\text{Calibration}}_{\mathbb{E}[(\bar{p}_b - \text{acc}_b)^2]} - \underbrace{\text{Resolution}}_{\mathbb{E}[(\text{acc}_b - \bar{y})^2]} + \underbrace{\text{Uncertainty}}_{\bar{y}(1-\bar{y})}$$

A model that always predicts the base rate $\bar{y}$ is perfectly calibrated but has zero resolution — it provides no useful information. A model should be calibrated **and** sharp (confident predictions that are correct). Temperature scaling often improves ECE but reduces sharpness (makes predictions more uncertain), trading one for the other.

**ECE is biased by bin choice.** With 10 equal-width bins, many bins may be empty for highly confident models, making ECE an unreliable estimate. Alternatives: (1) **Adaptive calibration error (ACE)** — equal-mass bins; (2) **Kernel-based calibration** (Widmann et al., 2019) — estimates calibration as a continuous function; (3) **RECE (Reliability Diagram ECE)** — uses all bin widths dynamically.

### Multiclass and Class-Conditional Calibration

Temperature scaling extends directly to multiclass — still one scalar $T$ applied to all logits. However, aggregate calibration can hide class-conditional miscalibration.

**Class-wise ECE:** compute ECE separately per class. A model can have ECE = 0.02 in aggregate but ECE = 0.15 for a rare class (e.g., a rare disease) — the aggregate metric is misleading when class frequencies are imbalanced.

**Canonical tensor calibration (Kull et al., 2019):** a general framework for multiclass calibration using matrix scaling (per-class temperature + off-diagonal adjustments). More powerful than scalar temperature scaling but requires more parameters and validation data.

### Conformal Prediction and Distribution-Free Calibration

Temperature scaling guarantees calibration only when the validation set is i.i.d. with the test set. **Conformal prediction** (Vovk et al., 2005) provides **distribution-free, finite-sample calibration guarantees**:

For a user-specified error rate $\alpha$, conformal prediction produces a prediction set $\mathcal{C}(x)$ satisfying:

$$P(y_{\text{test}} \in \mathcal{C}(x_{\text{test}})) \geq 1 - \alpha$$

for any test distribution, with no distributional assumptions. The coverage guarantee holds exactly (not just asymptotically).

**Split conformal prediction** (Papadopoulos et al., 2002): use a calibration set to compute nonconformity scores (e.g., $1 - \hat{p}_y$ for classifier outputs); set the threshold at the $(1-\alpha)(1+1/n_{\text{cal}})$ quantile of calibration scores. At test time, the prediction set includes all classes with score below the threshold.

**RAPS (Angelopoulos et al., 2021):** extends conformal prediction to produce small prediction sets (not just valid ones) via a regularized adaptive procedure. Used in production medical AI systems where valid coverage with minimal set size is required.

---

## Links

- [[activation-softmax]] — the softmax output is a probability distribution; well-calibrated models have softmax confidence = empirical accuracy, but modern networks are systematically overconfident
- [[statistical-inference-mle]] — cross-entropy minimization (MLE) maximizes likelihood but doesn't directly optimize calibration; NLL minimization and calibration align in the limit but diverge in practice
- [[bayesian-inference]] — Bayesian neural networks are better calibrated by construction; the posterior over weights marginalizes out uncertainty rather than collapsing to a point estimate
- [[evaluation-metrics-guide]] — ECE (Expected Calibration Error) measures calibration by binning confidence scores; a perfectly calibrated model has ECE = 0
- [[regularization-label-smoothing]] — label smoothing replaces one-hot targets with $(1-\epsilon)$ on the correct class and $\epsilon/(K-1)$ on others; this prevents the model from pushing logits to $\pm\infty$ and improves calibration
- [[loss-cross-entropy]] — calibrated cross-entropy requires the model's confidence to match empirical frequencies; temperature scaling $T > 1$ divides logits to soften the softmax and reduce overconfidence
