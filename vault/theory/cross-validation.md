---
title: Cross-Validation
tags: [statistics, model-selection, evaluation, generalization, hyperparameter-tuning, resampling]
aliases: [k-fold cross-validation, k-fold CV, leave-one-out cross-validation, LOOCV, nested cross-validation, stratified k-fold, CV]
difficulty: 1
status: complete
related: [bias-variance-double-descent, bootstrap, ensemble-methods, decision-trees, regularization-weight-decay]
depends_on: [statistical-inference-mle, bias-variance-double-descent]
---

# Cross-Validation

---

## Fundamental

**Definition.** Cross-validation (CV) estimates how well a model will generalize to unseen
data by repeatedly splitting the available data into a **training** portion and a **held-out
validation** portion, fitting on the former and scoring on the latter, and averaging the
scores across splits. It answers the question a single train/test split answers only noisily:
*"how would this model do on data it hasn't seen?"* — while using *every* example for both
training and validation (across different splits), rather than permanently sacrificing a chunk
of the data to a single static hold-out set.

**$k$-fold CV — the standard procedure:**
1. Shuffle the data and partition it into $k$ equal-sized folds $F_1,\ldots,F_k$
2. For each fold $i = 1,\ldots,k$: train the model on all folds *except* $F_i$, evaluate it on
   $F_i$, record the score $s_i$
3. Report $\bar{s} = \frac{1}{k}\sum_i s_i$ (and optionally its standard deviation, as a sense
   of how stable the estimate is across folds)

```
function k_fold_cv(data, k, fit, score):
    folds = split(shuffle(data), into=k, equal_parts=True)
    scores = []
    for i in 1..k:
        train_set = concat(folds except folds[i])
        model = fit(train_set)
        scores.append(score(model, folds[i]))
    return mean(scores), stdev(scores)
```

**Worked example.** Six labeled points $\{A,B,C,D,E,F\}$, $k=3$ (two points per fold:
$F_1=\{A,B\}$, $F_2=\{C,D\}$, $F_3=\{E,F\}$). Three iterations, each leaving one fold out as
validation:

| Iteration | Train on | Validate on | Validation accuracy |
|---|---|---|---|
| 1 | $\{C,D,E,F\}$ | $\{A,B\}$ | $0.50$ (1/2 correct) |
| 2 | $\{A,B,E,F\}$ | $\{C,D\}$ | $1.00$ (2/2 correct) |
| 3 | $\{A,B,C,D\}$ | $\{E,F\}$ | $0.50$ (1/2 correct) |

$\bar{s} = (0.50 + 1.00 + 0.50)/3 = 0.667$, $\text{std} = 0.236$. The mean is the CV
performance estimate; the spread tells you how much that estimate would itself wobble under a
different random fold assignment — a single train/test split would have reported just one of
these three numbers, with no way to know how representative it was.

> [!tip] CV measures *and* reduces estimate variance — at once
> A single 80/20 split gives you one noisy sample of "test performance." $k$-fold CV gives you
> $k$ samples of it, which (a) **averages away** much of that noise — exactly the
> $\text{Var}(\bar s) = \sigma^2/k$ effect that makes ensembling work (see
> [[ensemble-methods]]) — and (b) **exposes** how much noise was there in the first place, via
> the spread across folds. A model whose CV scores vary wildly fold-to-fold is telling you
> something a single split never could: its performance estimate is unreliable, regardless of
> what the mean says.

---

## Intermediate

**Choosing $k$ — a [[bias-variance-double-descent|bias-variance tradeoff]] in the *estimate itself*:**

| $k$ | Training set size per fold | Bias of estimate | Variance of estimate | Cost |
|---|---|---|---|---|
| Small (e.g. $k=2$) | Small (≈ $n/2$) | High — models trained on less data underperform the full-data model | Low | Cheap |
| Large (e.g. $k=n$, LOOCV) | Large (≈ $n-1$) | Low — nearly matches full-data training | High — folds overlap heavily, so scores are highly correlated | Expensive ($n$ fits) |
| $k=5$ or $k=10$ | Moderate | Moderate | Moderate | Moderate |

$k=5$ or $k=10$ is the standard default precisely because it sits in the sweet spot of this
tradeoff — empirically close to the best achievable bias while keeping both variance and
compute manageable.

**Leave-one-out CV (LOOCV)** is $k$-fold with $k=n$: train on all but one point, validate on
that one, repeat for every point. Least biased (each training set is almost the full dataset)
but most variable (the $n$ training sets overlap in $n-2$ of their $n-1$ points, so the $n$
scores are nearly identical to each other and barely independent — averaging them reduces
variance far less than averaging $k=5$ genuinely-different folds would).

**Stratified $k$-fold** preserves each fold's class proportions to match the overall dataset —
essential for imbalanced classification, where plain random folds can by chance produce a fold
with zero examples of a rare class, making that fold's score meaningless.

**Nested cross-validation** — for simultaneous hyperparameter tuning *and* unbiased
performance estimation: an **outer** CV loop estimates generalization performance; an
**inner** CV loop, run independently within each outer training set, selects hyperparameters.
Skipping the nesting (tuning on the same folds you report performance on) lets the
hyperparameter search "peek" at the validation folds — the reported score is then optimistic,
not a fair estimate of performance on genuinely new data (see the "winner's curse" note below).

---

## Advanced

**The "winner's curse" — CV for model selection vs. CV for performance estimation.** These are
two different jobs that are easy to conflate. If you evaluate 50 candidate models (or
hyperparameter settings) via $k$-fold CV and report the *best* one's CV score as *the* expected
test performance, you've introduced a selection bias: the winner was chosen partly *because*
it got lucky on these particular folds, so its CV score systematically overstates its true
generalization performance. The fix is exactly nested CV: use the inner loop to select, the
outer loop (which never saw the selection process) to report.

**Time-series CV — when shuffling is invalid.** Standard $k$-fold assumes exchangeable
(i.i.d.) data; shuffling a time series before splitting lets the model "see the future" during
training (e.g. predicting day 5 using a model fitted partly on day 10's data) — an optimistic
leak that wildly inflates the score. **Forward-chaining (rolling-origin) CV** instead grows the
training window forward in time and always validates on the period immediately *after* it:
train on $[1..t]$, validate on $[t{+}1..t{+}h]$, then advance $t$ and repeat — never validating
on data that precedes its training window.

**Group-aware CV.** When multiple examples share a dependency — e.g. several scans from the
same patient, or several crops from the same image — standard random folds can place correlated
examples on both sides of the train/validation split, letting the model "memorize" the group
identity rather than learn the generalizable signal (a subtle form of leakage). **Group
$k$-fold** / **leave-one-group-out** CV instead ensures every example from a given group lands
entirely in one fold.

**Computational alternatives.** For some model families, repeatedly refitting is wasteful:
**generalized cross-validation (GCV)** gives a closed-form approximation to LOOCV error for
linear smoothers ([[regularization-weight-decay|ridge regression]], smoothing splines) using the model's hat-matrix trace,
avoiding $n$ refits entirely. More generally, [[bootstrap|bootstrap-based]] estimates
(out-of-bag error for bagged ensembles) serve a similar role — a free-with-training validation
signal that sidesteps the extra fitting cost CV requires. See [[bootstrap]] for when each tool
is the better fit.

---

## Links

- [[statistical-inference-mle]] — CV estimates the generalization error of an estimator; it approximates the expected test error $E_{x_{\text{test}}}[L(f(x_{\text{test}}), y_{\text{test}})]$ without a separate test set
- [[bias-variance-double-descent]] — CV measures the full MSE = bias² + variance + noise; the optimal model complexity minimizes CV error, not training error
- [[bootstrap]] — bootstrap .632 estimator and k-fold CV both estimate test error; bootstrap is smoother but more computationally expensive; k-fold is the standard choice for hyperparameter tuning
- [[ensemble-methods]] — nested CV is required when both selecting a model and evaluating it; inner loop tunes hyperparameters, outer loop estimates generalization
- [[decision-trees]] — CART and random forests use CV to select max depth or number of estimators; CV is more reliable than AIC/BIC for non-parametric models
- [[regularization-weight-decay]] — the regularization coefficient $\lambda$ is chosen by CV; the optimal $\lambda$ balances in-sample fit against out-of-sample generalization
