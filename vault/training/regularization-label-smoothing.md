---
title: Label Smoothing
tags: [regularization, training, classification, label-smoothing, soft-labels]
aliases: [label smoothing, soft labels]
difficulty: 2
status: complete
related: [loss-cross-entropy, regularization-dropout, activation-softmax, model-calibration, knowledge-distillation]
---

# Label Smoothing

---

## Fundamental

**The problem.** [[loss-cross-entropy|Cross-entropy]] with one-hot targets pushes the model to output $z_y \to +\infty$ for the correct class. This causes large weight norms, overconfident predictions, and poor calibration.

**Definition.** Replace the hard one-hot target with a smoothed distribution:

$$\tilde{y}_k = \begin{cases} 1 - \varepsilon & k = y_{\text{true}} \\ \varepsilon / (K - 1) & k \neq y_{\text{true}} \end{cases}$$

where $\varepsilon$ is the smoothing factor (typically 0.1) and $K$ is the number of classes.

**Equivalently:**
$$\mathcal{L}_{\text{LS}} = (1-\varepsilon)\,\mathcal{L}_{\text{CE}} + \varepsilon\cdot H(\text{Uniform}, p)$$

Label smoothing = standard cross-entropy + penalty for KL distance from uniform — the model is discouraged from being arbitrarily confident.

**Numerical example.** $K = 4$ classes, true class = 2, $\varepsilon = 0.1$.

- Hard label: $[0,\, 1,\, 0,\, 0]$
- Smoothed: $[0.033,\, 0.9,\, 0.033,\, 0.033]$ (true class $= 1-0.1 = 0.9$; others $= 0.1/3$)

Model output $p = [0.1, 0.7, 0.1, 0.1]$:
- Hard CE: $-\log(0.7) = 0.357$
- Smoothed: $-(0.033\log 0.1 + 0.9\log 0.7 + 2\times 0.033\log 0.1) = 0.549$

The smoothed loss is higher, applying gradient pressure even when the model already puts 70% on the correct class.

---

## Intermediate

### Effect on the Optimal Logit Gap

With hard labels, the loss-minimizing model drives $z_y - z_k \to \infty$ for all $k \neq y$ — unbounded logit growth.

With label smoothing, the optimal logit gap is **finite**:

$$z_y - z_k = \log\frac{(K-1)(1-\varepsilon)}{\varepsilon}$$

For ImageNet ($K = 1000$, $\varepsilon = 0.1$): gap $= \log\frac{999 \times 0.9}{0.1} = \log 8991 \approx 9.1$ nats. The optimizer has a well-defined fixed point; logits cannot diverge.

### Benefits

| Benefit | Mechanism |
|---|---|
| Better calibration | Finite logit gap → confidence scores match accuracy |
| Improved accuracy | ~0.1–0.4% ImageNet gain (Szegedy et al., Inception v3) |
| Noisy label robustness | $\varepsilon = 0.1$ absorbs ~10% label noise gracefully |
| Implicit regularization | Prevents weight norms from diverging |

### When to Apply

- **Always in classification** unless training a teacher for distillation (see Advanced).
- Combined with mixup: both encourage smooth prediction surfaces.
- Default $\varepsilon = 0.1$ for image classification; lower ($0.05$) for fine-tuning on clean data.

---

## Advanced

### Penalty Reformulation: KL from Uniform

The smoothed loss can be rewritten as:

$$\mathcal{L}_{\text{LS}} = \mathcal{L}_{\text{CE}} - \varepsilon\, H(p) + \text{const}$$

where $H(p) = -\sum_k p_k \log p_k$ is the entropy of the model's output. Label smoothing explicitly **maximizes entropy** of the output distribution as a secondary objective — a maximum-entropy regularizer. High entropy → flat predictions → lower confidence.

This connects to maximum entropy principles (see [[maximum-entropy-principle]]): among all distributions consistent with the data, prefer the one with maximum entropy. Label smoothing implements this at the output level.

### Hurts Distillation Teachers

Hinton et al. showed that teacher models trained with label smoothing are **less useful** for knowledge distillation. Why: the soft logits of a label-smoothed teacher carry less information about inter-class similarities (the "[[knowledge-distillation|dark knowledge]]"). The smoothing has already compressed the logit distribution toward uniform — there is less signal in the off-diagonal entries for the student to learn from.

**Rule:** train distillation teachers without label smoothing. Apply smoothing to students instead, or use the teacher's soft outputs (which naturally provide smoothing).

### Calibration Connection

Label smoothing directly addresses calibration (see [[model-calibration]]): by capping logit growth, the [[activation-softmax|softmax]] output is prevented from saturating to 1.0. The model's confidence is better aligned with its empirical accuracy. Temperature scaling post-hoc achieves a similar effect but at inference time; label smoothing bakes it into training.

Empirically: Guo et al. (2017) found that modern deep networks trained with hard labels are severely overconfident (reliability diagram curves far below the diagonal). Label smoothing with $\varepsilon = 0.1$ reduces this gap substantially, often making post-hoc temperature scaling unnecessary.

### Relationship to Soft Labels from Mixup

Mixup training $(\tilde{x}, \tilde{y}) = (\lambda x_i + (1-\lambda)x_j,\; \lambda y_i + (1-\lambda)y_j)$ produces soft convex combinations of one-hot labels. This is a *data-dependent* form of label smoothing: the target varies per example based on the mixing ratio, rather than a fixed $\varepsilon$. Both prevent hard confidence; mixup also improves calibration in the feature interpolation region.

---

*See also: [[loss-cross-entropy]] · [[activation-softmax]] · [[model-calibration]] · [[knowledge-distillation]] · [[regularization-dropout]] · [[entropy-mutual-info]] · [[maximum-entropy-principle]]*
