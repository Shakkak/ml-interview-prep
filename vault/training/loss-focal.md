---
title: Focal Loss
tags: [loss, object-detection, class-imbalance, retinanet]
aliases: [focal loss, FL, RetinaNet loss]
difficulty: 2
status: complete
related: [loss-cross-entropy, loss-dice, evaluation-metrics-guide]
depends_on: [loss-cross-entropy, evaluation-metrics-guide]
---

# Focal Loss

---

## Fundamental

**The problem:** in single-stage object detectors (SSD, RetinaNet), the model evaluates ~100,000 candidate boxes per image. The vast majority are background (easy negatives). A typical ratio: 1 positive : 1000 negatives.

With standard [[loss-cross-entropy|cross-entropy]], easy negatives have small individual loss (e.g., 0.001 each), but $1000 \times 0.001 = 1.0$ total — the gradient is completely dominated by background. The model learns to predict background well and fails on actual objects.

**Focal Loss** (Lin et al., 2017, RetinaNet) down-weights easy examples:
$$FL(p_t) = -(1-p_t)^\gamma \log(p_t)$$

where $p_t = p$ if $y=1$, else $p_t = 1-p$ (probability of the correct class). Compare to standard cross-entropy: $CE(p_t) = -\log(p_t)$.

The **modulating factor** $(1-p_t)^\gamma$:
- Easy example ($p_t \to 1$): factor $\to 0$ → loss contribution $\to 0$
- Hard example ($p_t \to 0$): factor $\to 1$ → loss = standard CE

**Effect of $\gamma$:**

| $\gamma$ | Behavior |
|----------|---------|
| 0 | Standard cross-entropy |
| 1 | Linearly down-weights easy examples |
| 2 | Default; quadratically down-weights; large effect on easy negatives |
| 5 | Very aggressive; only hardest examples matter |

With $\gamma=2$ and a well-classified example ($p_t = 0.9$): $(1-0.9)^2 = 0.01$ — the loss is 100× smaller than cross-entropy.

**Numerical comparison:**

Easy negative ($p_t = 0.99$): CE = 0.010, FL($\gamma=2$) = $(0.01)^2 \times 0.010 = 0.0001$ — 100× smaller.
Hard example ($p_t = 0.5$): CE = 0.693, FL($\gamma=2$) = $(0.5)^2 \times 0.693 = 0.173$ — only 4× smaller.

---

## Intermediate

**Alpha balancing:** Focal loss is often combined with a class-frequency weighting factor $\alpha_t$:
$$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

where $\alpha_t$ = class-frequency weight ($\alpha_t = \alpha$ for foreground, $1-\alpha$ for background), $p_t$ = probability of the correct class, $\gamma$ = focusing parameter, $(1-p_t)^\gamma$ = modulating factor that down-weights easy examples.

Typical values: $\alpha_t = 0.25$ for foreground (rare), $\alpha_t = 0.75$ for background (common). This balances absolute class frequency; focal modulation handles the easy/hard imbalance. In the original RetinaNet paper, $\alpha = 0.25$ and $\gamma = 2$ together gave the best results.

**Focal loss vs hard negative mining:** two-stage detectors (Faster R-CNN) avoid the imbalance problem by sampling: during ROI proposal, keep a fixed ratio of positive to negative examples. This is explicit curation. Focal loss enables single-stage detectors to handle the full distribution implicitly, without needing to curate anchor boxes. The payoff is that single-stage detectors with focal loss achieve accuracy competitive with two-stage detectors at lower latency.

**Gradient analysis:** the gradient of focal loss with respect to the logit $z$ (for a sigmoid output) is:
$$\frac{\partial FL}{\partial z} = \alpha_t(1-p_t)^\gamma\left[\gamma p_t \log(p_t) + p_t - 1\right]$$

where $z$ = pre-sigmoid logit, $p_t = \sigma(z)$ = sigmoid probability of correct class, $\alpha_t$ = class weight, $(1-p_t)^\gamma$ = modulating factor, $\gamma p_t \log(p_t) + p_t - 1$ = correction term that adjusts the standard BCE gradient.

The factor $(1-p_t)^\gamma$ appears in the gradient, confirming that easy examples ($p_t \approx 1$) contribute near-zero gradient even under the full backward pass.

**RetinaNet architecture context:** RetinaNet uses a Feature Pyramid Network backbone with class and box subnet heads applied at each pyramid level. Each level predicts for anchors at multiple scales and aspect ratios. With focal loss, all $\sim 100k$ candidate boxes can be used during training (no sampling), which is architecturally cleaner and enables better localization from hard examples.

**When focal loss is not needed:**
- Two-stage detectors: ROI sampling already balances positives/negatives.
- Dense classification tasks where all classes are equally frequent.
- Tasks where hard negative mining explicitly rebalances the gradient.

---

## Advanced

**Equalization Loss** (Tan et al., 2020) for long-tail recognition: focal loss down-weights easy examples globally. For long-tail datasets where rare categories always appear as "hard" (few training examples), focal loss paradoxically hurts them by increasing the gradient ratio toward frequent/easy classes. Equalization Loss instead suppresses gradients from backgrounds and suppresses negatives from categories with fewer than $\lambda$ total instances, directly addressing the frequency imbalance rather than the difficulty imbalance.

**VariFocal Loss** (Zhang et al., 2021) for detection quality estimation: standard focal loss uses binary foreground/background labels. VariFocal Loss uses an asymmetric formulation where the modulating factor depends on the Intersection-over-Union ([[evaluation-metrics-guide|IoU]]) quality score:
$$VFL(p, q) = \begin{cases} -q(q\log(p) + (1-q)\log(1-p)) & q > 0 \text{ (foreground)} \\ -(1-p)^\gamma \log(1-p) & q = 0 \text{ (background)} \end{cases}$$

where $p$ = predicted classification score, $q$ = IoU quality score as soft target (0 for background, IoU score for foreground), $\gamma$ = focusing parameter for background, foreground case uses a weighted BCE between $p$ and $q$.

This trains the classification head to also predict localization quality, used in YOLOX and TOOD for joint classification-localization confidence.

**Distribution Focal Loss** (Li et al., 2020): instead of treating bounding box regression as a point estimate, DFL models the target as a distribution over discrete bins. The loss is a weighted cross-entropy:
$$DFL(S_i, S_{i+1}) = -((y_{i+1} - y)\log(S_i) + (y - y_i)\log(S_{i+1}))$$
where $y$ is the target and $S_i, S_{i+1}$ are probabilities of adjacent discrete bins. This enables better localization uncertainty estimation and improved AP on small objects, used in GFL (Generalized Focal Loss, Li et al., 2020) and subsequent detection frameworks.

**Focal loss as an instance of multiplicative margin:** the modulating factor $(1-p_t)^\gamma$ can be seen as an adaptive margin that scales the loss contribution based on prediction confidence. This connects focal loss to curriculum learning: easy examples are down-weighted as if they were already "learned," while hard examples receive the full learning signal. The implicit curriculum emergent from focal loss often outperforms explicit curriculum learning schedules on detection tasks.

---

## Links

- [[loss-cross-entropy]] — focal loss is $FL(p_t) = -(1-p_t)^\gamma \log p_t$; when $\gamma=0$, it reduces to cross-entropy; the modulating factor $(1-p_t)^\gamma$ down-weights easy examples
- [[evaluation-metrics-guide]] — focal loss was designed to address the $1000:1$ foreground-background imbalance in one-stage detectors; AP (average precision) is the standard metric
- [[loss-dice]] — focal loss and Dice loss both address class imbalance; focal loss operates on per-pixel classification, Dice loss on segmentation masks; they are often combined (combo loss)
