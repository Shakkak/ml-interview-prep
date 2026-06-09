---
title: Anomaly Detection
tags: [anomaly-detection, self-supervised, unsupervised, computer-vision, statistics]
aliases: [anomaly detection, out-of-distribution detection, novelty detection, outlier detection]
difficulty: 2
status: complete
related: [self-supervised-overview, contrastive-learning, distributions-gaussian, evaluation-metrics-guide]
---

# Anomaly Detection

---

## Fundamental

Given a dataset of **normal** examples, detect when a new example is abnormal. Key constraint: anomalies are rare, unknown, or too diverse to enumerate — you cannot train a supervised classifier.

**Three related problems:**

| Term | Training data | Goal |
|------|--------------|-------|
| Anomaly detection | Normal only | Flag deviations at test time |
| Novelty detection | Normal + some known anomalies | Flag known + unknown anomalies |
| OOD detection | In-distribution data | Flag inputs outside training distribution |

**Method 1 — Reconstruction-based:** train an autoencoder or U-Net on normal data only. Assumption: the model learns to reconstruct normal patterns well; anomalies reconstruct poorly.

Anomaly score: $s(x) = \|x - \hat{x}\|_2^2$ (reconstruction error). Flag $x$ as anomalous if $s(x) > \tau$.

Variants: [[variational-autoencoders|VAE]] uses ELBO as anomaly score ($-\log p(x)$ under the model). PatchWise computes reconstruction error per spatial patch, producing a localization map.

Weakness: very expressive decoders can sometimes reconstruct anomalies well — the network generalizes too broadly.

**Method 2 — One-class classification (SVDD):** find the smallest hypersphere enclosing all normal features in embedding space. Anomaly = outside the sphere.

Deep SVDD trains a neural network to map normal data into a compact sphere centered at a fixed point $c$:

$$\mathcal{L} = \frac{1}{n}\sum_i \|f(x_i) - c\|^2$$

Hypersphere collapse: if $f$ learns $f(x) = c$ for all inputs (trivial), the loss is zero. Fix: initialize $c$ from the mean of initial feature vectors; do not use bias in the last layer (would allow shifting to $c$ trivially).

**Threshold selection:** all methods output a continuous score $s(x)$. Low threshold: high recall (catch most anomalies), many false alarms. High threshold: high precision, miss anomalies.

Best practice: use the few available anomaly examples as a validation set to tune $\tau$ — do **not** use them for training.

**Metric:** [[evaluation-metrics-guide|AUROC]] (Area Under ROC Curve) is standard — threshold-free, measures discrimination between normal and anomalous scores. Standard benchmark: MVTec AD (Bergmann et al., 2019) for industrial defect inspection.

---

## Intermediate

**Method 3 — Feature-space distance (PatchCore, Roth et al., 2022):**

Extract CNN features from normal training images. Build a memory bank of feature vectors.

At test time: compute distance from each test feature to its nearest neighbor in the memory bank. Large distance → anomaly.

Steps:
1. Extract patch features from a pretrained CNN (no fine-tuning required)
2. Subsample the feature bank using coreset selection (greedy farthest-first algorithm) to reduce memory
3. Anomaly score = distance to nearest coreset neighbor

Why it works: pretrained CNN features already encode rich visual information. Normal patches cluster tightly in feature space; anomalous patches fall outside. This entirely avoids training on anomalous data.

**Method 4 — Density estimation:**

[[normalizing-flows|Normalizing flows]] learn an invertible mapping $f: x \mapsto z$ such that $z \sim \mathcal{N}(0,I)$:

$$\log p(x) = \log p_z(f(x)) + \log|\det J_f(x)|$$

Both terms are tractable. Anomaly score: $-\log p(x)$. Advantage over reconstruction: gives exact likelihoods rather than proxy distances.

[[distributions-gaussian|Gaussian]] Mixture Model: fit $K$ Gaussians to normal data features; score = $-\log \sum_k \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$.

**Method 5 — Self-supervised anomaly detection:**

Train on normal data with a [[self-supervised-overview|self-supervised]] pretext task. The model learns a representation of normal data; anomalies score poorly on the pretext task.

**CutPaste (Li et al., 2021):** augment normal images with a "cut and paste" operation that simulates surface defects. Train a classifier to distinguish original vs. cut-pasted. The classifier's score serves as the anomaly signal. This creates synthetic anomalies from normal data, enabling supervised training without real anomaly examples.

**Comparison:**

| Method | Needs retraining? | Localization? | Speed |
|--------|-----------------|--------------|-------|
| Reconstruction | Yes | Yes (pixel map) | Medium |
| Normalizing flow | Yes | Yes | Slow |
| PatchCore | No (pretrained backbone) | Yes (patch scores) | Fast inference |
| Deep SVDD | Yes | No | Fast |
| CutPaste | Yes | Yes | Medium |

**Evaluating without anomaly labels:** use a held-out set of normals to check the false positive rate at a given threshold; use synthetic anomalies (CutPaste, Perlin noise) as proxy; report AUROC on standard benchmarks.

---

## Advanced

**The reconstruction paradox:** high-capacity decoders (e.g., powerful convolutional networks) can reconstruct anomalies well, violating the assumption that anomalies reconstruct poorly. This happens because the decoder learns to generalize from normal patterns to nearby anomalous patterns. Mitigation strategies: use a bottleneck architecture that forces compression, apply perceptual loss (feature space rather than pixel space), or use gradient-based anomaly scoring (input gradients indicate what the reconstruction depends on).

**EfficientAD (Batzner et al., 2023):** combines teacher-student feature distillation with a normalizing flow. A student network is trained to match a pretrained teacher's features on normal data. Anomaly score = discrepancy between teacher and student features. The flow is trained to model the distribution of teacher features. The combination achieves near-perfect AUROC on MVTec AD with fast inference, establishing the state of the art.

**PANDA (Reiss et al., 2021):** standard pretrained feature approaches suffer from "compactness" failure — pretrained features are not compact enough for one-class classification. PANDA adapts pretrained features using iterative pseudo-labeling: the model identifies likely normal examples from a mix, updates its one-class representation, and repeats. This addresses the problem that pretrained features were not trained to discriminate normal from anomalous.

**OOD detection vs semantic anomaly detection:** a key distinction often ignored. OOD detection asks "is this input from the training distribution?" (distribution-level). Semantic anomaly detection asks "does this contain a defect?" (semantic-level). A misaligned bolt with correct image statistics is OOD-normal (same photographic distribution) but semantically anomalous. Methods that measure feature distance capture semantic anomalies; methods that measure input likelihood detect distributional shifts. Most industrial inspection tasks require semantic anomaly detection.

**Anomaly detection in foundation model era:** [[clip|CLIP]] and DINOv2 features dramatically outperform supervised ImageNet features for anomaly detection because they capture richer semantic information. WinCLIP (Jeong et al., 2023) uses CLIP's text encoder to define "normal" through text descriptions rather than images, enabling zero-shot anomaly detection without any reference normal images — a fundamentally new capability that doesn't fit the classical framework.

---

*See also: [[self-supervised-overview]] · [[contrastive-learning]] · [[distributions-gaussian]] · [[evaluation-metrics-guide]] · [[variational-autoencoders]] · [[normalizing-flows]] · [[clip]]*
