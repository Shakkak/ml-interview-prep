---
title: Cross-Entropy Loss
tags: [loss, classification, probability, information-theory]
aliases: [log loss, NLL loss, categorical cross-entropy]
difficulty: 1
status: complete
related: [loss-kl-divergence, entropy-mutual-info, activation-softmax, activation-sigmoid-tanh, backpropagation]
---

# Cross-Entropy Loss

---

## Fundamental

Cross-entropy measures the average number of bits needed to encode samples from distribution $p$ using the optimal code for distribution $q$:

$$H(p, q) = -\sum_c p_c \log q_c$$

In classification: $p$ = true distribution (one-hot labels), $q$ = model predictions ([[activation-softmax|softmax]] probabilities).

**Binary cross-entropy** for label $y \in \{0, 1\}$ and predicted probability $\hat{y} \in (0,1)$:
$$L_{BCE} = -\left[y \log \hat{y} + (1-y)\log(1-\hat{y})\right]$$

- If $y=1$: loss $= -\log\hat{y}$ — penalizes low predicted probability for the true class.
- If $y=0$: loss $= -\log(1-\hat{y})$ — penalizes high predicted probability for the false class.

For sigmoid output $\hat{y} = \sigma(z)$, the gradient simplifies to $\frac{\partial L}{\partial z} = \hat{y} - y$ — prediction minus label.

**Categorical cross-entropy** for $C$ classes with one-hot label $y$ and softmax output $\hat{y}$:
$$L_{CE} = -\sum_c y_c \log \hat{y}_c = -\log \hat{y}_{y_\text{true}}$$

Only the log-probability of the correct class contributes (all other $y_c = 0$). Gradient with softmax: $\frac{\partial L}{\partial z_c} = \hat{y}_c - y_c$.

**When to use:** multi-class classification with softmax output, binary classification with sigmoid, language modeling (next-token prediction).

**Numerical stability:** never compute $\log(\text{softmax}(z))$ directly — use `log_softmax(z)` or `cross_entropy(logits, targets)` which uses the log-sum-exp trick to avoid overflow for large logits.

---

## Intermediate

**Connection to maximum likelihood estimation:** minimizing cross-entropy is equivalent to maximizing log-likelihood under the model distribution:
$$\arg\min_q H(p, q) = \arg\max_q \mathbb{E}_p[\log q] = \arg\max_q \sum_i \log q(y_i \mid x_i)$$

Training with cross-entropy = MLE of model parameters. The choice of loss function implicitly specifies the assumed noise distribution: cross-entropy corresponds to a categorical (multinomial) likelihood.

**Connection to [[loss-kl-divergence|KL divergence]]:**
$$H(p, q) = H(p) + D_{KL}(p \,\|\, q)$$

Minimizing cross-entropy = minimizing KL divergence between label distribution and model (since $H(p)$ is constant w.r.t. model parameters).

**Gradient behavior and training dynamics:** cross-entropy with softmax gives the gradient $\hat{y}_c - y_c$ (prediction minus one-hot label). For well-classified examples ($\hat{y}_{y_\text{true}} \approx 1$), the gradient magnitude $\approx 0$. For misclassified examples, the gradient is strong. This natural weighting toward hard examples makes cross-entropy well-suited for classification — compare with MSE on logits, which has no such normalization.

**Cross-entropy vs MSE for classification:** MSE on softmax outputs produces a loss landscape where the gradient of a misclassified example is bounded, while cross-entropy's $-\log\hat{y}$ produces unbounded gradient for near-zero predicted probabilities. This makes cross-entropy more aggressive at correcting confident misclassifications, leading to faster convergence in practice.

**Label smoothing interaction:** cross-entropy with hard one-hot labels drives logits to $\pm\infty$. Label smoothing (see [[regularization-label-smoothing]]) replaces one-hot targets with $(1-\varepsilon, \varepsilon/(K-1), \ldots)$, bounding the optimal logit gap to $\log\frac{(K-1)(1-\varepsilon)}{\varepsilon}$.

---

## Advanced

**Symmetric cross-entropy for noisy labels** (Wang et al., 2019): standard cross-entropy is asymmetric — $H(p, q) = -\sum p \log q$ has unbounded gradient when $q \to 0$ but finite gradient when $p \to 0$. For noisy labels, this causes the model to aggressively fit noise. Symmetric cross-entropy adds the reverse term:
$$L_{SCE} = \alpha \cdot H(p, q) + \beta \cdot H(q, p)$$
The reverse KL term $H(q, p) = -\sum_c q_c \log p_c$ bounds the gradient of noisy labels, providing noise tolerance without sacrificing clean-label accuracy.

**Cross-entropy and the log-likelihood geometry:** for a $C$-class problem, the cross-entropy loss defines a statistical manifold over model outputs. The Fisher information matrix of the multinomial model is $F = \text{diag}(\hat{y}) - \hat{y}\hat{y}^T$ (exactly the softmax Jacobian). Natural gradient descent uses $F^{-1}g$ as the update direction, moving in the direction of steepest descent in KL-divergence space rather than Euclidean parameter space. This is the motivation behind K-FAC and Shampoo optimizers.

**Polya-Gamma augmentation and Bayesian cross-entropy:** Bayesian treatment of logistic regression requires integrating over the posterior $p(\theta | \text{data}) \propto p(\text{data}|\theta) p(\theta)$ where $p(\text{data}|\theta)$ is the cross-entropy likelihood. The Polya-Gamma data augmentation (Polson et al., 2013) introduces auxiliary variables that make the posterior Gaussian-conditionally, enabling exact Gibbs sampling — a rare case of tractable Bayesian inference for cross-entropy models.

**Temperature scaling and calibration:** after training, model predictions can be calibrated post-hoc by finding a temperature $T$ that minimizes cross-entropy on a held-out validation set: $\hat{p}_i = \text{softmax}(z_i / T)$. Guo et al. (2017) showed this simple one-parameter recalibration is as effective as more complex methods for most architectures, despite the model parameters being fixed. Modern large models trained with label smoothing or trained on diverse data tend to be better calibrated, partially explaining why GPT-family models do not require temperature scaling as strongly as older discriminative classifiers.

---

*See also: [[loss-kl-divergence]] · [[entropy-mutual-info]] · [[loss-focal]] · [[regularization-label-smoothing]] · [[activation-softmax]]*
