---
title: RMSProp and Adagrad
tags: [optimizer, training, adaptive-learning-rate]
aliases: [RMSProp, Adagrad, adaptive gradient]
difficulty: 2
status: complete
related: [optimizer-adam, optimizer-sgd-momentum, optimizer-lr-schedules]
---

# RMSProp and Adagrad

---

## Fundamental

**Adagrad** (Duchi et al., 2011) — accumulates the sum of squared gradients per parameter:
$$G_t = G_{t-1} + g_t^2$$
$$\theta_t \leftarrow \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon}\, g_t$$

**Key idea:** parameters with large past gradients get smaller effective learning rates. Parameters with sparse/small gradients get larger effective rates.

**Best use case:** sparse features (NLP bag-of-words, recommendation systems). A rarely-updated embedding parameter gets a full $\eta$ update; a frequently-updated one gets a smaller step — addressing the frequency imbalance naturally.

**Fatal flaw:** $G_t$ is monotonically increasing. After many steps, $\eta/\sqrt{G_t} \to 0$ and learning stops. Unsuitable for deep networks trained for many epochs.

**RMSProp** (Hinton, 2012) — replaces cumulative sum with exponential moving average:
$$v_t \leftarrow \beta v_{t-1} + (1 - \beta) g_t^2$$
$$\theta_t \leftarrow \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon}\, g_t$$

Typical $\beta = 0.9$. Now $v_t$ reflects *recent* gradient magnitudes; the effective learning rate no longer decays to zero. RMSProp = Adam without the first moment ($m_t$) and without bias correction.

---

## Intermediate

**Worked numerical example** (one parameter $\theta$, $\eta=0.01$, gradients $g_{1,2,3}=3.0$, then $g_4=0.1$):

| Step | $g_t$ | Adagrad $G_t$ | Adagrad step | RMSProp $v_t$ | RMSProp step |
|------|--------|-------------|-------------|-------------|-------------|
| 1 | 3.0 | 9.0 | −0.010 | 0.9 | −0.032 |
| 2 | 3.0 | 18.0 | −0.0071 | 1.71 | −0.023 |
| 3 | 3.0 | 27.0 | −0.0058 | 2.439 | −0.019 |
| 4 | 0.1 | 27.01 | −0.000192 | 2.196 | −0.00067 |

At step 4: Adagrad barely changes step size despite $30\times$ smaller gradient ($G_4 \approx G_3$). RMSProp produces an appropriately small step, but $v_4$ still reflects recent history — it will decay toward 0.01 after many more small-gradient steps, letting the LR recover.

**Optimizer lineage comparison:**

| | Adagrad | RMSProp | Adam |
|---|---|---|---|
| Accumulates | Sum $\sum g^2$ | EMA of $g^2$ | EMA of $g$ and $g^2$ |
| LR decay | Permanent → 0 | Adaptive (no decay) | Adaptive |
| Bias correction | No | No | Yes |
| Momentum | No | No | Yes |
| Memory per param | 1 | 1 | 2 |
| Best for | Sparse NLP (short runs) | RNNs, online learning | Transformers, general DL |

**When to use RMSProp over Adam:** RMSProp has fewer hyperparameters ($\eta, \beta$). In online or non-stationary settings, Adam's bias correction can cause instability because the corrected moment estimates overshoot in rapidly changing gradient landscapes. RMSProp is less aggressive in these cases. In practice, AdamW has largely superseded both for deep learning.

**Adagrad for sparse NLP:** in word embedding training, most embeddings are updated rarely (words appearing infrequently). Adagrad's monotonically growing $G_t$ is acceptable for short training runs: rare-word embeddings get large early updates, frequent-word embeddings quickly get small updates. For longer training (e.g., millions of steps), RMSProp or Adam with small $\beta_1$ is preferred.

---

## Advanced

**Diagonal vs full-matrix adaptive methods:** Adagrad and RMSProp maintain a diagonal preconditioner (one scalar per parameter). The full-matrix version of Adagrad uses $G_t = \sum_{s \leq t} g_s g_s^\top \in \mathbb{R}^{p \times p}$ and updates $\theta \leftarrow \theta - \eta G_t^{-1/2} g_t$. This accounts for gradient correlations across parameters and provably achieves better regret bounds on convex problems, but is $O(p^2)$ in memory — infeasible for neural networks. Shampoo (Gupta et al., 2018) uses Kronecker-factored matrix preconditioners to approximate this full-matrix update for neural network parameters.

**RMSProp origin and informal publication:** RMSProp was never formally published — it was introduced in Hinton's Coursera lecture slides (2012) as an unpublished method to fix Adagrad's LR decay. It became one of the most widely used optimizers despite having no formal derivation or convergence proof in its original form. Convergence analysis was provided later (Défossez et al., 2020), showing RMSProp with constant $\beta$ does not converge for general non-convex objectives without additional learning rate decay.

**Adam as RMSProp with momentum:** Adam's first moment $m_t$ provides gradient direction (momentum effect: smooths noisy gradients and accelerates in consistent directions). Adam's second moment $v_t$ provides the RMSProp-style adaptive scaling. Bias correction addresses the cold-start problem that RMSProp has implicitly (RMSProp starts with $v_0=0$, so early steps are too large — effectively the same problem that bias correction fixes in Adam). Without bias correction at step 1: $v_1 = (1-\beta_2)g_1^2$ is $(1-\beta_2)$ of the true second moment — the step is $(1-\beta_2)^{-1/2} \approx 31.6\times$ too large for $\beta_2 = 0.999$.

---

*See also: [[optimizer-adam]] · [[optimizer-sgd-momentum]]*
