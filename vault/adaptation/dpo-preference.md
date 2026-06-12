---
title: Direct Preference Optimization (DPO)
tags: [dpo, rlhf, preference-learning, alignment, reward-model-free]
aliases: [DPO, direct preference optimization, reward-free RLHF, preference optimization]
difficulty: 2
status: complete
related: [rlhf, instruction-tuning, loss-cross-entropy, statistical-inference-mle, autoregressive-models]
---

# Direct Preference Optimization (DPO)

---

## Fundamental

### The RLHF Problem DPO Solves

Standard RLHF (see [[rlhf]]) is a three-stage pipeline:
1. Supervised fine-tuning (SFT)
2. Train a reward model $r_\phi$ on preference data
3. Run PPO to maximize $r_\phi$ while staying close to the SFT model (KL constraint)

Stage 3 (PPO) is unstable, compute-intensive, and sensitive to hyperparameters. The reward model can also be exploited ("reward hacking").

**DPO (Rafailov et al., 2023)** collapses stages 2 and 3 into a single supervised fine-tuning step — no separate reward model, no RL.

### The Key Insight

Given the standard RLHF objective (maximize expected reward minus KL from reference):

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[r(x,y)] - \beta \text{KL}(\pi \| \pi_\text{ref})$$

The optimal policy has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

Inverting this: $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$

**DPO substitutes this reward expression directly into the Bradley-Terry preference model** (probability that $y_w$ is preferred over $y_l$):

$$p(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

---

## Intermediate

### DPO Loss Function

The DPO training loss is a binary cross-entropy over preference pairs $(x, y_w, y_l)$:

$$\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\!\left(\beta \left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right)\right]$$

**Implementation:** compute log-probabilities of $y_w$ and $y_l$ under both $\pi_\theta$ (trainable) and $\pi_\text{ref}$ (frozen SFT model). The difference of log-ratio differences is a scalar; apply BCE loss. No sampling required.

**$\beta$ hyperparameter:** controls the KL constraint strength:
- Small $\beta$: model can deviate far from the reference — stronger preference signal, risk of distribution collapse
- Large $\beta$: model stays close to reference — conservative updates, may underfit preferences

Typical values: $\beta \in [0.1, 0.5]$.

### DPO vs PPO

| Property | PPO (RLHF) | DPO |
|----------|-----------|-----|
| Reward model | Separate model required | Implicit in policy |
| Training | RL loop, online rollouts | Supervised, offline |
| Stability | Often unstable | Generally stable |
| Compute | High (PPO sampling) | ~Same as SFT |
| Reward hacking | Possible | Less likely |
| Data requirement | Can use online feedback | Requires preference pairs |

---

## Advanced

### Variants and Extensions

**IPO (Identity Preference Optimization, Azar et al., 2024):** DPO assumes the Bradley-Terry model for preferences. IPO directly minimizes a regularized preference objective without this assumption, sometimes giving more stable training.

**KTO (Kahneman-Tversky Optimization, Ethayarajh et al., 2024):** works with binary feedback (thumbs up/down) rather than pairwise preferences. Useful when collecting paired comparisons is difficult.

**ORPO (Monolithic Odds Ratio Preference Optimization, Hong et al., 2024):** combines SFT loss and preference loss into a single stage — no separate SFT phase needed. The odds ratio replaces the log-ratio difference.

**SimPO (Simple Preference Optimization, Meng et al., 2024):** removes the reference model entirely by using the average log-probability as an implicit reference. Simpler, often competitive with DPO.

### Practical Considerations

**Data quality matters most:** DPO is a discriminative loss — it separates $y_w$ from $y_l$ but doesn't ensure absolute quality. If $y_w$ itself is poor, DPO just makes the model slightly prefer it over something worse.

**Length bias:** DPO tends to increase output length because longer responses have lower per-token average probability, making the log-ratio difference artificially favorable. Length-controlled variants normalize by sequence length.

**Reference model alignment:** the reference $\pi_\text{ref}$ should be the SFT model. If the SFT model is poor, DPO can only improve within the distribution of the SFT model.

*See also: [[rlhf]] · [[instruction-tuning]] · [[autoregressive-models]] · [[loss-cross-entropy]]*
