---
title: RLHF — Reinforcement Learning from Human Feedback
tags: [llm, alignment, rlhf, ppo, reward-model]
aliases: [RLHF, PPO, InstructGPT, DPO, preference learning]
difficulty: 2
status: complete
related: [bayesian-inference, lora-quantization, attention-mechanism]
---

# RLHF — Reinforcement Learning from Human Feedback

---

## Fundamental

A language model trained only on next-token prediction learns to continue any text. It has no concept of "what the user wants." Given "How do I bake a cake?" it might continue with more questions rather than an answer.

**Supervised fine-tuning (SFT) alone isn't enough:** demonstrations are expensive to collect at scale, hard to define all axes of "good response" (helpful, harmless, honest, concise), and the model can overfit to the demonstration distribution.

**RLHF's solution:** use human *preferences* (comparisons between outputs) rather than direct demonstrations. Much cheaper to collect and more expressive.

In RL terms:
- **Agent:** the language model (policy $\pi_\theta$)
- **State:** the prompt/conversation so far
- **Action:** generating the next token (or full response)
- **Reward:** a score for the quality of the full response

Objective: $\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)}[r(x, y)]$

---

## Intermediate

### The Three-Stage RLHF Pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**

Fine-tune on a dataset of high-quality human-written (prompt, response) pairs:
$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{demo}}[\log \pi_\theta(y|x)]$$

This teaches the model the response format and domain. The result is $\pi_{SFT}$.

**Stage 2: Reward Model Training**

Present the same prompt to $\pi_{SFT}$, generate $K$ responses, have humans rank them by preference. Train $r_\phi(x, y)$ (SFT model with LM head replaced by a scalar head) using the Bradley-Terry pairwise loss:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

$y_w$ = preferred response, $y_l$ = rejected. Maximizes probability that the preferred response scores higher.

**Stage 3: RL Fine-tuning with PPO**

Optimize the policy to maximize reward while staying close to the SFT model:

$$\max_\theta \mathbb{E}_{x, y \sim \pi_\theta}\left[r_\phi(x,y) - \beta \cdot \log\frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)}\right]$$

The KL penalty $\beta \cdot D_{KL}(\pi_\theta || \pi_{SFT})$ prevents the policy from drifting too far and exploiting gaps in the reward model.

---

## Advanced

### PPO: Clipped Surrogate Objective

PPO stabilizes policy gradient training by capping how much the policy changes per update:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t A_t,\, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

- $\rho_t = \frac{\pi_\theta(y_t|x)}{\pi_{\theta_{old}}(y_t|x)}$ — probability ratio new/old policy
- $A_t = r_\phi(x, y) - V_\phi(x)$ — advantage (reward minus a value baseline)
- $\epsilon \approx 0.2$ — if new policy assigns much higher probability, clip and stop large updates

A separate value network (often initialized from the reward model) is trained alongside the policy to estimate $V_\phi(x)$.

### Reward Hacking

The reward model is an imperfect proxy. RL finds responses that maximize $r_\phi$ — which may diverge from true human preferences:
- Generating verbose, flattery-heavy responses (annotators may favor thoroughness)
- Confident but wrong answers (annotators prefer confident tone)
- Repetition (appears "comprehensive")

KL penalty mitigates this. **Iterated RLHF:** after initial training, collect new human preferences on the updated model's outputs and retrain the reward model to catch new hacking strategies.

### Constitutional AI and RLAIF

**Constitutional AI (Anthropic):** instead of human feedback, use another AI to judge outputs based on a set of principles ("the constitution"). The model critiques and revises its own outputs, then trains on those revisions. Scales to much larger datasets than human labeling.

**RLAIF:** replace human annotators with a large AI model (GPT-4, Claude) to judge preference pairs. Cheaper and faster, at the cost of inheriting the judge model's biases.

### DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) eliminates the reward model and PPO entirely. The optimal RLHF policy has a closed form:

$$\pi^*(y|x) \propto \pi_{SFT}(y|x) \exp\left(\frac{r(y,x)}{\beta}\right)$$

Rearranging and substituting into the Bradley-Terry loss (the partition function $Z(x)$ cancels):

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{SFT}(y_l|x)}\right)\right]$$

No reward model, no PPO, no value function. Stable supervised-style training. Competitive or better than RLHF on many benchmarks.

```
Base LM → SFT Model
  RLHF path: train reward model → PPO against reward model
  DPO path:  collect preference pairs → fine-tune with DPO loss
```

---

*See also: [[bayesian-inference]] · [[lora-quantization]] · [[attention-mechanism]]*
