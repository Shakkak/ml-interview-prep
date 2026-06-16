---
title: RLHF — Reinforcement Learning from Human Feedback
tags: [llm, alignment, rlhf, ppo, reward-model]
aliases: [RLHF, PPO, InstructGPT, DPO, preference learning]
difficulty: 2
status: complete
depends_on: [instruction-tuning, loss-kl-divergence, bayesian-inference]
related: [bayesian-inference, lora-quantization, attention-mechanism]
---

# RLHF — Reinforcement Learning from Human Feedback

---

## Fundamental

A language model trained only on [[autoregressive-models|next-token prediction]] learns to continue any text. It has no concept of "what the user wants."

**Intuition:** imagine training a dog by showing it what to do (SFT) vs. training it by reward and correction (RL). Demonstrations are limited — you can only show so many examples. But evaluating responses is much easier than writing them: it's faster to say "response A was better than response B" than to write a perfect response from scratch. RLHF exploits this asymmetry: collect cheap preference comparisons, train a reward model to generalize preferences, then use RL to optimize the LM against that reward signal. Given "How do I bake a cake?" it might continue with more questions rather than an answer.

**[[instruction-tuning|Supervised fine-tuning (SFT)]] alone isn't enough:** demonstrations are expensive to collect at scale, hard to define all axes of "good response" (helpful, harmless, honest, concise), and the model can overfit to the demonstration distribution.

**RLHF's solution:** use human *preferences* (comparisons between outputs) rather than direct demonstrations. Much cheaper to collect and more expressive.

In RL terms:
- **Agent:** the language model (policy $\pi_\theta$)
- **State:** the prompt/conversation so far
- **Action:** generating the next token (or full response)
- **Reward:** a score for the quality of the full response

Objective: $\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)}[r(x, y)]$ — find model parameters $\theta$ that maximize the expected reward $r(x,y)$ over prompts $x$ from dataset $\mathcal{D}$ and responses $y$ sampled from the policy $\pi_\theta$.

---

## Intermediate

### The Three-Stage RLHF Pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**

Fine-tune on a dataset of high-quality human-written (prompt, response) pairs:
$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{demo}}[\log \pi_\theta(y|x)]$$

where $\mathcal{D}_{demo}$ is the demonstration dataset of (prompt, response) pairs, $\pi_\theta(y|x)$ is the model's probability of generating response $y$ given prompt $x$, and $-\log \pi_\theta(y|x)$ is the negative log-likelihood (cross-entropy loss). This teaches the model the response format and domain. The result is $\pi_{SFT}$.

**Stage 2: Reward Model Training**

Present the same prompt to $\pi_{SFT}$, generate $K$ responses, have humans rank them by preference. Train $r_\phi(x, y)$ (SFT model with LM head replaced by a scalar head) using the Bradley-Terry pairwise loss:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

where:
- $r_\phi(x, y)$ — the reward model (parameterized by $\phi$): a scalar head on top of the SFT model, outputting a real-valued score for response $y$ to prompt $x$
- $y_w$ — the preferred (winning) response; $y_l$ — the rejected (losing) response
- $\sigma(z) = 1/(1+e^{-z})$ — sigmoid function, converts the score difference into a probability
- The loss maximizes $\log\sigma(r_\phi(y_w) - r_\phi(y_l))$, i.e., the probability that the preferred response scores higher — the Bradley-Terry pairwise ranking model

$y_w$ = preferred response, $y_l$ = rejected. Maximizes probability that the preferred response scores higher.

**Stage 3: RL Fine-tuning with PPO**

Optimize the policy to maximize reward while staying close to the SFT model:

$$\max_\theta \mathbb{E}_{x, y \sim \pi_\theta}\left[r_\phi(x,y) - \beta \cdot \log\frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)}\right]$$

where:
- $\theta$ — the policy model parameters being optimized
- $r_\phi(x,y)$ — the frozen reward model's score for response $y$ to prompt $x$
- $\beta$ — KL penalty strength (typically $0.01$–$0.1$): how much the model is penalized for moving away from the SFT baseline
- $\log\frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)}$ — log-ratio of new policy vs SFT policy, equal to the [[loss-kl-divergence|KL divergence]] per sample

The [[loss-kl-divergence|KL penalty]] $\beta \cdot D_{KL}(\pi_\theta || \pi_{SFT})$ prevents the policy from drifting too far and exploiting gaps in the reward model.

---

## Advanced

### PPO: Clipped Surrogate Objective

PPO stabilizes policy gradient training by capping how much the policy changes per update:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t A_t,\, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

- $\rho_t = \frac{\pi_\theta(y_t|x)}{\pi_{\theta_{old}}(y_t|x)}$ — probability ratio new/old policy
- $A_t = r_\phi(x, y) - V_\phi(x)$ — advantage (reward minus a value baseline)
- $\epsilon \approx 0.2$ — if new policy assigns much higher probability, clip and stop large updates

A separate value network (often initialized from the reward model) is trained alongside the policy to estimate $V_\phi(x)$ — a baseline predicting the expected future reward for a given prompt $x$ before any response is generated. Subtracting this baseline from the reward gives the *advantage* $A_t$: "how much better was this specific response than average?" This variance reduction is critical — without it, gradient estimates are too noisy for stable training.

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

where $\pi^*(y|x)$ is the optimal policy (proportional to the SFT policy upweighted by the exponentiated reward, softened by $\beta$).

Rearranging and substituting into the Bradley-Terry loss (the partition function $Z(x)$ cancels):

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{SFT}(y_l|x)}\right)\right]$$

where $(y_w, y_l)$ are the preferred and rejected responses respectively, $\pi_\theta$ is the model being trained, $\pi_{SFT}$ is the frozen reference, and the log-ratio difference serves as an implicit reward signal.

No reward model, no PPO, no value function. Stable supervised-style training. Competitive or better than RLHF on many benchmarks.

```
Base LM → SFT Model
  RLHF path: train reward model → PPO against reward model
  DPO path:  collect preference pairs → fine-tune with DPO loss
```

---

## Links

- [[instruction-tuning]] — RLHF follows SFT; the SFT model provides the reference policy whose KL divergence constrains how far RLHF can deviate
- [[loss-kl-divergence]] — the RLHF objective includes a KL penalty $-\beta D_{KL}(\pi_\theta \| \pi_{ref})$ to prevent reward hacking by staying near the reference policy
- [[bayesian-inference]] — the reward model is a Bayesian regression of human preference comparisons; the Bradley-Terry model is the likelihood function
- [[lora-quantization]] — LoRA enables RLHF fine-tuning of 7B–70B models; only the LoRA adapters are updated during PPO training
- [[attention-mechanism]] — the policy model is an autoregressive transformer; the reward model adds a linear head on top of the last token embedding
- [[dpo-preference]] — DPO is the reward-model-free alternative to RLHF; it directly optimizes the policy from preference pairs without PPO
