---
title: Constitutional AI (CAI)
tags: [constitutional-ai, rlhf, alignment, rlaif, self-critique]
aliases: [Constitutional AI, CAI, RLAIF, AI feedback, self-critique, Anthropic alignment]
difficulty: 2
status: complete
related: [rlhf, dpo-preference, instruction-tuning, prompt-engineering, autoregressive-models]
---

# Constitutional AI

---

## Fundamental

### Motivation

Standard RLHF (see [[rlhf]]) requires extensive human feedback: crowdworkers compare model outputs to train a reward model. This is:
- Expensive and slow
- Inconsistent across annotators
- Hard to scale to new behaviors or values
- Potentially harmful (workers read harmful content)

**Constitutional AI (CAI, Bai et al., 2022)** replaces human preference labeling with AI-generated feedback guided by a set of **principles** (the "constitution") — a list of values and rules the model should follow.

### The Constitution

A constitution is a list of short principles, e.g.:
- "Respond in ways that are helpful, harmless, and honest"
- "Do not assist with activities that could harm people"
- "Prefer responses that are less likely to be used to generate harmful content even if the user doesn't intend harm"

The constitution encodes human values in natural language, enabling consistent feedback without per-example human labeling.

---

## Intermediate

### The CAI Pipeline

**Phase 1: Supervised Learning from AI Feedback (SL-CAI)**

1. Sample potentially harmful responses from a helpful-only base model
2. Ask the model to critique its own response against constitution principles: "Identify any ways this response is harmful according to [principle]"
3. Ask the model to revise based on the critique
4. Repeat the critique-revision loop $k$ times (typically 2–3)
5. Fine-tune the model on the final revised responses (behavior cloning on improved outputs)

**Phase 2: RLAIF (RL from AI Feedback)**

1. Sample pairs of responses (helpful-only model vs SL-CAI model)
2. Ask a feedback model (prompted with the constitution) to choose which response is more aligned with the principles
3. Train a preference model (reward model) on these AI-labeled comparisons
4. Fine-tune with RL (PPO) maximizing the preference model reward

The key innovation: human feedback is replaced by AI feedback, scaled via the constitution.

### Comparison to Standard RLHF

| Aspect | RLHF | CAI |
|--------|------|-----|
| Feedback source | Human annotators | AI model + constitution |
| Scalability | Bottlenecked by humans | Scales with compute |
| Consistency | Variable across annotators | Consistent given fixed model |
| Transparency | Implicit in human judgments | Explicit principles in constitution |
| Harmful content exposure | Human workers exposed | AI handles it |

---

## Advanced

### Scaling RLAIF

**Anthropic finding:** as AI feedback models scale (larger, better models writing critiques and comparisons), the quality of the trained model improves proportionally. This creates a feedback loop: better models → better constitutions → better aligned models.

**Constitutional AI vs DPO:** DPO (see [[dpo-preference]]) can be combined with CAI by generating AI preference pairs and training directly with DPO (no separate RL step), simplifying the pipeline.

### Multi-Constitution and Debate

**Multi-constitution CAI:** different agents use different constitutions (e.g., one focused on helpfulness, one on harmlessness) and debate to reach consensus. Implements a form of constitutional democracy where competing values are weighed.

**AI Safety via Debate (Irving et al., 2018):** a related approach — two AI agents debate, a human judge determines the winner. If the debate is honest, the winner is the truthful agent. CAI's self-critique is a degenerate single-agent version of debate.

### Practical Applications

CAI principles have informed Anthropic's training of Claude models. Key constitutional principles include:
- Avoiding assistance with mass casualty weapons
- Being honest about uncertainty
- Respecting user autonomy while avoiding harm to third parties
- Transparency about being an AI

*See also: [[rlhf]] · [[dpo-preference]] · [[instruction-tuning]] · [[prompt-engineering]]*
