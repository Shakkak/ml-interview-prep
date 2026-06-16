---
title: Constitutional AI (CAI)
tags: [constitutional-ai, rlhf, alignment, rlaif, self-critique]
aliases: [Constitutional AI, CAI, RLAIF, AI feedback, self-critique, Anthropic alignment]
difficulty: 2
status: complete
depends_on: [rlhf, dpo-preference]
related: [rlhf, dpo-preference, instruction-tuning, prompt-engineering, autoregressive-models]
---

# Constitutional AI

---

## Fundamental

### The Problem

Standard [[rlhf|RLHF]] requires extensive human feedback: crowdworkers compare model outputs pair-by-pair to train a reward model. This is expensive, slow, inconsistent across annotators, hard to scale to new behaviors, and harmful to the workers who must read and evaluate disturbing content.

**Constitutional AI (CAI, Bai et al., 2022)** replaces human preference labeling with AI-generated feedback guided by a set of written **principles** — the "constitution" — a short list of values the model should follow.

### Intuition

Think of it as replacing a large team of human raters with a model that has been given a rulebook. Instead of asking humans "which response is better?", you ask the model "does this response violate principle X? If so, rewrite it." The feedback is now unlimited (any prompt can be critiqued) and consistent (same rules every time).

The constitution is the key idea: it externalizes human values into explicit, readable natural language. Anyone can inspect and revise the constitution — something impossible with implicit human preferences baked into annotations.

### The Constitution

A constitution is a list of short principles, for example:
- "Respond in ways that are helpful, harmless, and honest"
- "Do not assist with activities that could harm people"
- "Prefer responses that are less likely to generate harmful content, even if harm is not intended"

These principles encode human values in natural language, enabling consistent feedback at scale without per-example human labeling.

---

## Intermediate

### The CAI Pipeline

**Phase 1: Supervised Learning from AI Feedback (SL-CAI)**

1. Sample potentially harmful responses from a helpful-only base model
2. Ask the model to critique its own response against the constitution: "Identify ways this response violates [principle]"
3. Ask the model to revise based on the critique
4. Repeat the critique–revision loop $k$ times (typically 2–3 rounds)
5. Fine-tune the model on the final revised responses (behavior cloning on self-improved outputs)

**Phase 2: RL from AI Feedback (RLAIF)**

1. Sample pairs of responses (helpful-only model vs SL-CAI model)
2. Ask a feedback model — prompted with the constitution — to choose which response is more aligned with the principles
3. Train a preference model (reward model) on these AI-labeled comparisons
4. Fine-tune with RL using Proximal Policy Optimization (PPO — a policy gradient algorithm that clips gradient updates to prevent catastrophic policy changes; see [[rlhf]]) to maximize the learned reward

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

**Anthropic finding:** as the AI feedback model scales (larger, more capable models writing critiques and comparisons), the quality of the trained model improves proportionally. This creates a feedback loop: better models → better constitutions → better aligned models.

**Constitutional AI with DPO:** [[dpo-preference|DPO]] can replace the PPO RL step. AI-generated preference pairs (constitution-guided) are used directly to train with DPO's supervised objective — no separate reward model needed. This simplifies the pipeline substantially.

### Multi-Constitution and Debate

**Multi-constitution CAI:** different agents use different constitutions (one focused on helpfulness, one on harmlessness) and debate to reach consensus. Implements a form of value pluralism where competing principles are weighed against each other.

**AI Safety via Debate (Irving et al., 2018):** a related approach — two AI agents debate a claim; a human judge determines the winner. The key claim: if both agents are honest, the truthful argument wins. CAI's self-critique is a degenerate single-agent version of debate (the model debates with itself).

### Practical Applications

CAI principles have informed Anthropic's training of Claude models. Key constitutional principles include:
- Avoiding assistance with mass casualty weapons
- Being honest about uncertainty
- Respecting user autonomy while avoiding harm to third parties
- Transparency about being an AI

## Links

- [[rlhf]] — CAI replaces human preference labelers with AI feedback (RLAIF); the same PPO-based RL training loop runs but uses AI-generated reward signals instead of human annotations
- [[dpo-preference]] — DPO can replace PPO in CAI to directly optimize from AI-generated preference pairs without training a separate reward model
- [[instruction-tuning]] — CAI's SL-CAI phase is instruction tuning on model-generated critiques and revisions; the RLAIF phase uses preference optimization with AI feedback
- [[prompt-engineering]] — the constitutional principles are applied via carefully crafted prompts asking the model to critique and revise its own responses
