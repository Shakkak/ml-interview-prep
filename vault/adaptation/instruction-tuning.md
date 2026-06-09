---
title: Instruction Tuning and Supervised Fine-Tuning (SFT)
tags: [instruction-tuning, sft, supervised-fine-tuning, flan, alpaca, chat-templates, catastrophic-forgetting]
aliases: [instruction tuning, SFT, supervised fine-tuning, FLAN, Alpaca, instruction following, chat fine-tuning]
difficulty: 2
status: complete
related: [rlhf, lora-quantization, transfer-learning, autoregressive-models, regularization-weight-decay]
---

# Instruction Tuning and Supervised Fine-Tuning (SFT)

---

## Fundamental

A pretrained language model (GPT, LLaMA) is trained on [[autoregressive-models|next-token prediction]] over raw text. It learns to continue text — not to follow instructions. Given "Write a summary of this article:", a raw pretrained model may respond by continuing with more instructions or producing irrelevant text.

**Instruction tuning** (also called supervised fine-tuning, SFT) bridges this gap by fine-tuning the model on curated (instruction, response) pairs. After SFT, the model learns the format: "given an instruction, produce a helpful response."

SFT is the first step in the alignment pipeline:
```
Pretraining → SFT → Reward model training → RL fine-tuning (PPO)
```
[[rlhf|RLHF]] requires a well-behaved SFT model as the starting point.

---

## Intermediate

### FLAN — Multi-Task Instruction Tuning

FLAN (Finetuned Language Networks, Wei et al., 2022) showed that instruction tuning dramatically improves zero-shot generalization.

**Key idea:** fine-tune on many NLP tasks, all expressed as natural language instructions. Tasks: summarization, translation, classification, QA, NLI, etc. Each task has multiple instruction templates ("Summarize:", "TL;DR:", "Give a brief description:").

**Result:** FLAN-137B outperformed GPT-3-175B zero-shot on most benchmarks despite being smaller. The model learned to generalize the instruction-following pattern to unseen tasks.

**FLAN-T5:** same approach applied to T5 encoder-decoder models. Publicly available; strong baseline for instruction-following tasks. FLAN-T5 XL (3B params) often outperforms GPT-3 (175B) on academic benchmarks.

### Alpaca / Vicuna Recipe

After LLaMA's release, the community built instruction-following models cheaply:

**Alpaca (Stanford, 2023):**
1. Start with LLaMA-7B (pretrained).
2. Generate 52,000 (instruction, input, output) triples using GPT-3.5 (self-instruct method).
3. Fine-tune LLaMA-7B on these 52K examples using standard [[loss-cross-entropy|cross-entropy]].
Cost: ~$600 to generate the dataset + ~$100 to fine-tune.

**Vicuna (Berkeley, 2023):** fine-tune LLaMA-13B on ~70K conversations collected from ShareGPT (ChatGPT conversations shared publicly). Multi-turn dialogue; qualitatively impressive results.

**Key lesson:** dataset quality matters more than quantity. 52K high-quality instruction examples are more effective than 500K noisy ones.

### Chat Templates

Instruction-tuned models expect input in a specific format. Different models use different templates:

**Alpaca format:**
```
### Instruction:
{instruction}

### Input:
{input (optional)}

### Response:
```

**ChatML format (OpenAI/Mistral):**
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{user message}
<|im_end|>
<|im_start|>assistant
```

**LLaMA-2 Chat:**
```
[INST] <<SYS>>
{system prompt}
<</SYS>>
{user message} [/INST]
```

Using the wrong template at inference causes dramatic degradation — the model was trained to expect the specific tokens that mark instruction boundaries. Always use the tokenizer's `apply_chat_template()` method.

---

## Advanced

### Data Quality vs Quantity

**What makes a good SFT dataset:**
- **Diverse instructions:** mix of tasks (summarize, translate, explain, code, math, creative writing).
- **Accurate responses:** incorrect answers are directly trained into the model.
- **Appropriate length:** responses that are too short are unhelpful; too long responses train verbose behavior.
- **Consistent format:** use the target chat template throughout.

**LIMA (Zhou et al., 2023):** showed that **1,000 carefully curated examples** can match models trained on 52K examples in head-to-head evaluations. They called this "superficial alignment" — the format and style of instruction following is learned from relatively few examples; the knowledge comes from pretraining. Going from 52K to 500K examples often provides diminishing returns if quality is similar.

### Catastrophic Forgetting

Fine-tuning on instruction data can overwrite the pretraining knowledge if done aggressively:

**Symptoms:**
- Model loses performance on standard benchmarks (MMLU, HellaSwag) after SFT.
- Model refuses to answer questions outside the fine-tuning distribution.
- Model produces repetitive or formulaic responses.

**Mitigations:**
- **Low learning rate:** 1e-5 to 2e-5 (vs 1e-3 for pretraining). Small updates preserve pretrained weights.
- **Few epochs:** 1–3 epochs. More overfits the fine-tuning distribution.
- **LoRA fine-tuning:** update only a small fraction of parameters — pretrained knowledge in frozen weights is preserved. See [[lora-quantization]].
- **Data mixing:** include a small fraction (~5%) of pretraining data in the SFT mix to maintain general capabilities.

### SFT vs RLHF vs DPO

| Method | Signal | Cost | Effect |
|--------|--------|------|--------|
| SFT | Gold response labels | Low | Teaches format; limited alignment |
| RLHF (PPO) | Preference scores → reward model → RL | High | Aligns with nuanced human preferences |
| DPO | Preference pairs (chosen, rejected) | Medium | Same goal as RLHF; simpler; no reward model |

SFT is almost always done first. RLHF/DPO then refines the SFT model to match human preferences — particularly helpfulness, harmlessness, and honesty.

---

*See also: [[rlhf]] · [[lora-quantization]] · [[transfer-learning]] · [[autoregressive-models]] · [[loss-cross-entropy]]*
