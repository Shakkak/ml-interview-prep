---
title: Continual Learning
tags: [continual-learning, catastrophic-forgetting, ewc, replay, lifelong-learning]
aliases: [lifelong learning, catastrophic forgetting, EWC, elastic weight consolidation, replay methods]
difficulty: 2
status: complete
related: [transfer-learning, domain-adaptation, regularization-weight-decay, meta-learning, neural-tangent-kernel]
depends_on: [transfer-learning, backpropagation, regularization-weight-decay]
---

# Continual Learning

---

## Fundamental

### Catastrophic Forgetting

When a neural network is fine-tuned on a new task $B$ after training on task $A$, performance on task $A$ typically **catastrophically degrades** — sometimes to random chance within a few hundred steps.

This happens because:
- Gradient updates for task $B$ move weights in directions that were important for task $A$
- Neural networks have overlapping representations — there are no isolated "task A circuits"
- SGD has no mechanism to preserve old functionality while learning new

**Human analogy:** humans learn task $B$ while retaining task $A$ through sleep consolidation, sparse coding, and hippocampal-neocortical interaction. Neural networks lack these mechanisms by default.

### The Continual Learning Setting

**Task sequence:** model encounters tasks $T_1, T_2, \ldots, T_n$ sequentially, with no (or limited) access to previous data.

**Evaluation:**
- **Backward transfer:** how much learning $T_n$ hurts earlier tasks ($\Delta_{i<n}$)
- **Forward transfer:** how much prior tasks help later tasks ($\Delta_{i>n}$)
- **Plasticity:** ability to learn new tasks quickly
- **Stability:** ability to retain old task performance

The core tension: **stability-plasticity dilemma** — a network that is plastic (fast learner) is fragile; a stable network is rigid.

---

## Intermediate

### Elastic Weight Consolidation (EWC)

**EWC (Kirkpatrick et al., 2017)** adds a regularization term that penalizes moving weights that were important for previous tasks:

$$\mathcal{L}_\text{new} = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $\theta^*$ are weights after training on task $A$ and $F_i$ is the Fisher information (importance) for parameter $i$.

**Intuition:** $F_i$ measures how much the loss on task $A$ would increase if we moved parameter $\theta_i$. High Fisher → parameter is critical → penalize changing it.

**Limitation:** as tasks accumulate, storing and adding one quadratic penalty per task becomes expensive. Approximate variants merge Fisher matrices.

### Replay Methods

**Experience replay:** maintain a small buffer $\mathcal{B}$ of examples from previous tasks. When training on $T_n$, mix in buffer samples:
$$\mathcal{L} = \mathcal{L}_n + \alpha \mathcal{L}_{\text{replay from } \mathcal{B}}$$

where $\mathcal{L}_n$ = current task loss, $\mathcal{B}$ = replay buffer of stored examples from previous tasks, $\alpha$ = replay mixing coefficient (weight of old task loss).

Buffer management: random sampling, reservoir sampling, or prioritized (store "hard" examples).

**Generative replay (DGR):** train a generative model (GAN, VAE) alongside the main model. When learning $T_n$, generate pseudo-examples of $T_1, \ldots, T_{n-1}$ and replay them. No real data buffer needed — but generative quality degrades over time.

**Dark Experience Replay (DER):** store logits (soft predictions) instead of labels. Replaying logits provides richer supervision — the model must match its previous full output distribution, not just the argmax.

### Architecture-Based Methods

**Progressive Neural Networks (Rusu et al., 2016):** freeze all columns for past tasks, add a new column for each new task with lateral connections to old columns. No forgetting by construction — but parameter count grows linearly with tasks.

**PackNet (Mallya & Lazebnik, 2018):** iteratively prune the network after each task, "pack" the pruned weights for that task (mark them as frozen), use the remaining weights for the next task. Achieves zero forgetting with fixed capacity.

---

## Advanced

### Parameter-Efficient Continual Learning

Modern approaches use parameter-efficient adapters (see [[lora-quantization]]) to isolate task-specific capacity:
- Freeze pretrained backbone
- Train a task-specific adapter or LoRA module
- No interference between tasks at all

This essentially reframes continual learning as a storage/retrieval problem: which adapter to use at inference. Used widely in LLM continual fine-tuning.

### Continual Pre-Training of LLMs

Large-scale continual learning for LLMs:
- **Temporal drift:** new web data (post-training cutoff) should be incorporated
- **Domain specialization:** adapt a general LLM to code, medicine, law
- Common approach: small learning rate, replay of original data (5–10% of new batch), cosine LR schedule

Catastrophic forgetting is less severe for large pretrained models: the large overparameterized network has many redundant parameters, so a new task can use different parameters than those critical for the original distribution.

## Links

- [[transfer-learning]] — continual learning is transfer learning across a sequence of tasks; the key challenge is backward transfer (forgetting old tasks when learning new ones)
- [[backpropagation]] — catastrophic forgetting is caused by SGD's gradient updates overwriting weights critical to previous tasks; EWC adds a quadratic penalty on weight changes to prevent this
- [[regularization-weight-decay]] — EWC is a form of L2 regularization towards previous weights: $\Omega = \sum_i F_i(\theta_i - \theta_i^*)^2$ where $F_i$ is the Fisher information importance weight
- [[meta-learning]] — meta-continual learning (OML, ANML) finds initializations that minimize forgetting across tasks; MAML's inner loop is adapted to support replay and fast adaptation
- [[domain-adaptation]] — domain-incremental learning is a CL setting where the input distribution shifts; the model must adapt without access to old data
- [[lora-quantization]] — LoRA for continual learning trains separate low-rank adapters per task and merges them; the low rank limits interference between tasks
