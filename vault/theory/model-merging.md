---
title: Model Merging
tags: [model-merging, task-vectors, linear-interpolation, slerp, ties-merging, model-fusion]
aliases: [task vectors, model soup, weight averaging, TIES merging, SLERP, linear mode connectivity]
difficulty: 2
status: complete
related: [transfer-learning, lora-quantization, generalization-bounds, domain-adaptation, ensemble-methods]
depends_on: [transfer-learning, linear-algebra-fundamentals, lora-quantization]
---

# Model Merging

---

## Fundamental

### Weight Space Arithmetic

**Model merging** combines multiple fine-tuned models by operating directly in **weight space** — averaging or interpolating parameters rather than ensembling predictions.

**Model soup (Wortsman et al., 2022):** average the weights of several independently fine-tuned checkpoints of the same base model:

$$\theta_\text{merged} = \frac{1}{K} \sum_{k=1}^K \theta_k$$

where $K$ = number of independently fine-tuned checkpoints, $\theta_k$ = parameters of the $k$-th checkpoint (all fine-tuned from the same base model).

Surprisingly, the average model often outperforms any individual model. This works because the fine-tuned models live in the same loss basin as the base model — linear interpolation stays within the basin.

---

## Intermediate

### Task Vectors

**Task vectors (Ilharco et al., 2023):** define the **task vector** as the difference between a fine-tuned model and the base model:

$$\tau = \theta_\text{fine-tuned} - \theta_\text{base}$$

Task vectors support arithmetic:
- **Task addition:** $\theta_\text{merged} = \theta_\text{base} + \lambda_1 \tau_1 + \lambda_2 \tau_2$ — combine capabilities from two tasks
- **Task negation:** $\theta_\text{merged} = \theta_\text{base} - \lambda \tau$ — remove a capability (e.g., remove sentiment bias)
- **Analogy:** $\tau_A - \tau_B + \tau_C$ can sometimes transfer A→B adaptation to a new task C

**Linear mode connectivity:** this arithmetic works because the fine-tuned models are in a linearly connected valley of the loss landscape (see [[loss-landscape]]).

### SLERP (Spherical Linear Interpolation)

For interpolating between two models $A$ and $B$:

**Linear interpolation (LERP):** $\theta_t = (1-t)\theta_A + t\theta_B$

**SLERP:** treats model parameters as a point on a hypersphere, interpolates along the great circle:
$$\theta_t = \frac{\sin((1-t)\Omega)}{\sin\Omega}\theta_A + \frac{\sin(t\Omega)}{\sin\Omega}\theta_B, \quad \Omega = \arccos\left(\frac{\theta_A \cdot \theta_B}{\|\theta_A\|\|\theta_B\|}\right)$$

where $t \in [0,1]$ = interpolation parameter ($t=0$ gives $\theta_A$, $t=1$ gives $\theta_B$), $\Omega$ = angle between the two weight vectors on the unit hypersphere (the great-circle arc length).

SLERP preserves the norm of the interpolated model and tends to give smoother capability interpolation than LERP, particularly for fine-tuned models at different specializations.

### TIES Merging

Simple weight averaging fails when fine-tuned models have conflicting updates — one model increases a weight while another decreases it, and the average cancels both.

**TIES (Trim, Elect Sign, then Disjoint Merge, Yadav et al., 2023):**
1. **Trim:** set small task-vector values to zero (only keep significant updates)
2. **Elect sign:** for each parameter, take the majority sign across models
3. **Disjoint merge:** average only models that agree on the elected sign

This reduces interference between conflicting task vectors and improves multi-task merging.

---

## Advanced

### DARE (Drop and Rescale)

**DARE (Yu et al., 2023):** randomly drop task-vector parameters with probability $p$, then rescale by $1/(1-p)$ to preserve the expected magnitude. The sparsified task vectors have less interference when merged. Works well for merging LoRA adapters.

### Evolutionary Model Merging

**EvoMerge:** instead of hand-designing the merge recipe, use evolutionary optimization to search for the best merge coefficients $\lambda_k$ per layer or parameter group. Expensive but produces state-of-the-art merged models (Sakana AI, 2024).

### When Model Merging Works

Model merging requires models to be in the **same loss basin** — verified by low train/val loss at interpolated weights. Key conditions:
- Same base model architecture and weights
- Fine-tuned with relatively small learning rate (large LR leads to distant basins)
- Tasks are not adversarially conflicting
- LoRA merging works especially well (small task vectors, predictable interference)

Model merging fails for:
- Models fine-tuned from different bases
- Very different task domains with opposing gradient directions
- Large fine-tuning learning rates that push models far from base

## Links

- [[transfer-learning]] — model merging combines fine-tuned variants of the same pretrained base; the key insight is that fine-tuning creates task vectors (weight deltas) in a shared parameter space
- [[linear-algebra-fundamentals]] — task vector merging is linear arithmetic in weight space: $\theta_{\text{merged}} = \theta_{\text{base}} + \sum_k \alpha_k \Delta\theta_k$; the geometry of weight space determines merge quality
- [[lora-quantization]] — LoRA adapters can be merged via task vector addition; TIES-merging resolves sign conflicts that arise when merging multiple LoRA fine-tunes
- [[ensemble-methods]] — model merging is a cheaper alternative to ensembling: one forward pass instead of $k$; but a single merged model loses the variance reduction of true ensembles
- [[domain-adaptation]] — merging a domain-adapted model with the base model can recover lost capabilities from catastrophic forgetting, at the cost of partial task performance
