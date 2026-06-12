---
title: Meta-Learning
tags: [meta-learning, maml, few-shot-learning, prototypical-networks, learning-to-learn]
aliases: [MAML, meta-learning, learning to learn, few-shot learning, prototypical networks, model-agnostic meta-learning]
difficulty: 3
status: complete
related: [transfer-learning, continual-learning, bayesian-inference, contrastive-learning, loss-cross-entropy]
---

# Meta-Learning

---

## Fundamental

### Learning to Learn

**Meta-learning** trains a model to learn new tasks quickly from few examples, by learning across many tasks. The meta-learner extracts a learning algorithm or a good initialization that transfers to new tasks.

**Two-level optimization:**
- **Outer loop (meta-training):** optimize over a distribution of tasks, learning what makes learning fast
- **Inner loop (task-specific):** adapt to a specific task using few examples

**Episode format:** each training episode samples a task with a **support set** (few labeled examples for adaptation) and a **query set** (for evaluation). Standard benchmarks: 5-way 1-shot or 5-way 5-shot classification.

---

## Intermediate

### MAML: Model-Agnostic Meta-Learning

**MAML (Finn et al., 2017)** finds an initialization $\theta^*$ from which a few gradient steps produce a well-adapted model for any task in the distribution.

**Inner loop** (for task $\tau_i$):
$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\tau_i}^{\text{support}}(\theta)$$

**Outer loop** (meta-update):
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\tau_i} \mathcal{L}_{\tau_i}^{\text{query}}(\theta'_i)$$

The outer gradient $\nabla_\theta \mathcal{L}(\theta')$ requires differentiating through the inner gradient update — second-order derivatives. **MAML computes the Hessian of the task loss**, which is expensive.

**First-order MAML (FOMAML):** drop the second-order term, use the task-adapted gradient directly. Surprisingly competitive with full MAML at much lower cost.

**Reptile (Nichol et al., 2018):** even simpler — just move toward the task-adapted weights: $\theta \leftarrow \theta + \epsilon(\theta'_i - \theta)$. No inner gradient computation needed.

### Prototypical Networks

**Prototypical Networks (Snell et al., 2017)** compute class prototypes as the mean embedding of support examples, then classify by nearest prototype in embedding space.

For class $c$ with support examples $S_c$:
$$\mathbf{p}_c = \frac{1}{|S_c|} \sum_{(\mathbf{x}_i, y_i) \in S_c} f_\theta(\mathbf{x}_i)$$

$$p(y = c | \mathbf{x}) = \frac{\exp(-d(f_\theta(\mathbf{x}), \mathbf{p}_c))}{\sum_{c'} \exp(-d(f_\theta(\mathbf{x}), \mathbf{p}_{c'}))}$$

Training with episodic cross-entropy optimizes for this nearest-prototype classification. Simpler than MAML, often competitive. Natural connection to metric learning (see [[loss-triplet]]).

### Matching Networks

**Matching Networks (Vinyals et al., 2016):** non-parametric — classify a query by attention-weighted nearest neighbor over support examples:

$$\hat{y} = \sum_{i} a(\hat{x}, x_i) y_i, \quad a(\hat{x}, x_i) = \frac{\exp(\text{sim}(\hat{x}, x_i))}{\sum_j \exp(\text{sim}(\hat{x}, x_j))}$$

The "attention" over examples is essentially a soft nearest-neighbor classifier. The embedding $f$ is trained end-to-end to make this work.

---

## Advanced

### Meta-Learning vs In-Context Learning

Modern LLMs exhibit **in-context learning** (ICL) — they can solve new tasks from examples in the prompt without any weight updates. This is a form of meta-learning implemented implicitly during pretraining.

**Connection:** pretraining on diverse tasks implicitly trains the model to meta-learn — it has seen many (context, answer) patterns and learned to extract the in-context task structure. MAML provides an explicit mechanism; ICL provides an emergent mechanism.

**Key difference:** MAML updates weights at meta-test time; ICL never updates weights — adaptation is "in the activations" (via attention over context tokens).

### Bayesian Meta-Learning

Meta-learning can be cast as Bayesian inference over tasks:
- Prior $p(\theta)$: learned from task distribution
- Posterior $p(\theta | S_c)$: updated by support examples
- Prediction: $\int p(y | x, \theta) p(\theta | S_c) d\theta$

**MAML as MAP inference:** the inner loop is (approximate) MAP inference with the meta-trained prior as initialization; the gradient step is a Laplace approximation.

**CNAP (Conditional Neural Adaptive Processes):** explicitly parameterize the posterior as a function of support statistics, enabling uncertainty-aware few-shot predictions.

### Applications

- **Drug discovery:** few-shot property prediction for new molecules
- **Robotics:** adapt motor controllers to new physical configurations
- **NLP:** few-shot task adaptation for low-resource languages
- **Computer vision:** open-vocabulary detection and segmentation

*See also: [[transfer-learning]] · [[continual-learning]] · [[contrastive-learning]] · [[bayesian-inference]] · [[loss-triplet]]*
