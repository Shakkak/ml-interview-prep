---
title: Hyperparameter Optimization
tags: [hyperparameter-optimization, bayesian-optimization, hyperband, nas, hpo]
aliases: [hyperparameter tuning, HPO, Bayesian optimization, Hyperband, ASHA, neural architecture search]
difficulty: 2
status: complete
related: [cross-validation, bias-variance-double-descent, gaussian-processes, optimizer-lr-schedules, generalization-bounds]
depends_on: [cross-validation, gaussian-processes, bayesian-inference]
---

# Hyperparameter Optimization

---

## Fundamental

### The Problem

**Hyperparameters** are settings not learned by gradient descent: learning rate, batch size, regularization strength, number of layers, dropout rate, etc. Poor choices cause underfitting or overfitting regardless of training quality.

**Manual tuning** works for 1–2 hyperparameters but fails for 5+. The search space grows exponentially with the number of hyperparameters.

### Grid Search and Random Search

**Grid search:** enumerate a Cartesian product of hyperparameter values. Covers all combinations but scales exponentially with the number of hyperparameters.

**Random search (Bergstra & Bengio, 2012):** sample hyperparameters randomly from specified distributions. Counterintuitively, this is **better than grid search** for most ML problems:
- If only a few hyperparameters matter (common in practice), random search spends its budget more efficiently
- For $n$ hyperparameters where only 2 matter, random search explores $n \times$ more values of those important hyperparameters per trial than grid search

---

## Intermediate

### Bayesian Optimization

**Bayesian optimization (BO)** builds a probabilistic surrogate model $p(y | \lambda)$ of the objective (validation loss) as a function of hyperparameters $\lambda$, then uses an **acquisition function** to select the next point.

**Algorithm:**
1. Evaluate $k$ initial random configurations
2. Fit a Gaussian process (or random forest) on $\{(\lambda_i, y_i)\}$
3. Maximize acquisition function to select $\lambda_\text{next}$
4. Evaluate $\lambda_\text{next}$, add to history
5. Repeat from step 2

**Acquisition functions:**
- **Expected Improvement (EI):** $\mathbb{E}[\max(y_\text{next} - y^*, 0)]$ — expected gain over current best $y^*$, where $y^*$ = best validation score seen so far, $y_\text{next}$ = predicted score at the next configuration
- **Upper Confidence Bound (UCB):** $\mu(\lambda) + \kappa \sigma(\lambda)$ — where $\mu(\lambda)$ = GP posterior mean, $\sigma(\lambda)$ = GP posterior standard deviation, $\kappa$ = exploration coefficient (high $\kappa$ explores uncertain regions)
- **Thompson Sampling:** sample one realization from the GP posterior, maximize it

**BO scales well** to 5–20 hyperparameters and typically finds good solutions in 50–200 evaluations vs thousands for random search.

**Tree Parzen Estimator (TPE):** models $p(\lambda | y < y^*)$ and $p(\lambda | y \geq y^*)$ separately using tree-structured kernel density estimation. Used in Hyperopt, Optuna — faster than GP-based BO, handles conditional hyperparameters.

### Hyperband and ASHA

For expensive training runs, early stopping saves compute. **Hyperband (Li et al., 2017):**

1. Sample many configurations (with small budget = few training steps)
2. Evaluate and keep the top fraction
3. Double the budget, repeat
4. Repeat until full budget

This is a principled version of "train briefly and keep the promising ones." Uses $O(\log N)$ resources to find the best of $N$ configurations.

**ASHA (Asynchronous Successive Halving Algorithm):** async version of Hyperband for parallel execution. Each worker trains a configuration until a rung; configurations are promoted based on performance at that rung. Near-linear speedup with workers.

---

## Advanced

### Neural Architecture Search (NAS)

**NAS** automates the design of the neural network architecture itself (number of layers, operation types, skip connections, etc.) — treating architecture as a hyperparameter.

**Reinforcement learning NAS (Zoph & Le, 2017):** an RNN controller generates architecture strings; child networks are trained and evaluated; REINFORCE updates the controller to generate better architectures. Expensive ($~450$ GPU-years for NASNet).

**DARTS (Liu et al., 2019):** differentiable NAS — maintain a mixture of candidate operations at each edge, jointly optimize operation weights and architecture parameters via gradient descent. Much cheaper than RL-based NAS.

**Hardware-aware NAS (ProxylessNAS, Once-for-All):** optimize for both accuracy and target hardware latency. Produces models Pareto-optimal on the accuracy-latency frontier.

### Multi-Fidelity and Transfer

**Multi-fidelity BO:** use cheaper approximations (small dataset, few epochs, low-resolution) to guide search, escalating fidelity for promising configurations. Reduces evaluation cost by 10–100×.

**Warm-starting from similar tasks:** initialize the BO surrogate with results from similar tasks (e.g., same model, different dataset). Reduces required evaluations significantly.

## Links

- [[cross-validation]] — HPO uses CV to evaluate each hyperparameter configuration; the CV score is the acquisition function's "truth value" that the surrogate model learns to predict
- [[gaussian-processes]] — Bayesian Optimization uses a GP as the surrogate model: GP posterior gives a mean prediction and uncertainty for each hyperparameter configuration
- [[bayesian-inference]] — the BO acquisition function (EI, UCB) balances exploration (uncertainty) and exploitation (expected improvement) using the GP posterior
- [[optimizer-lr-schedules]] — the learning rate schedule is itself a hyperparameter; HPO often jointly optimizes initial LR, decay schedule, and warmup steps
- [[bias-variance-double-descent]] — the regularization coefficient $\lambda$ and model capacity are the primary hyperparameters controlling the bias-variance tradeoff; HPO finds the optimal tradeoff
