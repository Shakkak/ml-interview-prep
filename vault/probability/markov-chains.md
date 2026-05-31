---
title: Markov Chains
tags: [probability, markov-chains, mcmc, sequential]
aliases: [Markov chain, Markov property, stationary distribution, MCMC, Markov decision process]
difficulty: 2
status: complete
related: [distributions-overview, bayesian-inference, entropy-mutual-info]
---

# Markov Chains

---

## Fundamental

### The Markov Property

A sequence $X_0, X_1, X_2,\ldots$ is a Markov chain if:

$$P(X_{t+1} = x | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} = x | X_t)$$

The future depends only on the present state, not on the full history. This is the **minimal memory assumption**.

### Transition Matrix

For a finite state space $\{1,\ldots,n\}$, the transition probability matrix $P$ has entries $P_{ij} = P(X_{t+1} = j | X_t = i)$. Each row sums to 1. The $k$-step distribution: $\pi^{(t+k)} = \pi^{(t)} P^k$.

**Weather example:** 2 states — Sunny (S) and Rainy (R).

$$P = \begin{bmatrix} 0.9 & 0.1 \\ 0.4 & 0.6 \end{bmatrix}$$

After 2 days starting Sunny: $P^2_{SS} = 0.85$ (90% after 1 day — slowly converging).

### Stationary Distribution

A distribution $\pi$ is stationary if $\pi P = \pi$ — the chain in $\pi$ stays in $\pi$.

**Solving for $\pi$ (weather example):**
$0.1\pi_S = 0.4\pi_R$, so $\pi_S = 4\pi_R$. With $\pi_S + \pi_R = 1$: $\pi_S = 0.8$, $\pi_R = 0.2$.

**Verify:** $[0.8, 0.2] \cdot P = [0.8(0.9)+0.2(0.4),\; 0.8(0.1)+0.2(0.6)] = [0.8, 0.2]$ ✓

Convergence from Rainy start:

| $k$ | $P(\text{Sunny})$ |
|-----|-------------------|
| 0 | 0.0 |
| 1 | 0.4 |
| 5 | ~0.74 |
| 20 | ~0.80 |
| $\infty$ | 0.80 |

### Ergodicity Conditions

An ergodic chain converges to a unique stationary distribution from any starting state:

- **Irreducible:** can reach any state from any state
- **Aperiodic:** no cyclic behavior (not returning every exactly $k$ steps)

If either condition fails, the chain may have multiple stationary distributions or oscillate.

---

## Intermediate

### Detailed Balance

A sufficient (not necessary) condition for $\pi$ to be the stationary distribution:

$$\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j$$

"Flow from $i$ to $j$ equals flow from $j$ to $i$" — the chain is **reversible**. Detailed balance is the key property exploited in MCMC to construct chains with a desired stationary distribution.

### MCMC: Sampling via Markov Chains

**Goal:** sample from a target distribution $\pi(x)$ (e.g., Bayesian posterior) that's hard to sample from directly.

**Idea:** construct a Markov chain with stationary distribution $\pi$. Run until it mixes (converges), then collect samples.

**Metropolis-Hastings:**
1. At state $x$, propose $x' \sim q(x'|x)$
2. Accept with probability $\alpha = \min\!\left(1, \frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}\right)$
3. Move to $x'$ if accepted; stay at $x$ otherwise

Key: only $\pi(x')/\pi(x)$ is needed — the normalizing constant cancels. Perfect for posteriors where $\pi(x) \propto p(\text{data}|x)p(x)$.

**Numerical example:** $\pi(x) \propto e^{-x^2/2}$, current $x=1.0$, propose $x' = 1.5$. $\alpha = \min(1, e^{-0.625}) = 0.535$. The move away from the mode is accepted only 54% of the time.

**Gibbs sampling:** for multivariate targets, cycle through each variable sampling from its conditional: $x_i^{(t+1)} \sim p(x_i | x_{-i}^{(t)})$. Requires tractable conditionals. Always accepts — no tuning needed. Slow when variables are highly correlated.

### Applications in ML

**n-gram language models:** order-$n$ Markov chain. The stationary distribution is the model's marginal word distribution.

**PageRank:** random surfer — with probability $d \approx 0.85$ follows a random link; with $1-d$ jumps to a random page. PageRank = stationary distribution of this Markov chain.

$$\pi = d P^\top \pi + (1-d)/n \cdot \mathbf{1}$$

**MDPs (Markov Decision Processes):** RL environments are MDPs: $(S, A, T, R, \gamma)$. The Markov property is the key assumption — the optimal policy depends only on the current state. Bellman equation: $V^\pi(s) = R(s) + \gamma \sum_{s'} T(s'|s, \pi(s)) V^\pi(s')$.

---

## Advanced

### Mixing Times and Spectral Gap

The **mixing time** $t_{mix}(\epsilon)$ is the number of steps until the chain is within $\epsilon$ (in total variation distance) of stationarity from the worst-case start:

$$t_{mix}(\epsilon) \leq \frac{\ln(1/\epsilon)}{1 - \lambda_2}$$

where $\lambda_2$ is the second-largest eigenvalue of the transition matrix. The **spectral gap** $1 - \lambda_2$ controls the mixing rate — a small gap means slow mixing. High-dimensional distributions with many modes (e.g., posteriors with isolated probability mass) have near-zero spectral gap and require exponentially many steps to mix. This is the fundamental obstacle for MCMC in complex Bayesian models.

### Hidden Markov Models

Latent states $Z_t$ form a Markov chain; observations $X_t$ depend only on $Z_t$:

$$P(Z_{t+1}|Z_t, Z_{t-1},\ldots) = P(Z_{t+1}|Z_t), \quad P(X_t|Z_t, \ldots) = P(X_t|Z_t)$$

**Inference algorithms:**
- **Forward-backward (Baum-Welch):** compute $P(Z_t | X_{1:T})$ in $O(TK^2)$
- **Viterbi:** find the most likely state sequence (MAP decoding) in $O(TK^2)$
- **Parameter learning:** EM (Baum-Welch) for learning transition, emission, and initial distributions

Used for speech recognition, POS tagging, gene sequence modeling.

### Hamiltonian Monte Carlo (HMC)

Standard Metropolis-Hastings with a random walk proposal mixes poorly in high dimensions — the acceptance rate drops and the chain explores slowly. HMC (Duane et al., 1987) introduces auxiliary momentum variables and simulates Hamiltonian dynamics:

$$H(x, p) = -\log \pi(x) + \frac{1}{2}p^\top M^{-1}p$$

Leapfrog integration approximates Hamiltonian dynamics, producing proposals that can travel far while maintaining high acceptance rates. HMC scales to hundreds of parameters; the No-U-Turn Sampler (NUTS, Hoffman & Gelman 2014) automatically tunes the trajectory length.

**Practical insight:** HMC requires gradients of the log-target density — available in modern probabilistic programming frameworks (Stan, PyMC, Pyro). For neural posteriors, this requires backpropagation through the network.

### Stochastic Gradient MCMC

**SGLD (Stochastic Gradient Langevin Dynamics, Welling & Teh 2011):** inject noise into SGD updates:

$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\nabla\log p(\theta_t|D) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \epsilon_t I)$$

As the learning rate $\epsilon_t \to 0$ following a specific schedule, the iterates converge to samples from the posterior. This bridges optimization and sampling — the same algorithm transitions from optimization (high learning rate) to posterior sampling (low learning rate). Used in Bayesian deep learning to sample neural network posteriors at scale.

---

*See also: [[bayesian-inference]] · [[distributions-overview]] · [[entropy-mutual-info]] · [[sampling-methods]]*
