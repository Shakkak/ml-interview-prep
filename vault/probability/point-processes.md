---
title: Point Processes
tags: [point-processes, poisson-process, hawkes-process, intensity-function, event-sequences]
aliases: [Poisson process, Hawkes process, temporal point process, counting process, intensity function]
difficulty: 2
status: complete
related: [distributions-overview, markov-chains, monte-carlo-methods, bayesian-inference, rnn-lstm]
---

# Point Processes

---

## Fundamental

### What Are Point Processes?

A **point process** is a random collection of points in some space (often time or 2D space). In the temporal case, it models sequences of events: earthquake occurrences, neural spike trains, user clicks, financial transactions.

A **temporal point process** is characterized by its **intensity function** $\lambda(t)$ — the instantaneous rate of events at time $t$:

$$\lambda(t) = \lim_{\Delta t \to 0} \frac{P(\text{event in } [t, t+\Delta t])}{\Delta t}$$

The expected number of events in $[a, b]$ is $\mathbb{E}[N(b) - N(a)] = \int_a^b \lambda(t) dt$.

### Poisson Process

**Homogeneous Poisson process:** constant intensity $\lambda(t) = \lambda$.
- Events arrive independently
- Inter-arrival times $\sim \text{Exp}(\lambda)$ (memoryless)
- Number of events in interval $[0, T]$ follows $\text{Poisson}(\lambda T)$

**Inhomogeneous Poisson process:** time-varying $\lambda(t)$ — models non-stationary arrival rates (rush hour traffic, seasonal sales patterns). Events still independent given $\lambda(t)$.

**Superposition:** sum of independent Poisson processes is Poisson with intensity $\lambda_1(t) + \lambda_2(t)$.

---

## Intermediate

### Hawkes Process: Self-Exciting Point Process

**Hawkes process (Hawkes, 1971):** the intensity depends on past events — each event "excites" future events:

$$\lambda(t) = \mu + \sum_{t_i < t} \phi(t - t_i)$$

where $\mu > 0$ is the baseline rate and $\phi(\cdot) \geq 0$ is the **excitation kernel** — how much a past event at $t_i$ increases the current rate.

**Exponential kernel:** $\phi(t) = \alpha e^{-\beta t}$ — exponentially decaying excitation.

**Branching interpretation:** events have "children" (triggered events) with each child independently triggering further descendants. The **branching ratio** $n = \int_0^\infty \phi(t) dt = \alpha/\beta$ is the expected number of children per event.
- $n < 1$: subcritical, process is stable (mean number of events per unit time is $\mu / (1 - n)$)
- $n \geq 1$: supercritical, events explode

**Applications:** earthquake aftershock modeling, social media retweet cascades, financial order book dynamics, crime prediction.

### Conditional Intensity and Likelihood

For an observed event sequence $\{t_1, \ldots, t_n\}$ in $[0, T]$:

$$\log p(\{t_i\}) = \sum_{i=1}^n \log \lambda(t_i) - \int_0^T \lambda(t) dt$$

This is the **log-likelihood** of the point process. The first term rewards the model for assigning high intensity at observed event times; the second penalizes excess predicted events elsewhere.

Maximum likelihood estimation of Hawkes process parameters is straightforward when the integral $\int_0^T \lambda(t)dt$ has a closed form (as it does for the exponential kernel).

---

## Advanced

### Neural Point Processes

**Neural Hawkes Process (Mei & Eisner, 2017):** replace the parametric excitation kernel with a continuous-time LSTM. The hidden state evolves continuously between events, enabling the model to capture complex non-parametric history dependence.

**Transformer Hawkes Process (Zuo et al., 2020):** model the intensity using self-attention over past events — each event type and timing attends over the full history to predict the next event type and timing.

**Intensity-free approaches (IFTPP):** instead of modeling $\lambda(t)$ directly, model the inter-event time distribution and event marks jointly as a conditional density.

### Spatial Point Processes

Extension to 2D: a **spatial Poisson process** with intensity $\lambda(x, y)$ models the distribution of locations (e.g., trees in a forest, crime locations, stars). The **K-function** and **pair correlation function** measure spatial clustering or repulsion beyond the Poisson baseline.

*See also: [[distributions-overview]] · [[markov-chains]] · [[monte-carlo-methods]] · [[rnn-lstm]]*
