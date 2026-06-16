---
title: Lottery Ticket Hypothesis
tags: [lottery-ticket, pruning, sparse-networks, subnetworks, iterative-magnitude-pruning]
aliases: [lottery ticket hypothesis, winning ticket, sparse subnetwork, IMP, iterative magnitude pruning]
difficulty: 2
status: complete
related: [pruning-sparsity, bias-variance-double-descent, regularization-weight-decay, initialization, generalization-bounds]
depends_on: [pruning-sparsity, initialization, generalization-bounds]
---

# Lottery Ticket Hypothesis

---

## Fundamental

### The Hypothesis

**Lottery Ticket Hypothesis (LTH, Frankle & Carlin, 2019):**

> A randomly initialized dense neural network contains a subnetwork (**winning ticket**) such that, when trained in isolation from the same initialization, the subnetwork can match the full network's test accuracy in a similar number of steps.

The "lottery" metaphor: the random initialization is a lottery — most parameter subsets are losers (can't train well), but a few winning tickets exist with the right initialization.

**Operational definition:** a winning ticket is found by **Iterative Magnitude Pruning (IMP)**:
1. Train the dense network to convergence: $\theta \to \theta^*$
2. Prune the $p\%$ lowest-magnitude weights (globally or per-layer)
3. **Rewind** the remaining weights to their initial values $\theta_0$ (not $\theta^*$)
4. Retrain only the remaining weights from $\theta_0$
5. Repeat steps 1–4 (iteratively prune further)

The "winning ticket" = the sparse mask + the rewound initialization.

---

## Intermediate

### Rewinding is Critical

**Key finding:** resetting remaining weights to their initial values (rewind) is critical. Simply retraining from a new random initialization does not reproduce the result.

This implies the winning ticket depends on the specific initialization, not just the sparse topology. The initialization encodes a useful inductive bias for the subnetwork.

**Strong lottery ticket hypothesis:** large random networks contain winning tickets for any given function, even before training (Ramanujan et al., 2020 — the "edge-popup" algorithm finds untrained sparse networks matching trained dense networks).

### Findings at Scale

Original LTH experiments used small networks (VGG, ResNets on CIFAR). For larger networks:

**Early rewinding (Frankle et al., 2020):** for very deep networks, rewinding to $\theta_0$ (step 0) fails — the winning ticket is found by rewinding to early training steps $\theta_{k}$ (e.g., after 1–5% of training). This suggests large networks require a "warm-up" period before the useful subnetwork structure forms.

**Instability analysis:** a network is "stable" to SGD noise if small perturbations at initialization don't significantly change the solution. Stability at initialization correlates with whether IMP finds winning tickets.

---

## Advanced

### Supermasks and Untrained Networks

**Zhou et al. (2019):** found that at initialization, there exist **supermasks** (binary masks applied to random weights) such that the masked network without any training achieves >50% accuracy on MNIST. This is much stronger than LTH — no training needed.

**Edge-popup algorithm (Ramanujan et al., 2020):** use gradient descent to find binary masks (treated as continuous variables during optimization, then thresholded) on a random network. Finds untrained subnetworks matching trained full networks in quality.

These results suggest that large random networks contain the "structure" needed for useful functions — training is partly a search for the right subnetwork.

### Implications for Theory

LTH connects to several theoretical questions:
- **Why overparameterization helps:** larger networks contain more lottery tickets → higher probability of finding a good one with SGD
- **Implicit regularization of SGD:** SGD may preferentially find winning tickets (sparse, low-complexity solutions)
- **Generalization:** winning tickets may generalize better than arbitrary sparse masks because their initialization encodes useful priors

### Limitations

- IMP is expensive ($O(k)$ training runs for $k$ prune-retrain iterations)
- Winning tickets don't always transfer across tasks (though some transfer is observed)
- No practical speedup during training (the lottery is only found after training)
- At very large scale (GPT-3 class), LTH has not been systematically verified

## Links

- [[pruning-sparsity]] — the lottery ticket is found by iterative magnitude pruning (IMP): train → prune lowest-magnitude weights → reset to original init → repeat; the surviving subnetwork is the ticket
- [[initialization]] — the winning ticket must be reset to its original initialization (not zero); re-initializing to random weights destroys the winning lottery ticket
- [[generalization-bounds]] — the lottery ticket hypothesis suggests that generalization is concentrated in sparse subnetworks; this challenges parameter-count-based complexity measures
- [[bias-variance-double-descent]] — the lottery ticket exists from initialization but only becomes visible after training; the sparse subnetwork has lower effective complexity and often generalizes better
- [[regularization-weight-decay]] — weight decay is implicit magnitude regularization: it drives low-importance weights toward zero, making them easier to prune without performance loss
