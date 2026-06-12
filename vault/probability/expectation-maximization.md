---
title: Expectation-Maximization Algorithm
tags: [expectation-maximization, em-algorithm, latent-variables, mixture-models, mle]
aliases: [EM algorithm, E-step, M-step, EM, expectation maximization]
difficulty: 2
status: complete
related: [statistical-inference-mle, gaussian-mixture-models, variational-inference, bayesian-inference, clustering, distributions-gaussian]
---

# Expectation-Maximization Algorithm

---

## Fundamental

### The Problem: Latent Variables in MLE

Maximum likelihood estimation (see [[statistical-inference-mle]]) maximizes $\log p(\mathcal{D}|\theta)$. For latent variable models, the log-likelihood involves a sum over all possible hidden states:

$$\log p(\mathbf{x}|\theta) = \log \sum_{\mathbf{z}} p(\mathbf{x}, \mathbf{z}|\theta)$$

The sum/integral over $\mathbf{z}$ makes this non-convex and generally intractable to optimize directly — taking the gradient involves a ratio of sums, not separable terms.

**Examples where this arises:**
- Gaussian Mixture Models: which component generated each observation? (see [[gaussian-mixture-models]])
- Hidden Markov Models: which sequence of hidden states explains the observed sequence?
- Factor analysis: what are the latent factors?
- PCA/missing data: what are the missing values?

### The EM Algorithm

EM introduces an auxiliary distribution $q(\mathbf{z})$ and uses the ELBO identity (see [[variational-inference]]):

$$\log p(\mathbf{x}|\theta) \geq \underbrace{\mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z}|\theta)] - \mathbb{E}_{q(\mathbf{z})}[\log q(\mathbf{z})]}_{\text{ELBO}(\theta, q)}$$

EM alternates:

**E-step:** fix $\theta^{(t)}$, maximize ELBO over $q$:
$$q^{(t+1)}(\mathbf{z}) = p(\mathbf{z}|\mathbf{x}, \theta^{(t)})$$

Setting $q = p(\mathbf{z}|\mathbf{x}, \theta^{(t)})$ makes the ELBO equal to $\log p(\mathbf{x}|\theta^{(t)})$ — the bound is tight at the current $\theta$.

**M-step:** fix $q^{(t+1)}$, maximize ELBO over $\theta$:
$$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{q^{(t+1)}(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z}|\theta)]$$

The M-step maximizes the **expected complete-data log-likelihood** — the likelihood if we knew both $\mathbf{x}$ and the soft assignments of $\mathbf{z}$.

**Convergence guarantee:** EM never decreases the log-likelihood: $\log p(\mathbf{x}|\theta^{(t+1)}) \geq \log p(\mathbf{x}|\theta^{(t)})$.

---

## Intermediate

### EM for Gaussian Mixture Models

For a GMM with $K$ components, the hidden variable $z_n \in \{1, \ldots, K\}$ is the component assignment for observation $n$.

**E-step:** compute soft assignments (responsibilities):
$$r_{nk} = p(z_n = k|\mathbf{x}_n, \theta^{(t)}) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n; \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n; \mu_j, \Sigma_j)}$$

**M-step:** update parameters using weighted sufficient statistics:
$$N_k = \sum_{n=1}^N r_{nk}, \qquad \pi_k = N_k / N$$
$$\mu_k = \frac{1}{N_k}\sum_n r_{nk} \mathbf{x}_n, \qquad \Sigma_k = \frac{1}{N_k}\sum_n r_{nk}(\mathbf{x}_n - \mu_k)(\mathbf{x}_n - \mu_k)^\top$$

**K-means as hard EM:** if we replace the soft responsibilities $r_{nk} \in [0,1]$ with hard 0/1 assignments (argmax), and use isotropic covariances, the EM updates become exactly the k-means update rule. K-means is EM with a degenerate (infinite precision) noise model.

### Convergence Properties

**Monotone ascent:** the log-likelihood strictly increases at each EM step (unless at a stationary point). This is because the E-step makes the ELBO tight (no gap), and the M-step increases the ELBO — so the log-likelihood can only go up.

**Local optima:** EM converges to a stationary point (local maximum or saddle point), not necessarily the global maximum. The objective is non-convex for mixture models. Initialization matters — common strategies:
- K-means initialization for GMMs
- Random restarts with multiple runs
- Spectral initialization (e.g., moment methods) for provably-good initialization

**Speed:** EM can be slow near convergence — the rate is linear with factor $\rho = 1 - \lambda_\min / \lambda_\max$ where $\lambda$ are eigenvalues of the missing information matrix. When the fraction of "missing information" is large, EM converges very slowly. For latent factor models with many hidden variables, each EM step may barely improve the objective.

### Applications Beyond Mixture Models

**Hidden Markov Models (HMMs):** E-step = forward-backward algorithm (efficient $O(TK^2)$ dynamic programming for $T$ time steps and $K$ states). M-step = closed-form parameter updates. Together: the Baum-Welch algorithm.

**Factor analysis:** E-step = posterior over factors (Gaussian); M-step = updates to loading matrix and noise variance.

**Principal Component Analysis (PCA):** probabilistic PCA (Tipping & Bishop, 1999) is a factor analysis model. EM converges to the principal subspace. E-step = project data onto current components; M-step = update loadings. This EM interpretation allows handling missing data naturally.

**Missing data imputation:** treat missing values as latent variables. E-step = compute expected values of missing data given observed data and current parameters. M-step = update parameters using completed data. This is multiple imputation's probabilistic foundation.

---

## Advanced

### Generalized EM and Partial M-steps

**Generalized EM (GEM):** the M-step need not find the exact maximum — just any $\theta$ that increases the ELBO. This allows:
- Gradient ascent M-steps (one gradient step rather than exact maximization)
- Online EM / stochastic EM: process one mini-batch per E-step and M-step
- ECM (Expectation-Conditional-Maximization): partition $\theta$ and update each subset while holding others fixed

**Stochastic EM** is important for large-scale applications: the E-step and M-step are done on a mini-batch. The sufficient statistics are maintained as exponential moving averages. This gives mini-batch EM that scales to millions of points — used in online topic modeling (LDA), streaming GMMs.

### EM as Coordinate Ascent on ELBO

EM is coordinate ascent on the ELBO jointly in $(q, \theta)$:
- E-step: ascent in $q$ with $\theta$ fixed → makes ELBO equal to $\log p(\mathbf{x}|\theta)$ (tight)
- M-step: ascent in $\theta$ with $q$ fixed → increases ELBO

When the E-step is intractable (cannot compute $p(\mathbf{z}|\mathbf{x}, \theta)$ analytically), we can replace it with a variational approximation $q_\phi(\mathbf{z}) \approx p(\mathbf{z}|\mathbf{x}, \theta)$ — this is **variational EM**, equivalent to training a VAE (see [[variational-autoencoders]]).

### Information-Geometric Interpretation

At each E-step, we project the model distribution onto the family $\{p(\mathbf{z}|\mathbf{x}, \theta)\}$ via an m-projection (minimizing reverse KL). At each M-step, we project back onto the model family via an e-projection (minimizing forward KL). EM traces a zigzag path through two manifolds in information geometry, guaranteed to converge to their intersection (the MLE).

---

*See also: [[statistical-inference-mle]] · [[gaussian-mixture-models]] · [[variational-inference]] · [[bayesian-inference]] · [[clustering]] · [[distributions-gaussian]]*
