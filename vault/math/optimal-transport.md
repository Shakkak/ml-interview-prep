---
title: Optimal Transport and Wasserstein Distance
tags: [optimal-transport, wasserstein, earth-movers-distance, generative-models, sinkhorn]
aliases: [Wasserstein distance, Earth Mover's Distance, EMD, optimal transport, WGAN, Sinkhorn]
difficulty: 4
status: complete
related: [generative-adversarial-networks, loss-kl-divergence, normalizing-flows, entropy-mutual-info]
depends_on: [distributions-overview, loss-kl-divergence, linear-algebra-fundamentals]
---

# Optimal Transport and Wasserstein Distance

---

## Fundamental

### The Earth Mover's Distance (Wasserstein-1)

Imagine two probability distributions as piles of dirt. The **Earth Mover's Distance (EMD)**, also called **Wasserstein-1 distance**, measures the minimum amount of "work" required to move the dirt in one pile to match the shape of the other:

$$W_1(p, q) = \inf_{\gamma \in \Gamma(p,q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]$$

where $\Gamma(p,q)$ is the set of all **joint distributions** (transport plans) $\gamma(x,y)$ with marginals $p$ and $q$.

**Intuition:** each unit of probability mass in $p$ must be transported to some location in $q$. The cost is the distance traveled. The EMD is the cheapest transport plan.

**Example:** $p = \delta_{0}$ (all mass at 0), $q = \delta_{1}$ (all mass at 1):
- $W_1(p, q) = 1$ (move the mass one unit)
- $D_{KL}(p \| q) = \infty$ ([[loss-kl-divergence|KL divergence]] is infinite for non-overlapping supports)
- $JS(p, q) = \log 2$ (a constant, providing zero gradient)

**Key advantage over KL/JS:** Wasserstein distance is finite and provides meaningful gradients even when the two distributions have **disjoint supports** — a critical property for generative model training.

---

## Intermediate

### The Kantorovich Formulation

The dual (Kantorovich) formulation of $W_1$ via the **Kantorovich-Rubinstein duality**:

$$W_1(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{x \sim q}[f(x)]$$

where the supremum is over all **1-Lipschitz functions** $f$ (i.e., $|f(x) - f(y)| \leq \|x - y\|$ for all $x, y$).

This dual form is what makes Wasserstein distance **computationally tractable** for high-dimensional distributions: instead of computing the primal transport plan (exponentially expensive), we optimize a Lipschitz function.

### Wasserstein-2 Distance

The general Wasserstein-$p$ distance:

$$W_p(p, q) = \left(\inf_{\gamma \in \Gamma(p,q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|^p]\right)^{1/p}$$

$W_2$ uses squared distances and is especially natural for [[distributions-gaussian|Gaussians]]:

$$W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) = \|\mu_1 - \mu_2\|^2 + \text{tr}\!\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right)$$

The Bures metric on Gaussian distributions. $W_2$ geometry has applications in distributional robustness and mean estimation.

### WGAN: Wasserstein Distance for Generative Adversarial Networks

Standard [[generative-adversarial-networks|Generative Adversarial Networks (GANs)]] minimize the Jensen-Shannon (JS) divergence between real distribution $p_r$ and generator distribution $p_g$. When $p_r$ and $p_g$ have disjoint supports (common early in training), $JS(p_r \| p_g) = \log 2$ — a constant that provides **zero gradient** to the generator.

**WGAN** (Arjovsky et al., 2017) replaces JS with $W_1$:

$$\min_G \max_{D: \|D\|_L \leq 1} \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

The discriminator (renamed "critic") is now required to be 1-Lipschitz rather than bounded in $[0,1]$. The loss value itself is a meaningful estimate of $W_1(p_r, p_g)$, and its gradient is **non-zero** even with disjoint supports.

**Lipschitz enforcement:**
- Weight clipping (original WGAN): clip weights to $[-c, c]$. Simple but hurts capacity.
- Gradient penalty (WGAN-GP, Gulrajani et al., 2017): penalize $(\|\nabla_x D(x)\| - 1)^2$ at interpolated points. Better in practice.
- Spectral normalization (Miyato et al., 2018): normalize each layer's weight matrix by its largest singular value. Computationally cheap and effective.

---

## Advanced

### Sinkhorn Algorithm for Regularized Optimal Transport

Computing exact $W_p$ requires solving a linear program — $O(n^3)$ for $n$ support points. **Sinkhorn's algorithm** (Cuturi, 2013) adds an entropy regularization term:

$$W_\epsilon(p, q) = \min_{\gamma \in \Gamma(p,q)} \langle C, \gamma \rangle - \epsilon H(\gamma)$$

where $C_{ij} = \|x_i - y_j\|^2$ is the cost matrix and $H(\gamma) = -\sum_{ij} \gamma_{ij} \log \gamma_{ij}$ is the transport plan's [[entropy-mutual-info|entropy]].

The solution satisfies $\gamma_{ij}^* = u_i K_{ij} v_j$ where $K_{ij} = e^{-C_{ij}/\epsilon}$, and **Sinkhorn iterations** alternate:

$$u \leftarrow a / (Kv), \quad v \leftarrow b / (K^\top u)$$

where $a \in \mathbb{R}^n_+$ = discrete probability vector for distribution $p$ (marginal constraint), $b \in \mathbb{R}^m_+$ = discrete probability vector for distribution $q$ (marginal constraint), $K = e^{-C/\epsilon}$ = elementwise exponential of cost matrix divided by temperature, $u \in \mathbb{R}^n_+$ and $v \in \mathbb{R}^m_+$ = scaling vectors that enforce the marginal constraints. Division is elementwise.

Until convergence. Each iteration is $O(n^2)$ — a dramatic speedup. The approximation quality improves as $\epsilon \to 0$ (recovers exact OT) and degrades as $\epsilon \to \infty$ (approaches independence coupling, $W = 0$).

**GeomLoss** and **POT** (Python Optimal Transport) implement differentiable Sinkhorn for use in neural network losses.

### Frechet Inception Distance (FID) and $W_2$ Connection

The Fréchet Inception Distance used to evaluate generative models is exactly $W_2^2$ between two **Gaussians** fitted to Inception feature activations:

$$FID = \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

FID is thus a Wasserstein-2 distance in feature space, which is why it is more sensitive to mode dropping and mode coverage than IS (Inception Score).

### Optimal Transport in Machine Learning

| Application | OT Formulation |
|-------------|----------------|
| WGAN training | $W_1$ via Lipschitz critic |
| Domain adaptation | $W_2$ as distributional distance between source/target |
| Point cloud matching | $W_p$ between 3D point clouds |
| Sentence similarity | Word Mover's Distance (WMD) — $W_1$ on word embeddings |
| FID metric | $W_2$ in Inception feature space |
| Data augmentation (MixUp) | Wasserstein barycenter interpolation |

---

## Links

- [[distributions-overview]] — OT defines distances between probability distributions; the Wasserstein distance is defined as the minimum-cost transport plan between two measures
- [[loss-kl-divergence]] — KL divergence is another distance between distributions but is asymmetric and infinite when supports don't overlap; Wasserstein distance is continuous and symmetric
- [[linear-algebra-fundamentals]] — the Kantorovich LP formulation of OT has $n^2$ variables and $2n$ constraints; the Sinkhorn algorithm solves it with matrix scaling operations
- [[generative-adversarial-networks]] — WGAN uses the Wasserstein distance as its training objective; the critic approximates the Kantorovich dual of the transport problem
- [[normalizing-flows]] — flow matching and rectified flows find straight-line transport plans; OT theory explains why straight paths minimize transport cost
- [[entropy-mutual-info]] — entropic regularization adds $\epsilon H(\gamma)$ to the OT objective; this converts the LP into the Sinkhorn algorithm and makes OT differentiable
