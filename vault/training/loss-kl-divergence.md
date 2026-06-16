---
title: KL Divergence
tags: [loss, probability, information-theory, bayesian]
aliases: [KL divergence, Kullback-Leibler, relative entropy]
difficulty: 2
status: complete
depends_on: [entropy-mutual-info, distributions-overview, math-convexity-jensen]
related: [loss-cross-entropy, entropy-mutual-info, variational-autoencoders, bayesian-inference]
---

# KL Divergence

---

## Fundamental

When training a model to approximate a distribution — a VAE encoder approximating a Gaussian prior, a language model approximating the human preference distribution — you need to measure how different two probability distributions are. **KL divergence** is the standard tool: it measures the extra bits needed to encode samples from $p$ if you built your code for $q$ instead. It is always non-negative (you can never save bits by using the wrong code) and asymmetric (measuring "$q$ approximating $p$" differs from "$p$ approximating $q$"). This asymmetry is critical for understanding variational inference and RLHF.

The KL divergence from $q$ to $p$ measures how much information is lost when $q$ is used to approximate $p$:
$$D_{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\left[\log\frac{p}{q}\right]$$

where $p(x)$ = the true (reference) distribution, $q(x)$ = the approximating distribution, $\sum_x$ = sum over all possible outcomes, $\log \frac{p(x)}{q(x)}$ = the log-ratio (how much more likely $x$ is under $p$ than under $q$), and $\mathbb{E}_p[\cdot]$ = expectation taken under $p$ (samples from the true distribution). The notation $D_{KL}(p \| q)$ reads "KL divergence from $q$ to $p$" — the direction matters.

For continuous distributions: $D_{KL}(p \,\|\, q) = \int p(x) \log \frac{p(x)}{q(x)}\, dx$.

**Key properties:**
- **Non-negative:** $D_{KL}(p \,\|\, q) \geq 0$, with equality iff $p = q$ (Gibbs inequality / Jensen's inequality)
- **Asymmetric:** $D_{KL}(p \,\|\, q) \neq D_{KL}(q \,\|\, p)$ — direction matters critically
- **Not a distance:** no triangle inequality

> [!tip] Why KL divergence is always ≥ 0 ([[math-convexity-jensen]])
> Since $\log$ is concave, Jensen's inequality gives $\mathbb{E}[\log X] \leq \log(\mathbb{E}[X])$.
> Set $X = q(x)/p(x)$ and take the expectation under $p$:
> $\mathbb{E}_p\!\left[\log\frac{q}{p}\right] \leq \log\!\left(\mathbb{E}_p\!\left[\frac{q}{p}\right]\right) = \log\!\left(\sum_x p(x)\frac{q(x)}{p(x)}\right) = \log\!\left(\sum_x q(x)\right) = \log 1 = 0$
> Therefore $D_{KL}(p\,\|\,q) = \mathbb{E}_p[\log p/q] = -\mathbb{E}_p[\log q/p] \geq 0$.
> Equality holds iff $q/p$ is constant everywhere — i.e., $p = q$.

**Connection to cross-entropy:** $D_{KL}(p \,\|\, q) = H(p, q) - H(p)$. Minimizing KL divergence = minimizing [[loss-cross-entropy|cross-entropy]] (since $H(p)$ is constant w.r.t. model parameters).

**Worked numerical example:** $p = [0.4, 0.4, 0.2]$, $q = [0.5, 0.3, 0.2]$:
$$D_{KL}(p\,\|\,q) = 0.4\log\frac{0.4}{0.5} + 0.4\log\frac{0.4}{0.3} + 0.2\log(1) = -0.089 + 0.115 + 0 = 0.026 \text{ nats}$$

---

## Intermediate

**Forward vs reverse KL and their qualitative differences:**

*Forward KL* $D_{KL}(p \,\|\, q)$: minimizing this requires $q$ to cover all modes of $p$. If $p(x) > 0$ but $q(x) \approx 0$, the term $p(x)\log(p(x)/q(x)) \to +\infty$. So $q$ is forced to put mass wherever $p$ does — **mass-covering** or **mean-seeking**. Result: broad distributions, potentially poor fit to any single mode.

*Reverse KL* $D_{KL}(q \,\|\, p)$: minimizing this penalizes $q$ for putting mass where $p$ has none. If $q(x) > 0$ but $p(x) \approx 0$, the term blows up. So $q$ avoids regions where $p$ is small — **mode-seeking**. Result: narrow distributions capturing one mode well, ignoring others.

For a bimodal $p$: forward KL gives a broad $q$ that sits between the modes; reverse KL gives a narrow $q$ that sits on one mode.

**Gaussian KL (used in [[variational-autoencoders|VAEs]]):** for $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$:
$$D_{KL}(q \,\|\, p) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log\sigma^2 - 1\right)$$

where $q = \mathcal{N}(\mu, \sigma^2)$ is the approximate (encoder) distribution, $p = \mathcal{N}(0, 1)$ is the standard Gaussian prior, $\mu^2$ penalizes the mean being far from 0, $\sigma^2 - \log\sigma^2 - 1$ penalizes the variance deviating from 1 (this term equals 0 when $\sigma=1$), and the $1/2$ comes from the Gaussian KL derivation.

For $d$-dimensional diagonal Gaussian: $D_{KL} = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$.

Minimum at $\mu=0, \sigma=1$: substituting gives $\frac{1}{2}(0+1-0-1) = 0$. This closed form enables the VAE ELBO to be computed and differentiated analytically.

**KL in ML applications:**

| Context | Which KL | Purpose |
|---------|----------|---------|
| VAE regularization | $D_{KL}(q_\phi \,\|\, p)$ | Force [[bayesian-inference\|posterior]] toward prior |
| RLHF KL penalty | $D_{KL}(\pi_\theta \,\|\, \pi_{SFT})$ | Keep policy near base model |
| [[knowledge-distillation\|Knowledge distillation]] | $D_{KL}(p_\text{teacher} \,\|\, p_\text{student})$ | Student matches teacher |
| Variational inference | $D_{KL}(q \,\|\, p)$ | Approximate posterior (reverse KL) |

---

## Advanced

**The ELBO and KL as a tightness measure:** the evidence lower bound (ELBO) for a latent variable model $p(x, z) = p(x|z)p(z)$ is:
$$\log p(x) = \mathcal{L}_{ELBO}(q) + D_{KL}(q(z|x) \,\|\, p(z|x))$$
$$\mathcal{L}_{ELBO} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) \,\|\, p(z))$$

where $q(z|x)$ = variational posterior (encoder), $p(z|x)$ = true posterior (intractable), $D_{KL}(q\|p(z|x))$ = gap between ELBO and log evidence (zero when $q$ = true posterior), $\mathbb{E}_q[\log p(x|z)]$ = reconstruction term, $D_{KL}(q\|p(z))$ = regularization toward prior $p(z)$.

Maximizing the ELBO is equivalent to minimizing the KL between the variational posterior $q$ and the true posterior $p(z|x)$. The KL term is exactly zero iff $q = p(z|x)$ (posterior is recovered exactly). VAE training maximizes the ELBO via the reparameterization trick, which allows gradients to flow through the sampling operation $z = \mu + \sigma\epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$.

**KL in RLHF and the KL constraint choice:** in RLHF policy optimization, the KL penalty $D_{KL}(\pi_\theta \,\|\, \pi_\text{SFT})$ serves as a trust region constraint. The direction matters: reverse KL means the policy $\pi_\theta$ avoids actions the SFT model would never take (mode-seeking toward the SFT's preferences). This prevents reward hacking — the model can't shift probability to sequences that maximize reward but are out-of-distribution from the SFT model. Ziegler et al. (2019) showed this direction is crucial; using forward KL allows the policy to mode-collapse toward a narrow set of high-reward outputs.

**Jensen-Shannon divergence as a symmetric alternative:**
$$JS(p \,\|\, q) = \frac{1}{2}D_{KL}(p \,\|\, m) + \frac{1}{2}D_{KL}(q \,\|\, m), \quad m = \frac{p+q}{2}$$

where $m = (p+q)/2$ = mixture distribution (midpoint between $p$ and $q$), factor $1/2$ = symmetrizing constant, result bounded in $[0, \log 2]$ unlike KL which can be infinite.

JSD is symmetric, bounded in $[0, \log 2]$, and well-defined even when supports don't overlap. The original GAN training objective is equivalent to minimizing the JSD between the data distribution and the generator distribution (Goodfellow et al., 2014). When supports don't overlap (early in training), JSD has zero gradient — a fundamental problem that motivates Wasserstein GAN, which uses the Earth mover's distance instead.

**$f$-divergences and the general framework:** KL is a special case of an $f$-divergence:
$$D_f(p \,\|\, q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right)dx$$

where $f$ = convex function with $f(1)=0$ (determines which divergence), $p(x)/q(x)$ = density ratio, $q(x)f(\cdot)$ = $q$-weighted expectation of $f$ applied to the ratio.

for a convex function $f$ with $f(1)=0$. KL forward: $f(t) = t\log t$. KL reverse: $f(t) = -\log t$. JSD: $f(t) = t\log\frac{2t}{t+1} + \log\frac{2}{t+1}$. Chi-squared: $f(t) = (t-1)^2$. All $f$-divergences share the non-negativity property from Jensen's inequality. Different divergences emphasize different parts of the distribution (tails vs. modes), motivating the choice based on the application's requirements.

---

## Links

- [[entropy-mutual-info]] — cross-entropy $H(p,q) = H(p) + D_{KL}(p \| q)$; Shannon entropy is the prerequisite for understanding KL
- [[distributions-overview]] — KL is defined between two distributions; knowing which family to use is the starting point
- [[math-convexity-jensen]] — KL $\geq 0$ is proven via Jensen's inequality applied to the concave log function
- [[loss-cross-entropy]] — minimizing cross-entropy is equivalent to minimizing KL when true labels are fixed
- [[variational-autoencoders]] — the VAE ELBO contains a KL term forcing the approximate posterior toward the prior
- [[bayesian-inference]] — variational inference minimizes $D_{KL}(q \| p(\theta|D))$; the ELBO gap equals this KL
