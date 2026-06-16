---
title: Taylor Series & Power Series Expansions
tags: [taylor-series, calculus, approximation, numerical-methods]
aliases: [Taylor expansion, Maclaurin series, power series]
difficulty: 2
status: complete
depends_on: [matrix-calculus]
related: [matrix-calculus, math-convexity-jensen, lagrangian-optimization, activation-gelu-swish, loss-cross-entropy, backpropagation]
---

# Taylor Series & Power Series Expansions

---

## Fundamental

A Taylor series expresses any sufficiently smooth function $f$ as an infinite polynomial around a point $a$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

where $a$ = the expansion point (the point around which we approximate), $f^{(n)}(a)$ = the $n$-th derivative of $f$ evaluated at $a$, $n!$ = factorial of $n$ (e.g., $3! = 6$), and $(x-a)^n$ = the distance from $a$ raised to the $n$-th power.

When $a = 0$, this is called the **Maclaurin series**. The key insight: every smooth function looks like a polynomial locally. How far "locally" extends is determined by the radius of convergence.

**The five series every ML practitioner needs:**

$$e^x = \sum_{n=0}^\infty \frac{x^n}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots \qquad (\text{all } x)$$

$$\ln(1+x) = \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots \qquad (|x| \leq 1, x \neq -1)$$

$$\frac{1}{1-x} = \sum_{n=0}^\infty x^n = 1 + x + x^2 + x^3 + \cdots \qquad (|x| < 1)$$

$$\sqrt{1+x} = 1 + \frac{x}{2} - \frac{x^2}{8} + \frac{x^3}{16} - \cdots \qquad (|x| \leq 1)$$

$$(1+x)^\alpha = \sum_{n=0}^\infty \binom{\alpha}{n} x^n = 1 + \alpha x + \frac{\alpha(\alpha-1)}{2}x^2 + \cdots \qquad (|x| < 1)$$

**Geometric series sum:** $\sum_{n=0}^\infty x^n = \frac{1}{1-x}$ is the special case $\alpha = -1$. Memorize this — it appears in discounted rewards in RL, attention masking, and convergence proofs.

**First-order approximation (linearization):** near $a$, $f(x) \approx f(a) + f'(a)(x-a)$. This is the foundation of gradient descent: the gradient tells you the slope of the linear approximation of the loss surface.

**Second-order approximation:** $f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$ — a parabolic fit. The curvature $f''(a)$ tells you whether the function is locally convex (U-shaped) or concave. Used directly in Newton's method.

---

## Intermediate

### Radius of Convergence

Not all series converge everywhere. The radius of convergence $R$ is determined by the ratio test:

$$R = \lim_{n \to \infty} \left|\frac{a_n}{a_{n+1}}\right|$$

where $a_n$ is the coefficient of $x^n$ in the power series (e.g., for $e^x = \sum x^n/n!$, we have $a_n = 1/n!$) and $R$ is the radius of convergence (the series is valid for all $|x - a| < R$).

The series converges absolutely for $|x - a| < R$ and diverges for $|x - a| > R$. Boundary behavior ($|x-a| = R$) requires separate analysis.

**Why this matters in ML:** $\ln(1+x)$ only converges for $|x| \leq 1$. Using the series to approximate $\log(1 + e^z)$ for large $z$ fails — this is why the numerically stable log-sum-exp trick exists (see below).

**Truncation error (Taylor's remainder theorem):** the error from stopping at degree $n$ is:

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$$

where $R_n(x)$ is the truncation error from using only the first $n+1$ terms of the Taylor series, $c$ is some unknown point between $a$ and $x$ (Lagrange form — the exact value of $c$ is unknown but we know it exists), and $f^{(n+1)}(c)$ is the $(n+1)$-th derivative evaluated there.

for some $c$ between $a$ and $x$ (Lagrange form). This bounds how many terms you need for a given accuracy — critical for designing polynomial approximations in hardware or fast inference.

### The Log-Sum-Exp Trick

The softmax denominator $Z = \sum_k e^{z_k}$ overflows for large logits (e.g., $e^{1000} = \infty$ in float32). The fix uses $\ln(1+x)$ and the $e^x$ Taylor insight:

$$\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}, \quad m = \max_k z_k$$

Subtracting $m$ means the largest exponent is $e^0 = 1$; all others are $< 1$ and safe. This is numerically equivalent but avoids overflow. **The proof that it's exact:** $\log\sum_k e^{z_k} = \log(e^m \cdot \sum_k e^{z_k - m}) = m + \log\sum_k e^{z_k - m}$.

### GELU Approximation via Taylor

The exact GELU (Gaussian Error Linear Unit) uses the error function:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

Computing $\text{erf}$ is expensive. The standard approximation uses a Taylor-inspired polynomial fit to $\tanh$:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)$$

The $x + 0.044715 x^3$ term is a degree-3 polynomial approximation to the argument needed to make $\tanh$ match $\text{erf}$. The coefficient $0.044715$ was fit numerically; it corresponds to the $x^3/3$ term in the series for $\text{erf}^{-1}(\tanh(\cdot))$.

### Multivariate Taylor Expansion

For a function $f : \mathbb{R}^n \to \mathbb{R}$ around point $\mathbf{a}$:

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^\top H(\mathbf{a})(\mathbf{x} - \mathbf{a}) + \cdots$$

where $H(\mathbf{a}) = \nabla^2 f(\mathbf{a})$ is the Hessian matrix. This is the foundation of second-order optimization.

**For the loss $L(\theta)$ near current parameters $\theta_0$:**

$$L(\theta) \approx L(\theta_0) + g^\top(\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^\top H (\theta - \theta_0)$$

where $g = \nabla L(\theta_0)$ and $H = \nabla^2 L(\theta_0)$. Newton's method finds the minimum of this quadratic: $\theta^* = \theta_0 - H^{-1}g$.

---

## Advanced

### Second-Order Optimization via Taylor

The quadratic approximation gives the Newton step $\Delta\theta = -H^{-1}g$, which jumps directly to the minimum of the local quadratic. Gradient descent takes only the first-order term and ignores curvature:

$$\theta_\text{GD} = \theta_0 - \eta g \qquad \text{vs} \qquad \theta_\text{Newton} = \theta_0 - H^{-1}g$$

**Why not always use Newton?** The Hessian $H$ is $p \times p$ for $p$ parameters — storing it for a 70B-parameter model requires $70B^2 \cdot 4$ bytes $\approx 10^{19}$ bytes. Computing it requires $O(p^2)$ backprop passes. Modern second-order methods (K-FAC, Shampoo, Hessian-free optimization) approximate $H^{-1}g$ using Kronecker factorizations or conjugate gradient — trading some accuracy for tractability.

**Convergence rate of Newton:** for strictly convex functions near the minimum, Newton's method converges quadratically — the number of correct decimal digits doubles each iteration. Gradient descent converges linearly — the error shrinks by a constant factor each step. Newton's method is $O(\log\log(1/\epsilon))$ iterations; gradient descent is $O(\log(1/\epsilon))$.

### Taylor Series and Kernel Approximations

Random Fourier features (Rahimi & Recht, 2007) approximate the RBF kernel using the fact that:

$$k(x, x') = e^{-\|x-x'\|^2/2} = e^{-\|x\|^2/2} e^{x \cdot x'} e^{-\|x'\|^2/2}$$

The middle term $e^{x \cdot x'}$ expands as $\sum_{n=0}^\infty \frac{(x \cdot x')^n}{n!}$ — a polynomial series in the dot product. By sampling random features $\omega \sim \mathcal{N}(0, I)$, one can construct explicit feature maps $\phi(x)$ such that $\phi(x)^\top \phi(x') \approx k(x, x')$. This converts $O(n^2)$ kernel computation (for $n$ training points) to $O(nD)$ where $D$ is the number of random features.

### Taylor and Attention Approximation

Standard attention is $O(n^2)$ in sequence length $n$. The softmax denominator $\sum_j e^{q \cdot k_j / \sqrt{d}}$ is the expensive term. Linear attention approximates the exponential kernel $e^{q \cdot k}$ with a feature map:

$$e^{q \cdot k} \approx \phi(q)^\top \phi(k)$$

where $\phi$ uses the first few terms of the Maclaurin series $e^x \approx 1 + x + x^2/2 + \cdots$. By associativity of matrix multiplication, the attention sum then becomes $\sum_j \phi(q)^\top \phi(k_j) v_j = \phi(q)^\top \sum_j \phi(k_j) v_j^\top$, computable in $O(n)$ once $\sum_j \phi(k_j) v_j^\top$ is precomputed. The Performer (Choromanski et al., 2021) uses random features instead of truncated polynomial approximation for better approximation quality.

### Moment Generating Functions as Taylor Series

The moment generating function (MGF) of a random variable $X$ is $M(t) = \mathbb{E}[e^{tX}]$. Expanding $e^{tX}$ via Taylor:

$$M(t) = \mathbb{E}\left[\sum_{n=0}^\infty \frac{(tX)^n}{n!}\right] = \sum_{n=0}^\infty \frac{t^n}{n!} \mathbb{E}[X^n]$$

The $n$-th coefficient is $\mathbb{E}[X^n]/n!$ — the $n$-th moment divided by $n!$. Therefore:

$$\mathbb{E}[X^n] = M^{(n)}(0) = \frac{d^n M}{dt^n}\bigg|_{t=0}$$

This is the fundamental connection: the Taylor series of the MGF encodes all moments of the distribution. The cumulant generating function $K(t) = \log M(t)$ encodes cumulants — its first derivative at 0 is the mean, second is variance, third is skewness (scaled), fourth is excess kurtosis (scaled). Cumulants are additive for independent RVs, making them more convenient than moments for analyzing sums (central limit theorem, information theory).

---

## Links

- [[matrix-calculus]] — the multivariable Taylor expansion uses the gradient (first-order) and Hessian (second-order); these are the objects of matrix calculus
- [[math-convexity-jensen]] — a function is convex iff its second-order Taylor remainder is non-negative everywhere; convexity analysis relies on the Taylor approximation
- [[lagrangian-optimization]] — Newton's method solves the KKT stationarity condition using a second-order Taylor expansion of the Lagrangian
- [[activation-gelu-swish]] — GELU approximates the Gaussian CDF using a polynomial; the approximation is accurate because of the Taylor series for $\tanh$
- [[loss-cross-entropy]] — the cross-entropy approximation $-\log p \approx 1 - p$ for $p \approx 1$ is a first-order Taylor expansion; it explains the linear gradient at confident predictions
- [[backpropagation]] — the linearization of the network around current weights is a first-order Taylor approximation; gradient descent follows this linear approximation
- [[fisher-information]] — the Fisher information is the second-order Taylor coefficient of KL divergence at $d\theta = 0$; natural gradient uses this curvature
