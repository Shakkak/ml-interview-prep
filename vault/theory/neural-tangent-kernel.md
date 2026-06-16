---
title: Neural Tangent Kernel (NTK)
tags: [theory, neural-tangent-kernel, ntk, optimization, spectral-bias, infinite-width]
aliases: [NTK, neural tangent kernel, lazy training, infinite-width limit]
difficulty: 4
status: complete
related: [spectral-bias, optimizer-adam, bias-variance-double-descent, generalization-bounds]
depends_on: [kernel-methods, matrix-calculus, generalization-bounds]
---

# Neural Tangent Kernel (NTK)

---

## Fundamental

### The Linearization of Neural Networks

When a neural network $f(x; \theta)$ is overparameterized and initialized at $\theta_0$, gradient descent can behave like optimization on a **linear** model in disguise. The key insight: if parameters move very little during training (the **lazy training** regime), we can approximate

$$f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0)$$

This is a linear function of the parameter change $\theta - \theta_0$, with features given by the gradient vector $\nabla_\theta f$.

### The Neural Tangent Kernel

The **Neural Tangent Kernel (NTK)** is the kernel induced by these gradient features:

$$K_{NTK}(x, x') = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)$$

where $x, x'$ = two input examples, $\theta$ = network parameters (at initialization), $\nabla_\theta f(x; \theta) \in \mathbb{R}^{|\theta|}$ = gradient of the scalar network output w.r.t. all parameters (the "feature" vector induced by $x$).

This measures how correlated the gradient directions are for two inputs $x$ and $x'$. Jacot et al. (2018) proved that as network width $\to \infty$, during gradient descent training:

1. The NTK **stays constant** throughout training (it does not change as $\theta$ evolves).
2. Training dynamics become those of **kernel gradient descent** with kernel $K_{NTK}$.
3. The loss converges exponentially to zero at a rate governed by the NTK's smallest eigenvalue.

---

## Intermediate

### Infinite-Width Limit and Kernel Regression

In the infinite-width limit, training a neural network with squared loss and gradient descent is equivalent to **[[kernel-methods|kernel regression]]** with the NTK:

$$\hat{f}(x) = K_{NTK}(x, X_{train}) \bigl[K_{NTK}(X_{train}, X_{train})\bigr]^{-1} y_{train}$$

where $X_{train}$ and $y_{train}$ are the training inputs and labels.

This has a closed-form solution — no iterative optimization needed. The NTK for common architectures (fully-connected, convolutional) can be computed analytically.

**Practical implication:** at infinite width, neural networks are not learning features — they are performing kernel interpolation with a fixed kernel determined by the architecture and initialization. This is why the regime is called **lazy training**: the network doesn't learn representations, it just fits a kernel machine.

### NTK Eigenvalue Spectrum and Spectral Bias

The NTK $K_{NTK}$ is a positive semi-definite matrix over training inputs. Its eigendecomposition reveals the order in which functions are learned:

- **Large eigenvalue** → fast convergence along that eigendirection
- **Small eigenvalue** → slow convergence

For standard architectures with smooth activations, the NTK eigenvalues decay with frequency: low-frequency eigenfunctions have large eigenvalues, high-frequency ones have small eigenvalues.

**Consequence (spectral bias / frequency principle):** gradient descent learns low-frequency components of the target function first, regardless of the network size. This is the formal NTK explanation for the **[[spectral-bias|spectral bias]]** (Rahaman et al., 2019) observed empirically.

---

## Advanced

### Lazy Training vs Feature Learning

The NTK theory describes one extreme — **lazy training** (infinite width, parameters barely move). The other extreme is **rich/feature-learning** regime (finite width, parameters move substantially, representations change).

| Regime | Width | NTK | What's learned |
|--------|-------|-----|----------------|
| Lazy (NTK) | $\to \infty$ | Fixed throughout | Kernel interpolation; no feature learning |
| Feature learning | Finite | Changes during training | Representations evolve; can solve tasks kernel methods cannot |

**Key failure of NTK theory:** real-world networks operate in the feature learning regime, not the lazy regime. Experiments show the NTK changes substantially during training of practical networks (GPT, ResNet). NTK theory is exact in the limit but its quantitative predictions degrade as width decreases.

### NTK Limitations

1. **Finite-width corrections:** At finite width, the NTK evolves. The theory of **mean field** and **tensor programs** (Yang, 2019) extends NTK to track these corrections.

2. **Cannot explain grokking:** In the lazy regime, generalization happens immediately. The delayed generalization phenomenon (grokking) requires feature learning dynamics that NTK theory doesn't capture.

3. **Underestimates representation power:** NTK-equivalent kernel regression fails to match neural networks on structured tasks (e.g., parity functions, image classification) because the kernel is fixed and doesn't adapt to the data.

4. **Spectral bias as a double-edged sword:** while spectral bias means networks generalize well on smooth functions, it also means they **cannot efficiently learn high-frequency targets** (e.g., cryptographic functions) without architectural modifications like Fourier feature encodings.

### NTK and Generalization

Since the NTK regime produces kernel regression, its generalization is governed by classical kernel learning theory:

$$\text{test error} \lesssim \frac{\|y\|_{K^{-1}}^2}{n}$$

where $\|y\|_{K^{-1}}^2 = y^\top K^{-1} y$ = RKHS norm of the target in the NTK's reproducing kernel Hilbert space, $K$ = NTK Gram matrix on training inputs, $n$ = training set size. Small RKHS norm = target is "smooth" w.r.t. the NTK = good generalization. Functions that are **smooth in the NTK RKHS** generalize well with few samples; high-frequency functions require exponentially many samples.

---

## Links

- [[kernel-methods]] — the NTK is a kernel: $K_{\text{NTK}}(x,x') = \nabla_\theta f(x)\cdot\nabla_\theta f(x')$; in the infinite-width limit, training dynamics are governed by kernel regression with this kernel
- [[matrix-calculus]] — the NTK is the Gram matrix of the model's Jacobian; computing it requires differentiating the network output with respect to all parameters
- [[generalization-bounds]] — NTK theory gives PAC-Bayes-style bounds for overparameterized networks; the effective complexity is the NTK's trace, not the parameter count
- [[spectral-bias]] — the NTK's eigenfunctions are ordered by frequency; the lowest-frequency components (largest eigenvalues) are learned first, explaining spectral bias of wide networks
- [[bias-variance-double-descent]] — NTK theory proves that interpolating (zero training error) models can still generalize; the minimum-norm solution (NTK regression) corresponds to the second descent
- [[optimizer-adam]] — the NTK analysis assumes gradient flow (infinitesimal learning rate); finite learning rates (Adam, SGD with large $\eta$) move networks out of the lazy training regime
