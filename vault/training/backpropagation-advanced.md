---
title: Backpropagation — Advanced Topics
tags: [training, gradients, deep-learning, jacobians]
aliases: [vanishing gradients, exploding gradients, Jacobian, gradient flow]
difficulty: 2
status: complete
depends_on: [backpropagation, matrix-calculus, activation-sigmoid-tanh]
related: [backpropagation, normalization-layers, attention-mechanism]
---

# Backpropagation — Advanced Topics

---

## Fundamental

**Prerequisites:** read [[backpropagation]] first. This file covers the Jacobian formulation, gradient flow through non-trivial operations, vanishing/exploding gradients, gradient clipping, and weight initialization.

For a vector-to-vector function $y = f(x)$, the Jacobian is $J_{ij} = \frac{\partial y_i}{\partial x_j}$. The chain rule for backprop becomes:

$$\frac{\partial L}{\partial x} = J^T \frac{\partial L}{\partial y}$$

where $J = \partial y/\partial x \in \mathbb{R}^{m \times n}$ = Jacobian of $f$ (how each output $y_i$ changes with each input $x_j$), $J^T$ = transpose routes gradient from output space back to input space, $\partial L/\partial y$ = upstream gradient.

In deep networks, the total gradient is a product of per-layer Jacobians:

$$\frac{\partial L}{\partial x^{(0)}} = J_1^T J_2^T \cdots J_L^T \frac{\partial L}{\partial x^{(L)}}$$

where $J_l = \partial x^{(l)}/\partial x^{(l-1)}$ = per-layer Jacobian, product $J_1^T \cdots J_L^T$ = total gradient propagation (vanishes/explodes when singular values of $J_l$ are consistently $<1$ or $>1$).

This product is the mathematical root cause of vanishing and exploding gradients.

**Vanishing gradients:** if each Jacobian $J_l$ has singular values $\sigma_l < 1$, the product shrinks exponentially. For [[activation-sigmoid-tanh|sigmoid]] activations, $\sigma'(z) \leq 0.25$ — after 10 layers, gradients shrink by $0.25^{10} \approx 10^{-6}$.

**Exploding gradients:** if singular values $\sigma_l > 1$ throughout, the product grows exponentially — catastrophic in [[rnn-lstm|RNNs]].

**Weight initialization and variance preservation:** for a layer $y = Wx$, $\text{Var}(y_i) = n_{in} \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_j)$. For variance to be preserved, we need $\text{Var}(W_{ij}) = 1/n_{in}$.

- **Xavier/Glorot** (for tanh/sigmoid): $\text{Var}(W) = \frac{2}{n_{in}+n_{out}}$
- **He/Kaiming** (for ReLU, which zeros half the activations): $\text{Var}(W) = \frac{2}{n_{in}}$

---

## Intermediate

**Gradient flow through softmax:** the Jacobian of softmax $y_i = e^{x_i}/\sum_j e^{x_j}$ is:
$$\frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j), \quad J_{softmax} = \text{diag}(y) - yy^T$$

where $\delta_{ij}$ = Kronecker delta (1 if $i=j$, 0 otherwise), $y_i(\delta_{ij}-y_j)$ = off-diagonal: $-y_iy_j$ (captures correlation between outputs), $J_\text{softmax} \in \mathbb{R}^{K \times K}$ = symmetric matrix with rank $K-1$.

When composed with [[loss-cross-entropy|cross-entropy]] $L = -\sum_k t_k \log y_k$, the result simplifies beautifully: $\frac{\partial L}{\partial x_i} = y_i - t_i$. This is why [[activation-softmax|softmax]] + cross-entropy is the canonical output pairing — the gradient is prediction minus label.

*Saturation warning:* when one logit dominates ($y \approx (1,0,\ldots,0)$), gradients for non-maximum classes are near zero. The model has "decided" and stops learning from those classes, contributing to overconfidence and poor calibration.

**Gradient flow through convolution:** convolution $y = x * w$ is linear in both $x$ and $w$:
- $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} * \text{flip}(w)$ — correlation with flipped kernel
- $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} * x$ — correlation between input and upstream gradient

Backprop through a conv layer is itself a convolution — the same hardware-efficient operation.

**Gradient flow through [[normalization-layers|batch normalization]]:** BN introduces sample-interdependence in its Jacobian. For $\hat{x}_i = (x_i - \mu_B)/\sqrt{\sigma_B^2+\epsilon}$ and $y_i = \gamma\hat{x}_i + \beta$:

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_B^2+\epsilon}}\left[\frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_j \frac{\partial L}{\partial y_j} - \hat{x}_i \cdot \frac{1}{m}\sum_j \frac{\partial L}{\partial y_j}\hat{x}_j\right]$$

where $\gamma$ = learned BN scale, $\sigma_B^2$ = batch variance, $m$ = batch size, $\hat{x}_i$ = normalized activation, three bracketed terms = gradients through $\hat{x}_i$, $\mu_B$ (mean), and $\sigma_B^2$ (variance) respectively.

The three terms arise from differentiating through $\hat{x}_i$, $\mu_B$, and $\sigma_B^2$ respectively. The $1/\sqrt{\sigma_B^2+\epsilon}$ factor normalizes gradient scale — this is why BN keeps Jacobian singular values near 1.

**Gradient clipping:**

*Norm clipping (recommended):*
$$\text{if } \|g\| > c: \quad g \leftarrow g \cdot \frac{c}{\|g\|}$$

where $g$ = concatenated gradient vector over all parameters, $\|g\|$ = global L2 norm, $c$ = clipping threshold, $c/\|g\|$ = rescaling factor (brings gradient to norm $c$).

Preserves gradient direction, only reduces magnitude. Standard in RNN and transformer training.

*Value clipping:* $g_i \leftarrow \text{clamp}(g_i, -c, c)$. Simple but distorts direction when components clip unevenly — generally inferior.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Backpropagation through time (BPTT) for RNNs:** unrolling $T$ steps gives:
$$\frac{\partial L}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial L}{\partial h_T}, \quad \frac{\partial h_t}{\partial h_{t-1}} = W_{hh}^T \cdot \text{diag}(\sigma'(z_t))$$

where $h_t$ = hidden state at step $t$, $W_{hh}$ = recurrent weight matrix, $\sigma'(z_t)$ = activation derivative at step $t$, product over $T$ terms = cause of vanishing/exploding gradients for long sequences.

For $T = 100$ steps, this product causes catastrophic vanishing or exploding regardless of activation choice. This is the fundamental motivation for LSTM gating and transformer [[attention-mechanism|self-attention]].

> [!tip] How LSTM gating breaks the vanishing-gradient chain ([[rnn-lstm]])
> An LSTM maintains a **cell state** $c_t$ that flows through time with only additive updates:
> $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
> where $f_t \in (0,1)^d$ is the forget gate (learned per-timestep). The gradient of the loss w.r.t. $c_{t-1}$ is $f_t$ — not a matrix product through $\tanh'$. When $f_t \approx 1$, the gradient flows back with near-unit magnitude for as many steps as the forget gate stays open, sidestepping the compounding shrinkage that kills vanilla RNN gradients.

---

## Advanced

**The edge of stability phenomenon (Cohen et al., 2021):** during gradient descent training of neural networks, the sharpness (largest eigenvalue of the Hessian $\lambda_\text{max}$) progressively increases until it reaches $2/\eta$ and then stabilizes there — the "edge of stability." Beyond this threshold, vanilla GD diverges; SGD noise and the adaptive normalization in Adam prevent it empirically. This explains why small learning rates are not always safer: they allow the loss landscape to become sharper, and the model ends up in a sharper (worse-generalizing) region.

**[[neural-tangent-kernel|Neural tangent kernel (NTK)]] and infinite-width limit:** for infinitely wide networks, gradient descent training is described by the NTK $\Theta(x, x') = \sum_l \frac{\partial f(x)}{\partial \theta_l} \cdot \frac{\partial f(x')}{\partial \theta_l}$, which remains constant throughout training (kernel regime). The NTK view explains why wide networks converge to global minima and generalizes kernel regression theory to neural networks (Jacot et al., 2018). Finite networks deviate from the NTK regime — they are in the "feature learning" regime where representations change during training, empirically explaining why practical networks outperform their NTK predictions.

**Gradient pathologies in transformers:** in pre-norm transformers, the residual stream grows in norm throughout the forward pass (the residual accumulation problem). At depth $L$, the norm of the residual can scale as $O(\sqrt{L})$, causing gradient scales to differ between layers. This motivates $\mu$P (maximal update parameterization, Yang et al., 2022), which rescales weights to ensure activation and gradient norms are independent of width and depth, enabling hyperparameters to transfer across model scales — a key technique used in scaling large language models.

**Gradient flow in diffusion models:** diffusion model training (Ho et al., 2020) optimizes a denoising objective $\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$. The gradient signal flows through an extremely simple loss (MSE between noise vectors), but the model architecture $\epsilon_\theta$ is deep. The time embedding $t$ enters through adaptive layer normalization (AdaLN) where $\gamma, \beta$ are predicted from $t$, creating multiplicative gradient pathways. Analysis of gradient flow through the AdaLN parameterization reveals that early timesteps (high noise) receive stronger gradient signal — models tend to fit coarse structure before fine detail.

**Straight-through estimator (STE) and discrete backprop:** when a function is piecewise constant (argmax, sign, quantization), its gradient is 0 everywhere — useless for learning. The STE (Bengio et al., 2013) approximates the gradient of the discrete operation with the identity: $\partial f_\text{discrete}/\partial x \approx 1$. This is used in VQ-VAE codebook learning, binary neural networks, and straight-through Gumbel-Softmax. While theoretically unjustified, it works empirically because the true gradient exists in an expectation sense across minibatches.

---

## Links

- [[backpropagation]] — prerequisite: basic chain rule and the forward/backward pass algorithm
- [[matrix-calculus]] — Jacobians and the matrix chain rule are the mathematical engine of everything in this file
- [[activation-sigmoid-tanh]] — sigmoid's $\sigma'(z) \leq 0.25$ is the root cause of vanishing gradients; this file quantifies the Jacobian consequence
- [[normalization-layers]] — BN's three-term gradient keeps Jacobian singular values near 1, solving the vanishing/exploding problem
- [[attention-mechanism]] — transformer gradient pathologies (residual norm growth, attention sinks) are analyzed here
- [[optimizer-adam]] — $\mu$P derives initialization and LR jointly to keep gradient norms $O(1)$ regardless of width/depth
- [[rnn-lstm]] — BPTT unrolls into a product of Jacobians; LSTM gates provide an additive gradient path that bypasses the chain
- [[neural-tangent-kernel]] — infinite-width networks in the kernel regime have gradients governed by the NTK; finite networks deviate
- [[loss-cross-entropy]] — softmax + cross-entropy gradient simplifies to $\hat{y} - y$; the Jacobian of softmax is derived here
