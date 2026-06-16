---
title: Backpropagation
tags: [training, gradients, calculus, fundamentals]
aliases: [backprop, reverse-mode autodiff]
difficulty: 1
status: complete
depends_on: [matrix-calculus, activation-relu-variants]
related: [backpropagation-advanced, normalization-layers]
---

# Backpropagation

---

## Fundamental

**Backpropagation** is how neural networks learn — it computes the gradient of the loss with respect to every weight in the network. The core idea is the chain rule of calculus applied layer by layer from output back to input: each layer passes responsibility for the error backwards to the layers before it. Without backpropagation, training a deep network would be computationally intractable.

A neural network is a composed function $f_\theta : \mathbb{R}^n \to \mathbb{R}^m$. For a fully connected network with $L$ layers:

$$a^{(0)} = x \quad \text{(input)}$$
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \quad \text{(pre-activation)}$$
$$a^{(l)} = \sigma(z^{(l)}) \quad \text{(post-activation)}$$

where $l$ indexes layers ($l = 1, \ldots, L$), $W^{(l)}$ and $b^{(l)}$ are the weight matrix and bias at layer $l$, $a^{(l-1)}$ is the previous layer's output, $z^{(l)}$ is the pre-activation (before applying the nonlinearity), $\sigma$ is the activation function (e.g., ReLU or sigmoid — see [[activation-relu-variants]]), and $a^{(l)}$ is the post-activation output passed to the next layer.

Training finds $\theta$ minimizing a loss $L(\hat{y}, y)$. This requires $\frac{\partial L}{\partial \theta}$ for every parameter — the role of backpropagation.

**The chain rule** ([[matrix-calculus]]) is the entire mathematical engine:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

where $\partial L/\partial y$ = how the loss changes with intermediate quantity $y$ (already known from upstream), $\partial y/\partial x$ = local Jacobian (how $y$ changes with input $x$), product = how the loss changes with $x$.

For vector functions, this generalizes to Jacobian multiplication.

**Forward pass:** compute and store all intermediate activations $a^{(l)}$ and pre-activations $z^{(l)}$, then compute the loss.

**Backward pass:** define the error signal $\delta^{(l)} \triangleq \frac{\partial L}{\partial z^{(l)}}$ and propagate it:

$$\delta^{(L)} = \frac{\partial L}{\partial a^{(L)}} \odot \sigma'(z^{(L)})$$
$$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

where:
- $\delta^{(l)} \triangleq \frac{\partial L}{\partial z^{(l)}}$ — the error signal at layer $l$: how much the loss changes if we nudge the pre-activation $z^{(l)}$
- $\frac{\partial L}{\partial a^{(L)}}$ — gradient of the loss w.r.t. the final layer's output (depends on the loss function; e.g., for MSE this is $2(\hat{y} - y)$)
- $\sigma'(z^{(l)})$ — the derivative of the activation function evaluated at the pre-activation $z^{(l)}$; for ReLU: $\sigma'(z) = \mathbf{1}[z > 0]$
- $\odot$ — element-wise (Hadamard) multiplication
- $\left(W^{(l+1)}\right)^T$ — transpose of the weight matrix at the next layer; routes the error signal backwards

Parameter gradients follow directly:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

where $a^{(l-1)}$ is the previous layer's output (stored during the forward pass), giving the gradient for the weight matrix, and $\delta^{(l)}$ directly gives the gradient for the bias.

**Worked example — 1-neuron network (ReLU + MSE):**

Network: $z_1 = w_1 x$, $a_1 = \text{ReLU}(z_1)$ ([[activation-relu-variants]]), $\hat{y} = w_2 a_1$, $L = (\hat{y}-y)^2$.
Given: $x=2, y=1, w_1=0.5, w_2=3$.

Forward: $z_1=1$, $a_1=1$, $\hat{y}=3$, $L=4$.

Backward:
$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y}-y) = 4, \quad \frac{\partial L}{\partial w_2} = 4 \cdot a_1 = 4$$
$$\frac{\partial L}{\partial a_1} = 4 \cdot w_2 = 12, \quad \frac{\partial a_1}{\partial z_1} = \mathbf{1}[z_1>0] = 1$$
$$\frac{\partial L}{\partial w_1} = 12 \times 1 \times x = 24$$

The gradient update then moves each weight in the direction of steepest descent: $W \leftarrow W - \eta \frac{\partial L}{\partial W}$.

Crucially, the forward pass must store all intermediate activations $a^{(l)}$ for use in Step 5 — this is why training memory is proportional to depth × batch size. *Gradient checkpointing* trades compute for memory by recomputing activations during the backward pass rather than storing them.

---

## Intermediate

**Two-layer worked example (ReLU + sigmoid + cross-entropy):**

Network: $x \in \mathbb{R}^2$, hidden layer (ReLU), output layer (sigmoid), binary cross-entropy.

Forward:
$$z^{(1)} = W^{(1)}x + b^{(1)},\; a^{(1)} = \text{ReLU}(z^{(1)})$$
$$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)},\; \hat{y} = \sigma(z^{(2)})$$ ([[activation-sigmoid-tanh]])
$$L = -y\log\hat{y} - (1-y)\log(1-\hat{y})$$ ([[loss-cross-entropy|binary cross-entropy]])

Backward (the key simplification for sigmoid + cross-entropy):
$$\frac{\partial L}{\partial z^{(2)}} = \hat{y} - y$$

This clean form arises because the product $\frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z) = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$ telescopes exactly to $\hat{y} - y$. This is why sigmoid + cross-entropy is the canonical output pairing.

Continuing backward:
$$\frac{\partial L}{\partial W^{(2)}} = (\hat{y}-y) \cdot (a^{(1)})^T$$
$$\delta^{(1)} = (W^{(2)})^T(\hat{y}-y) \odot \mathbb{1}[z^{(1)} > 0]$$
$$\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} \cdot x^T$$

**Activation functions and gradient fate:**

| Activation | $\sigma'$ range | Gradient fate |
|-----------|----------------|---------------|
| Sigmoid | $[0, 0.25]$ | Vanishes quickly (×0.25 per layer) |
| Tanh | $[0, 1]$ | Vanishes, but slower |
| ReLU | $\{0, 1\}$ | Passes through or dies permanently |
| LeakyReLU | $\{0.01, 1\}$ | Almost never dies |
| [[activation-gelu-swish\|GELU]] | $\approx [0, 1]$ | Smooth, good empirical behavior |

**Dead ReLU problem:** if a neuron's pre-activation is always negative, $\sigma'(z) = 0$ forever, the neuron never receives updates and is permanently dead. Causes: large negative bias initialization or a very high learning rate early in training. Fixes: LeakyReLU or careful initialization.

**Mini-batch backprop:** gradients are averaged over a batch of $B$ samples, $\frac{\partial L}{\partial W} = \frac{1}{B}\sum_{i=1}^B \frac{\partial L_i}{\partial W}$, which is equivalent to the matrix math with an added batch dimension.

---

## Advanced

**Reverse-mode vs forward-mode autodiff:** backprop is reverse-mode automatic differentiation — it propagates a single scalar (the loss) backward through the computation graph. It requires one forward pass and one backward pass regardless of the number of parameters. Forward-mode AD propagates an input perturbation forward and costs one pass *per input dimension*, making it efficient for wide-input/narrow-output functions (e.g., [[matrix-calculus|Jacobian]] vector products) but impractical for training. Modern frameworks (PyTorch, JAX) implement both; reverse-mode is used for training, forward-mode for Jacobian–vector products during second-order methods.

**Gradient checkpointing tradeoffs:** storing all $L$ layers of activations requires $O(L \cdot B \cdot d)$ memory. Gradient checkpointing stores only $O(\sqrt{L})$ evenly spaced checkpoints and recomputes the others during the backward pass, reducing activation memory to $O(\sqrt{L} \cdot B \cdot d)$ at the cost of one extra forward pass. This is the standard technique for training very deep or long-context models (Chen et al., 2016).

**Backprop through non-differentiable operations:** ReLU is non-differentiable at 0; the standard practice uses the subgradient $\mathbf{1}[z > 0]$ (setting the gradient to 0 at exactly 0). More careful treatments use smooth approximations (softplus, GELU) or straight-through estimators for discrete operations. The straight-through estimator (Bengio et al., 2013) replaces the gradient of a step function with 1 during backprop — it is used in VQ-VAE, binary neural networks, and attention masking.

**Implicit differentiation and DEQ models:** Deep Equilibrium Models (Bai et al., 2019) define the network as the fixed point $z^* = f_\theta(z^*)$. Backprop computes gradients through this implicit equation via the implicit function theorem without storing any intermediate activations: $\frac{\partial z^*}{\partial \theta} = -(J_{z^*} - I)^{-1} \frac{\partial f}{\partial \theta}$, where $J_{z^*}$ is the Jacobian of $f$ at the fixed point. This reduces memory from $O(L)$ to $O(1)$ for arbitrarily deep networks.

**Second-order methods and Hessian-free optimization:** the Newton step $\Delta\theta = -H^{-1}g$ uses the Hessian $H = \nabla^2 L$, which is $O(p^2)$ to store and $O(p^3)$ to invert for $p$ parameters. Hessian-free (Martens, 2010) approximates $Hv$ via finite differences in gradient space, enabling Conjugate Gradient descent without ever forming $H$. K-FAC (Martens & Grosse, 2015) approximates the Fisher information matrix as a Kronecker product, giving a tractable second-order optimizer for neural networks that significantly outperforms Adam on certain tasks.

---

## Links

- [[matrix-calculus]] — provides the chain rule and Jacobian foundations that backprop is built on
- [[activation-relu-variants]] — gradient behavior of each activation directly determines what backprop computes at hidden layers
- [[activation-sigmoid-tanh]] — sigmoid's vanishing gradient (max 0.25 per layer) is the key failure backprop exposes in deep nets
- [[loss-cross-entropy]] — the loss function whose gradient initiates the backward pass
- [[backpropagation-advanced]] — extends to forward-mode autodiff, Jacobian-vector products, and implicit differentiation
- [[normalization-layers]] — BatchNorm changes the gradient landscape that backprop traverses
- [[optimizer-adam]] — uses the gradients computed here for adaptive per-parameter updates
- [[optimizer-sgd-momentum]] — applies gradients from here to update model parameters via velocity accumulation
