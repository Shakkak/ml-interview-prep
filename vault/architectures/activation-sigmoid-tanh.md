---
title: Sigmoid and Tanh
tags: [activation, training, probability]
aliases: [sigmoid, tanh, logistic function, saturating activations]
difficulty: 1
status: complete
related: [activation-relu-variants, activation-softmax, loss-cross-entropy, backpropagation-advanced]
---

# Sigmoid and Tanh

---

## Fundamental

**Sigmoid** and **tanh** were the original activation functions for neural networks. Sigmoid squashes any real input into $(0, 1)$, making it a natural output for probabilities. Tanh squashes into $(-1, 1)$, giving zero-centered outputs. Both suffer from vanishing gradients in deep networks — the main reason ReLU replaced them in hidden layers — but they still play specific roles in gates and output layers.

$$\sigma(x) = \frac{1}{1 + e^{-x}} \qquad \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**Sigmoid:** range $(0, 1)$. Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$. Maximum gradient at $x=0$: $\sigma'(0) = 0.25$.

**Tanh:** range $(-1, 1)$. Derivative: $\tanh'(x) = 1 - \tanh^2(x)$. Maximum gradient at $x=0$: $\tanh'(0) = 1$ — four times larger than sigmoid.

| $x$ | $\sigma(x)$ | $\sigma'(x)$ | $\tanh(x)$ | $\tanh'(x)$ |
|-----|-------------|--------------|------------|-------------|
| $-3$ | $0.047$ | $0.045$ | $-0.995$ | $0.010$ |
| $-1$ | $0.269$ | $0.197$ | $-0.762$ | $0.420$ |
| $0$ | $0.500$ | $0.250$ | $0.000$ | $1.000$ |
| $1$ | $0.731$ | $0.197$ | $0.762$ | $0.420$ |
| $3$ | $0.953$ | $0.045$ | $0.995$ | $0.010$ |

Both saturate for $|x| > 2$: gradient $\approx 0$.

---

## Intermediate

### The Vanishing Gradient Problem

In a 10-layer sigmoid network, the gradient at layer 1 involves:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_{10}} \cdot \prod_{k=2}^{10} \sigma'(z_k) \cdot W_k$$

If neurons are in their saturated regime, $\sigma'(z_k) \approx 0.05$ each. Then:
$$\text{gradient magnitude} \propto (0.05)^{9} \approx 2 \times 10^{-12}$$

Effectively zero. This is why sigmoid was abandoned for hidden layers.

### Zero-Centering: Why Tanh Is Better Than Sigmoid for Hidden Layers

When activations $a_j > 0$ always (sigmoid, [[activation-relu-variants|ReLU]]), the gradient of the loss w.r.t. weight $w_j$ is:

$$\frac{\partial L}{\partial w_j} = \delta \cdot a_j$$

where $\delta = \frac{\partial L}{\partial z}$ can be positive or negative, but $a_j > 0$ always. This means all gradients $\frac{\partial L}{\partial w_j}$ in a layer have the same sign as $\delta$. All weights in that layer update in the same direction — gradients zig-zag rather than moving diagonally toward the optimum in weight space. Zero-centered outputs (tanh) eliminate this sign-consistency constraint (see [[backpropagation]] for the full gradient chain).

### When to Use Each

- **Sigmoid:** binary classification output (probability), [[rnn-lstm|LSTM]] gates (forget, input, output). **Not** hidden layers.
- **Tanh:** LSTM cell state updates, hidden layers in shallow RNNs where zero-centering matters.
- **Sigmoid vs [[activation-softmax|softmax]]:** sigmoid per-class allows multi-label (cat AND dog simultaneously); softmax enforces mutual exclusivity (exactly one class).

---

## Advanced

### Numerical Stability: Log-Sigmoid

Computing $\log \sigma(x)$ directly causes underflow for large negative $x$:
$$\sigma(-100) \approx 3.7 \times 10^{-44} \to \text{float32 underflow to 0} \to \log(0) = -\infty$$

Numerically stable form using the softplus function:
$$\log \sigma(x) = -\log(1 + e^{-x}) = -\text{softplus}(-x)$$

For $x \ll 0$: $\approx x$ (exact). For $x \gg 0$: $\approx -e^{-x} \approx 0$. Framework implementations handle this automatically via fused ops.

### Sigmoid in LSTM Gates

In an LSTM, the forget gate is $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$. Sigmoid is the correct choice here — not tanh — because gates must output values in $[0,1]$ to control how much of the cell state passes through. The cell state itself uses tanh to remain zero-centered:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)$$

The tanh squashes values to $[-1, 1]$, preventing unbounded cell state growth. The design choice of tanh for cell states and sigmoid for gates is deliberate and mechanistic — not arbitrary.

### Softplus: The Smooth Envelope of ReLU

$$\text{softplus}(x) = \log(1 + e^x) \approx \begin{cases} e^x & x \ll 0 \\ x & x \gg 0 \end{cases}$$

This is a smooth approximation of ReLU. Its derivative is $\sigma(x)$. Rarely used as a hidden activation due to computational cost, but it appears in energy-based models and as a natural log-normalizer.

---

*See also: [[activation-relu-variants]] · [[activation-softmax]] · [[loss-cross-entropy]] · [[backpropagation-advanced]] · [[backpropagation]] · [[rnn-lstm]]*
