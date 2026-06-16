---
title: Numerical Methods & Floating-Point Arithmetic
tags: [numerical-methods, floating-point, stability, precision, mixed-precision]
aliases: [numerical stability, floating point, log-sum-exp, numerical precision]
difficulty: 2
status: complete
related: [mixed-precision, taylor-series, matrix-calculus, loss-cross-entropy, activation-softmax, linear-algebra-fundamentals]
depends_on: [linear-algebra-fundamentals, taylor-series]
---

# Numerical Methods & Floating-Point Arithmetic

---

## Fundamental

### Floating-Point Representation

Modern computers represent real numbers in **IEEE 754** floating-point format: $\pm m \times 2^e$, where $m$ is the mantissa and $e$ is the exponent.

| Format | Total bits | Mantissa bits | Exponent bits | Approx. decimal digits | Max value | Min positive |
|--------|-----------|---------------|---------------|------------------------|-----------|--------------|
| float64 (double) | 64 | 52 | 11 | ~15–16 | ~$1.8 \times 10^{308}$ | ~$5 \times 10^{-324}$ |
| float32 (single) | 32 | 23 | 8 | ~7–8 | ~$3.4 \times 10^{38}$ | ~$1.2 \times 10^{-38}$ |
| bfloat16 | 16 | 7 | 8 | ~2–3 | ~$3.4 \times 10^{38}$ | ~$1.2 \times 10^{-38}$ |
| float16 | 16 | 10 | 5 | ~3–4 | 65504 | ~$6 \times 10^{-8}$ |

**Key takeaway:** float16 has max value 65504 and only 5 exponent bits, making it prone to overflow and underflow. bfloat16 shares float32's exponent range but sacrifices mantissa precision — better for ML where range matters more than precision.

**Machine epsilon $\epsilon_\text{mach}$:** the smallest number $\epsilon$ such that $1 + \epsilon \neq 1$ in floating-point arithmetic. For float32: $\epsilon_\text{mach} \approx 1.19 \times 10^{-7}$. This is the fundamental precision limit — any computation involving numbers that differ by less than $\epsilon_\text{mach} \times |x|$ will be indistinguishable.

### Catastrophic Cancellation

When two nearly equal numbers are subtracted, most significant bits cancel and the result has few significant digits:

$$a = 1.000000001, \quad b = 1.000000000, \quad a - b = 0.000000001$$

In float32 (7 decimal digits): both $a$ and $b$ round to 1.000000, and $a - b = 0$ — **all precision is lost**.

**ML example — computing variance numerically:**
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

For activations with mean $\approx 100$ and variance $\approx 0.01$: $\mathbb{E}[X^2] \approx 10000.01$ and $(\mathbb{E}[X])^2 \approx 10000.00$. Their difference in float32 loses all significant digits. **Fix:** use the numerically stable Welford's online algorithm, which computes variance as a running update without computing large squares.

---

## Intermediate

### Log-Sum-Exp and Softmax Stability

The softmax denominator $Z = \sum_k e^{z_k}$ overflows for large logits and underflows for very negative ones.

**Overflow example:** if $z_k = 1000$, then $e^{1000} = \infty$ in float32 (max is $\approx 3.4 \times 10^{38} \approx e^{88}$).
**Underflow example:** if all $z_k = -1000$, then all $e^{z_k} = 0$, $Z = 0$, and softmax is $0/0$ = NaN.

**The log-sum-exp trick:** subtract the maximum before exponentiation:

$$\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}, \qquad m = \max_k z_k$$

After shifting, the maximum exponent is $e^0 = 1$, all others are $\leq 1$ — no overflow. The log pulls the shifted sum back to the right scale. **This is mathematically exact**, not an approximation.

**Numerically stable softmax:**
```python
def softmax_stable(z):
    m = z.max()
    z_shifted = z - m
    exp_z = np.exp(z_shifted)        # max exp is e^0 = 1
    return exp_z / exp_z.sum()       # denominator >= 1, always valid
```

**Numerically stable cross-entropy (log-softmax):**
```python
def log_softmax_stable(z):
    m = z.max()
    z_shifted = z - m
    log_sum = np.log(np.exp(z_shifted).sum())
    return z_shifted - log_sum  # exact: log(e^(z-m)) - log(sum e^(z-m))
```

PyTorch's `F.cross_entropy(logits, targets)` internally uses this — never compute `torch.log(torch.softmax(logits, dim=-1))` manually.

### Condition Number and Numerical Stability

The **condition number** $\kappa(A) = \|A\| \cdot \|A^{-1}\| = \sigma_\max / \sigma_\min$ (ratio of largest to smallest singular value) measures how sensitive the solution $Ax = b$ is to perturbations:

$$\frac{\|\Delta x\|}{\|x\|} \leq \kappa(A) \cdot \frac{\|\Delta b\|}{\|b\|}$$

A relative error in $b$ of size $\epsilon$ can cause relative errors in $x$ of size $\kappa(A) \cdot \epsilon$. For float32 precision ($\epsilon \approx 10^{-7}$), a system with $\kappa(A) = 10^{10}$ is numerically useless — the solution has no significant digits.

**In ML — why learning rate matters for conditioning:** the Hessian $H$ of the loss has condition number $\kappa(H) = \lambda_\max / \lambda_\min$. Gradient descent converges at rate $(\kappa-1)/(\kappa+1)$ per step — it takes $O(\kappa)$ steps to reduce the error by a factor $1/e$. Poorly conditioned Hessians (large $\kappa$) require many small steps or adaptive methods. [[normalization-layers|Batch normalization]] smooths the loss landscape and reduces the effective condition number.

**Normal equations for linear regression:** $X^\top X w = X^\top y$ — the condition number of $X^\top X$ is $\kappa(X)^2$. If $X$ has condition number $10^3$, $X^\top X$ has condition number $10^6$. For numerically robust regression, use QR decomposition on $X$ directly rather than forming $X^\top X$.

### Gradient Accumulation and Mixed Precision

**float16 gradient underflow:** gradients in neural networks typically have very small magnitude (e.g., $10^{-8}$). float16 can't represent values below $\approx 6 \times 10^{-8}$ — gradients smaller than this round to zero, killing learning. 

**Loss scaling** (used in mixed-precision training) multiplies the loss by a large factor $s$ (e.g., $s = 2^{15}$) before backprop, scaling all gradients by $s$ and preventing underflow. Gradients are then unscaled before the optimizer step. Dynamic loss scaling adjusts $s$ automatically: if any gradient is inf/nan, halve $s$ and retry; if no inf/nan for $N$ consecutive steps, double $s$.

```
Standard training:    forward → loss → backward → optimizer step
Mixed precision:      fp16 forward → fp16 loss × scale → fp16 backward
                      → unscale gradients (fp32) → check overflow
                      → fp32 optimizer step → update fp16 weights
```

**bfloat16 advantage:** same exponent range as float32, so no overflow/underflow for gradients. No loss scaling needed. The 7 mantissa bits (vs float32's 23) reduce precision but this is acceptable because gradients don't need high precision — direction matters more than magnitude.

### Numerical Stability of Batch Normalization

The BN denominator $\sqrt{\sigma^2 + \epsilon}$ adds $\epsilon$ (typically $10^{-5}$) to prevent division by zero. But there's a subtler issue: **variance explosion in the backward pass**.

The BN gradient involves terms like $\hat{x}_i^2 / \text{Var}(x)$. When the variance is very small (all activations nearly identical), $\hat{x}_i$ can be very large, causing gradient instability. In float16, the squared normalized activations can overflow. 

**Solution used in practice:** compute BN statistics in float32 even when activations are in float16 (PyTorch's `torch.cuda.amp.autocast` does this automatically for BN). This is why some layers are "not autocast" in mixed-precision training.

---

## Advanced

### Kahan Summation

Naive floating-point summation $\sum_{i=1}^n x_i$ accumulates error: after $n$ additions, absolute error is $O(n \epsilon_\text{mach} \cdot |x_i|)$. For $n = 10^6$ terms in float32, this is $O(10^{-1})$ — catastrophic.

**Kahan's compensated summation** reduces error to $O(\epsilon_\text{mach} \cdot |x_i|)$ independent of $n$:

```python
def kahan_sum(xs):
    total = 0.0
    compensation = 0.0  # running correction for lost low-order bits
    for x in xs:
        y = x - compensation          # compensate for previous error
        t = total + y                 # add corrected value
        compensation = (t - total) - y  # capture lost bits
        total = t
    return total
```

**Why it works:** `compensation` tracks the bits that were lost in the previous addition. On the next step, those bits are subtracted from the incoming value — effectively adding them back. The final error is $O(2\epsilon_\text{mach})$ regardless of $n$.

**ML applications:** any large sum where precision matters — gradient accumulation over many small batches, log-likelihood computation over many tokens, importance weight sums in REINFORCE.

### Numerically Stable Implementations of Common Functions

**Stable sigmoid:** $\sigma(z) = 1/(1 + e^{-z})$. For $z \ll 0$, $e^{-z}$ overflows.

```python
def sigmoid_stable(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
```

**Stable log(1 + exp(z)) (softplus):** for large positive $z$, $\log(1 + e^z) \approx z + \log(1 + e^{-z}) \approx z$:

```python
def softplus_stable(z):
    return np.where(z > 20, z, np.log1p(np.exp(np.minimum(z, 20))))
```

**Stable logsumexp of two values:** $\log(e^a + e^b) = a + \log(1 + e^{b-a})$ (assuming $a \geq b$). Reduces to computing $\log(1 + e^x)$ for $x = b - a \leq 0$ — always safe.

### Automatic Mixed Precision and the Stability Taxonomy

PyTorch's AMP (`torch.cuda.amp`) classifies operations into three categories:

1. **fp16 safe:** matmul, conv2d, linear — benefit most from fp16 speed (4× throughput on Tensor Cores), and their errors are acceptable
2. **fp32 required:** softmax, cross-entropy, batch norm statistics, log, exp, division — these require higher precision to avoid catastrophic cancellation or overflow
3. **fp32 promote:** operations with mixed precision inputs are promoted to fp32

The design principle: do the bulk computation (matmul, which dominates FLOPS) in fp16 for speed, but accumulate results and compute reductions in fp32 for correctness. This is why fp16 matmul followed by fp32 accumulation is the standard practice — Tensor Cores compute $A \times B$ in fp16 but accumulate the dot products in fp32 internally.

---

## Links

- [[linear-algebra-fundamentals]] — ill-conditioned matrices cause numerical instability in linear solves; the condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ quantifies how many bits of precision are lost
- [[taylor-series]] — numerical differentiation (finite differences) is derived from the Taylor expansion; truncation error is $O(h)$ for forward differences, $O(h^2)$ for central differences
- [[mixed-precision]] — FP16/BF16 has 3–4 fewer decimal digits of precision than FP32; numerical stability techniques (loss scaling, stable softmax) prevent underflow/overflow
- [[activation-softmax]] — numerically stable softmax subtracts $\max(x)$ before exponentiating; this prevents overflow without changing the output (since it cancels in numerator and denominator)
- [[loss-cross-entropy]] — log-sum-exp is the numerically stable way to compute $\log \sum_i e^{x_i}$; PyTorch's `CrossEntropyLoss` fuses softmax + log + NLL for stability
- [[normalization-layers]] — layer/batch norm adds $\epsilon$ to the denominator to prevent division by zero; $\epsilon=10^{-5}$ is standard
