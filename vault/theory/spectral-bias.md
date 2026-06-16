---
title: Spectral Bias in Neural Networks
tags: [theory, deep-learning, generalization, fourier, ntk]
aliases: [spectral bias, frequency principle, NTK, neural tangent kernel]
difficulty: 3
status: complete
related: [fourier-transform, bias-variance-double-descent, attention-mechanism, arch-positional-encoding]
depends_on: [fourier-transform, neural-tangent-kernel, backpropagation]
---

# Spectral Bias in Neural Networks

---

## Fundamental

**Spectral bias** (Rahaman et al., 2019; also called "Frequency Principle" by Xu et al., 2019):

> Neural networks trained with gradient descent learn low-frequency components of the target function first, and progressively fit higher-frequency components.

**The empirical observation.** Train an MLP to fit $f(x) = \sin(3x) + 0.5\sin(10x) + 0.1\sin(30x)$. Track the [[fourier-transform|Fourier]] spectrum of the error $f - \hat{f}(t)$ over training:
- **Early:** error dominated by low-frequency (3 Hz) components — these fit first.
- **Mid:** 10 Hz component begins to fit.
- **Late:** 30 Hz component fits last.

The network learns in frequency order: low → high.

**Why this matters for practitioners:**
- **Early stopping** = implicit low-pass filtering of the target: signal (smooth) fits before noise (irregular)
- **Sharp edges, fine textures** are slow to learn and may need longer training
- **[[arch-positional-encoding|Positional encodings]]** in transformers inject high-frequency information that would otherwise be learned slowly

**Practical implications table:**

| Situation | Effect of spectral bias |
|---|---|
| Image classification | Coarse shape learned before fine texture |
| Early stopping | Biases toward smooth, generalizable solutions |
| High-frequency tasks (edges, OCR) | Needs longer training or explicit Fourier features |
| Sinusoidal positional encoding | Injects high-frequency position directly |
| NeRF / coordinate networks | Fourier feature encoding is essential |

---

## Intermediate

### NTK Analysis: Why Low Frequencies Come First

For infinitely wide neural networks trained with gradient flow, training dynamics are governed by the **[[neural-tangent-kernel|Neural Tangent Kernel (NTK)]]**:

$$K(x, x') = \langle \nabla_\theta f(x;\theta), \nabla_\theta f(x';\theta) \rangle$$

In the infinite-width limit, $K$ is constant during training. The dynamics become a linear ODE. The error in the direction of NTK eigenfunction $\phi_k$ with eigenvalue $\lambda_k$ decays as:

$$\epsilon_k(t) = \epsilon_k(0)\, e^{-\lambda_k t}$$

where $\epsilon_k(t)$ = error component in the direction of NTK eigenfunction $\phi_k$ at training step $t$, $\epsilon_k(0)$ = initial error in that direction, $\lambda_k$ = NTK eigenvalue for eigenfunction $\phi_k$ (governs decay speed).

**Key fact:** the NTK of deep networks has higher eigenvalues for low-frequency eigenfunctions and lower eigenvalues for high-frequency ones. Therefore:
- Low frequency: high $\lambda_k$ → fast decay → learned first ✓
- High frequency: low $\lambda_k$ → slow decay → learned last ✓

This is the mathematical foundation of spectral bias.

### Early Stopping as Low-Pass Filtering

Stopping at time $t$ means the model has learned: $\hat{f}(x) \approx \mathcal{F}^{-1}[F_{\text{target}}(\omega) \cdot (1 - e^{-\lambda(\omega)t})]$ where $\lambda(\omega)$ decreases with frequency $\omega$.

For high-frequency components with small $\lambda(\omega)$, the factor $(1 - e^{-\lambda(\omega)t}) \approx 0$ — they haven't been learned yet. Early stopping therefore acts as a **frequency-domain low-pass filter**.

Since data labels are approximately $y = f_{\text{signal}}(x) + \text{noise}(x)$ where signal is smooth (low-frequency) and noise is irregular (high-frequency), early stopping captures signal while discarding noise — explaining why it improves generalization.

### Connection to Double Descent and Overparameterization

Spectral bias explains why **overparameterized models still generalize** in the interpolation regime (double descent). Very large networks can memorize training data exactly, yet test error is low because:
1. Gradient descent first learns the smooth low-frequency signal common to all data
2. Only then starts fitting high-frequency training-specific noise
3. At infinite time, the minimum-norm interpolating solution is found — which, for gradient descent initialization near zero, is still smooth

This is the spectral bias explanation for generalization without explicit regularization (see [[bias-variance-double-descent]]).

---

## Advanced

### Fourier Analysis of the NTK Spectrum

For a network with ReLU activations, the NTK in 1D has eigenvalues that decay polynomially with frequency: $\lambda_k \propto k^{-2s}$ where $s$ depends on depth and activation. For $L$-layer networks, the bias is stronger ($s$ increases) with depth.

The **RKHS (Reproducing Kernel Hilbert Space)** associated with the NTK contains functions with finite energy $\|f\|_\mathcal{H}^2 = \sum_k |c_k|^2 / \lambda_k$. High-frequency components with small $\lambda_k$ contribute disproportionately to the RKHS norm. Minimum-norm interpolation (what gradient descent finds) produces solutions with small $\|f\|_\mathcal{H}^2$ — i.e., solutions that suppress high frequencies. This is spectral bias from the RKHS perspective.

**Activation function affects the bias.** ReLU is a piecewise linear activation with NTK that has polynomial frequency decay. **Sinusoidal activations** (SIREN, Sitzmann et al., 2020) use $\sigma(x) = \sin(\omega_0 x)$ — the NTK eigenspectrum is flat across frequencies, enabling uniform learning of all frequencies. SIREN networks learn sharp discontinuities and high-frequency details much faster than ReLU networks — essential for representing implicit neural surfaces (NeRF, SDFs).

### Fourier Feature Networks and NeRF

For tasks that require high-frequency representation (neural radiance fields, neural SDFs, image regression at high resolution), the standard fix is **random Fourier features** (Tancik et al., 2020):

$$\gamma(x) = [\cos(2\pi B x), \sin(2\pi B x)]^\top, \quad B \in \mathbb{R}^{m\times d} \sim \mathcal{N}(0, \sigma^2)$$

where $B$ = random frequency matrix ($m$ frequencies, each $d$-dimensional), $m$ = number of random Fourier features, $\sigma$ = bandwidth controlling the frequency range represented (larger $\sigma$ → higher frequencies).

The standard deviation $\sigma$ controls which frequencies are represented. This preprocessing lifts the spectral bias by providing explicit high-frequency basis functions — the network only needs to learn the coefficients, not discover the frequencies.

**NeRF (Mildenhall et al., 2020):** without positional encoding, the MLP produces blurry reconstructions. With the sinusoidal encoding at 10 frequency octaves, it reproduces sharp color and geometry. The encoding is precisely overcoming spectral bias to enable high-frequency scene representation.

### Spectral Bias in Transformers and Attention Dynamics

The connection to transformers: [[attention-mechanism|attention]] sharpening over training mirrors spectral bias in MLPs.

**Early training:** attention weights are diffuse (high effective temperature) — the model attends broadly, capturing low-frequency spatial patterns (global structure).
**Late training:** attention heads sharpen — attending to specific positions and local patterns (high-frequency relationships).

Attention heads that specialize to positional patterns (local syntactic dependencies in NLP, adjacent patches in vision) emerge earlier in training than heads capturing long-range semantic dependencies — consistent with low-frequency-first learning.

**Sinusoidal positional encodings** (Vaswani et al., 2017) address exactly this: learning position-dependent behavior is a high-frequency task (the model must represent $\sin(\text{pos}/10000^{2i/d})$ for many frequencies $i$). By injecting these directly as input features, the network has immediate access to position information at all frequencies without having to learn it through the slow high-frequency path.

This also explains why **learned positional embeddings** (as in BERT) can underperform sinusoidal encodings for long-range extrapolation: the learned embeddings only capture the frequencies present in training sequence lengths, while sinusoidal encodings naturally extrapolate to unseen lengths via their analytic form.

---

## Links

- [[fourier-transform]] — spectral bias is defined in terms of Fourier frequency components; networks learn low-frequency functions (large spatial scales) before high-frequency ones (fine details)
- [[neural-tangent-kernel]] — the NTK eigenvalues determine the learning speed for each frequency; low-frequency eigenfunctions (large eigenvalues) are learned first, explaining spectral bias
- [[backpropagation]] — spectral bias emerges from gradient descent dynamics, not architecture; the gradient flow preferentially aligns with low-frequency eigenspaces of the NTK
- [[bias-variance-double-descent]] — spectral bias is a form of implicit regularization: networks converge to minimum-norm solutions biased toward low-frequency functions
- [[attention-mechanism]] — transformers with RoPE/sinusoidal encodings are less affected by spectral bias; position encodings inject specific frequencies into the network from the start
- [[arch-positional-encoding]] — Fourier positional encodings (NeRF's positional encoding) explicitly inject high-frequency inputs to overcome spectral bias for reconstruction tasks
