---
title: State Space Models and Mamba
tags: [state-space-models, ssm, mamba, s4, sequence-modeling, selective-ssm]
aliases: [SSM, S4, Mamba, selective state space model, linear recurrence, structured SSM]
difficulty: 3
status: complete
depends_on: [rnn-lstm, linear-algebra-fundamentals, fourier-transform]
related: [rnn-lstm, attention-mechanism, flash-attention, arch-kv-cache, autoregressive-models, fourier-transform]
---

# State Space Models and Mamba

---

## Fundamental

### Classical State Space Models

A **state space model** describes a sequence with a hidden state $h_t \in \mathbb{R}^N$ that evolves linearly:

$$h_t = A h_{t-1} + B x_t$$
$$y_t = C h_t + D x_t$$

where $x_t \in \mathbb{R}^d$ is the input, $y_t \in \mathbb{R}^d$ is the output, $A \in \mathbb{R}^{N \times N}$ is the state transition, $B \in \mathbb{R}^{N \times d}$ maps input to state, and $C \in \mathbb{R}^{d \times N}$ maps state to output.

**Recurrent form:** like an RNN (see [[rnn-lstm]]), the model maintains a state $h_t$ and updates it at each step. But unlike RNNs:
- $A, B, C$ are learned parameters, fixed across all time steps
- No nonlinearity in the state update (linear recurrence)
- No vanishing gradient problem for long sequences (under proper $A$ initialization)

**Convolutional form:** unrolling the recurrence gives:
$$y = \bar{K} * x, \qquad \bar{K} = (C\bar{B},\, CA\bar{B},\, CA^2\bar{B}, \ldots)$$

The sequence of outputs is a convolution of the input with the **SSM kernel** $\bar{K}$. This means SSMs can be computed as:
- **Training:** parallel convolution (fast, like BERT)
- **Inference:** sequential recurrence (constant memory, like LSTM)

The best of both worlds: efficient training + efficient autoregressive inference.

---

## Intermediate

### S4: Structured State Space Sequences

The challenge with classical SSMs: the state matrix $A$ is $N \times N$. If $N = 64$, the convolution kernel has $64$-dimensional states — manageable. But computing $CA^k B$ for long sequences requires $O(L^2N)$ operations naively.

**S4 (Gu et al., 2022)** uses a **diagonal-plus-low-rank (DPLR)** structure for $A$: $A = \Lambda - PQ^*$ where $\Lambda$ is diagonal. This allows computing the SSM kernel in $O(L \log L)$ via Cauchy kernels and the fast Fourier transform.

**HiPPO initialization:** critically, $A$ is initialized using HiPPO (High-order Polynomial Projection Operators) theory — an $A$ matrix designed to compress all previous inputs into the state optimally (under $L^2$ norm). Concretely, the HiPPO-LegS matrix makes $h_t$ encode the coefficients of the Legendre polynomial projection of the input history $x_{0:t}$. This initialization allows S4 to remember long-range dependencies from the start.

**S4 performance:** first SSM to match Transformers on long-range benchmarks (Path-X, ListOps requiring 16,000+ token context). Also efficient for raw audio (SampleRate-SSM, 16kHz audio = 16,000 steps per second).

### Mamba: Selective SSMs

**The limitation of S4:** the parameters $A, B, C$ are fixed — every input uses the same dynamics. This is suboptimal: for language, whether to remember "John" depends on whether we later encounter "he" referring to him — content-dependent selectivity is needed.

**Mamba (Gu & Dao, 2023)** introduces **input-dependent (selective) parameters**:

$$B_t = \text{linear}(x_t), \quad C_t = \text{linear}(x_t), \quad \Delta_t = \text{softplus}(\text{linear}(x_t))$$

where $\Delta_t$ is the discretization step size. Now $B_t$ and $C_t$ vary with the input — the model can selectively choose what to remember and what to ignore.

**Discretization:** continuous-time $A$ is discretized using $\Delta_t$:
$$\bar{A}_t = \exp(\Delta_t A), \qquad \bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I)\Delta_t B_t$$

where $A \in \mathbb{R}^{N \times N}$ = the (fixed) continuous-time state transition matrix, $\Delta_t > 0$ = the input-dependent discretization step size (computed by a linear layer + softplus from $x_t$), $\bar{A}_t$ = the discrete state transition for step $t$ (via matrix exponential — exact for linear ODEs), and $\bar{B}_t$ = the corresponding discrete input matrix (zero-order hold method). Large $\Delta_t$ → strong dependence on current input (ignore state). Small $\Delta_t$ → strong dependence on state (ignore current input). $\Delta_t$ becomes a learned gating mechanism — analogous to forget gate in LSTM.

**Why selective parameters break the parallel convolution:** with input-dependent $\bar{B}_t, \bar{A}_t$, the kernel $\bar{K}$ changes at every step — can't precompute it. Mamba uses a **hardware-aware parallel scan** algorithm: process the recurrence in $O(\log L)$ parallel steps using prefix-scan on GPU (like parallel prefix sum). This uses SRAM (fast) rather than HBM (slow) for intermediate states.

---

## Advanced

### Mamba Architecture

The Mamba block replaces the standard Transformer block:

```
Input → Linear + SiLU gate  →  SSM  →  × (gate)  →  Linear  →  Output
                     ↑
              input-dependent B, C, Δ
```

The gated MLP structure (similar to SwiGLU in LLaMA) surrounds the SSM layer. The SSM dimension is expanded $d_{\text{state}} = 16$ or $64$; typically $N = 16$ hidden state dimensions per channel.

**Parameter efficiency:** a Mamba layer with state size $N$ uses $O(dN)$ parameters for $B, C$, $O(1)$ for $A$ (diagonal), and $O(d^2)$ for the linear projections. Total: similar to a Transformer layer with the attention replaced.

**Mamba-2:** introduces **SSD (State Space Duality)** — proves that certain SSM classes are equivalent to a form of linear attention. This allows applying ideas from efficient attention research to SSMs and vice versa.

### Mamba vs Transformer

| Property | Transformer | Mamba |
|----------|------------|-------|
| Context | $O(L^2)$ attention (full context access) | $O(L)$ sequential (bounded state) |
| Training | Parallel, efficient | Parallel via scan |
| Inference | $O(L)$ per step (KV cache) | $O(1)$ per step (fixed state) |
| Memory at inference | $O(L \cdot d)$ KV cache grows | $O(Nd)$ fixed state |
| Long-range dependencies | Strong (full attention) | Good but lossy (bounded state) |
| Copying and retrieval | Excellent | Weaker (discrete tokens hard to copy) |
| Current scaling | Dominant (GPT, LLaMA) | Competitive at smaller scales |

**Where Mamba excels:** very long sequences (genomics, audio at sample level, time series), inference-constrained settings (constant memory), streaming applications (no growing KV cache).

**Where Transformers win:** tasks requiring exact retrieval of earlier content (in-context learning, few-shot reasoning), current large-scale pretraining infrastructure is optimized for attention.

### Hybrid Architectures

Jamba (AI21 Labs, 2024) interleaves Mamba and Transformer layers: Mamba layers handle long-range compression, Transformer layers handle precise retrieval. This hybrid achieves better than pure Mamba on retrieval tasks while maintaining linear inference cost for long sequences.

---

## Links

- [[rnn-lstm]] — SSMs generalize linear RNNs; Mamba's selective mechanism adds the gating of LSTMs to the SSM framework
- [[linear-algebra-fundamentals]] — SSMs are defined by matrix multiplication $h_t = Ah_{t-1} + Bx_t$; HiPPO initialization designs $A$ to preserve long-range history
- [[fourier-transform]] — SSMs with diagonalized $A$ can be computed as convolutions via FFT; the connection to spectral methods explains their sequence-length efficiency
- [[attention-mechanism]] — SSMs are the main alternative to attention for long-context modeling; Mamba achieves linear $O(N)$ complexity vs. attention's $O(N^2)$
- [[flash-attention]] — both target efficient sequence modeling; Flash Attention speeds up quadratic attention while SSMs avoid the quadratic cost entirely
- [[arch-kv-cache]] — SSMs maintain a fixed-size state $h_t \in \mathbb{R}^d$ analogous to KV cache, but with fixed memory regardless of context length
- [[autoregressive-models]] — Mamba can generate autoregressively with $O(1)$ per-step cost; transformer autoregressive generation is $O(N)$ per step with KV cache
