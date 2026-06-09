---
title: Recurrent Neural Networks (RNN, LSTM, GRU)
tags: [rnn, lstm, gru, sequence-modeling, recurrent-networks, bptt]
aliases: [RNN, LSTM, GRU, recurrent neural network, vanilla RNN, gated recurrent unit, BPTT, backpropagation through time, long short-term memory, sequence model]
difficulty: 2
status: complete
related: [backpropagation-advanced, attention-mechanism, activation-sigmoid-tanh, autoregressive-models, variational-autoencoders]
---

# Recurrent Neural Networks (RNN, LSTM, GRU)

---

## Fundamental

**The core idea:** process a sequence $x_1, \ldots, x_T$ one step at a time, carrying a hidden
state $h_t$ that summarizes everything seen so far:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h), \qquad y_t = W_{hy} h_t + b_y$$

The same weight matrices $W_{hh}, W_{xh}, W_{hy}$ are reused at every time step — this weight
sharing is what lets one fixed-size parameter set handle sequences of any length.

**The forward pass:**
```
h = h_0                      # often the zero vector
for t in 1..T:
    h = tanh(W_hh @ h + W_xh @ x[t] + b_h)
    y[t] = W_hy @ h + b_y
return y, h
```

**Worked example.** Toy 1-D RNN with $W_{hh} = 0.5$, $W_{xh} = 1.0$, $b_h = 0$, $h_0 = 0$, and
input sequence $x = (1, 1, 1)$:
- $h_1 = \tanh(0.5 \cdot 0 + 1.0 \cdot 1) = \tanh(1) \approx 0.762$
- $h_2 = \tanh(0.5 \cdot 0.762 + 1.0 \cdot 1) = \tanh(1.381) \approx 0.881$
- $h_3 = \tanh(0.5 \cdot 0.881 + 1.0 \cdot 1) = \tanh(1.440) \approx 0.894$

Each $h_t$ folds in everything before it through $h_{t-1}$ — a sequence of any length gets
compressed into one fixed-size vector. That compression is also the architecture's central
weakness (see Advanced).

> [!tip] Why $\tanh$, not ReLU or sigmoid, drives the hidden-state update
> [[activation-sigmoid-tanh|Tanh]]'s output is bounded in $(-1,1)$ and zero-centered. Bounded
> keeps $h_t$ from blowing up across hundreds of recurrent steps the way an unbounded activation
> (ReLU) could; zero-centered avoids the gradient sign-consistency problem that
> [[activation-sigmoid-tanh]] describes for ordinary feedforward layers — here it would
> compound at every single time step.

---

## Intermediate

### The Vanishing/Exploding Gradient Problem (BPTT)

Training unrolls the recurrence across all $T$ steps — **backpropagation through time (BPTT)**.
[[backpropagation-advanced]] derives the resulting gradient product
$\frac{\partial h_t}{\partial h_{t-1}} = W_{hh}^\top \cdot \mathrm{diag}(\sigma'(z_t))$ in full;
in short, the gradient reaching step 1 is a product of $T-1$ such Jacobians, so it shrinks
toward 0 (vanishing) or grows without bound (exploding) exponentially in $T$ unless every
factor stays close to 1. With $T$ in the hundreds this leaves vanilla RNNs unable to learn
long-range dependencies — by the time an error signal reaches an early step it has been
multiplied by $\sigma'(z_k) \le 0.25$ a hundred times over.

### LSTM — Long Short-Term Memory

LSTMs address this by adding a **cell state** $c_t$ that flows through time via *additive*,
*gated* updates rather than repeated matrix multiplication:

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) &&\text{forget gate} \\
i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) &&\text{input gate} \\
o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o) &&\text{output gate} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tanh(W_c[h_{t-1}, x_t] + b_c) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

> [!tip] Why sigmoid for gates and tanh for content — the load-bearing slice
> [[activation-sigmoid-tanh]] explains this design choice mechanistically: gates must output
> values in $[0,1]$ to act as "how much passes through" multipliers — exactly sigmoid's range —
> while the cell-state candidate and the cell-to-hidden projection need zero-centered, bounded
> values (tanh's range) to keep $c_t$ from growing unboundedly across steps. See that file's
> "Sigmoid in LSTM Gates" section for the full mechanistic argument; the gate equations above
> are the architecture this choice lives inside.

When the forget gate $f_t \approx 1$, gradients flow through $c_t$ across many steps almost
unchanged — additive updates don't repeatedly squash the signal the way $\tanh(W_{hh}h_{t-1})$
does in a vanilla RNN. That's the "long" in "Long Short-Term Memory": information persists over
long ranges once the gates learn to keep it.

### GRU — a Simplified Alternative

The Gated Recurrent Unit merges the forget and input gates into one **update gate** $z_t$ and
drops the separate cell state, folding everything into $h_t$:

$$
\begin{aligned}
z_t &= \sigma(W_z[h_{t-1}, x_t]) &&\text{update gate} \\
r_t &= \sigma(W_r[h_{t-1}, x_t]) &&\text{reset gate} \\
\tilde h_t &= \tanh(W[r_t \odot h_{t-1},\, x_t]) \\
h_t &= (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t
\end{aligned}
$$

Roughly 25% fewer parameters than an LSTM at the same hidden size, with competitive performance
on many tasks — often the "try this first" default before reaching for a full LSTM.

---

## Advanced

### Why Attention Replaced Recurrence

[[attention-mechanism]] states the core argument concisely: a recurrent network must compress
an arbitrarily long prefix into one fixed-size vector $h_t$ — an information bottleneck — and
its $O(T)$ sequential dependency chain blocks the parallel computation that makes transformer
training fast on GPUs. Attention lets every position look directly at every other position in
$O(1)$ sequential steps, removing the bottleneck and the parallelism problem at once. This is
*the* historical pivot: LSTMs augmented with attention (seq2seq) were state of the art in
machine translation right up until the Transformer (2017) showed recurrence could be dropped
entirely.

### Bidirectional and Seq2Seq Architectures

**Bidirectional RNN:** run two RNNs over the sequence — one forward, one backward — and
concatenate their hidden states at each position. Doubles the context available at every step;
standard for tasks where the whole sequence is available upfront (tagging, classification), but
unusable for autoregressive generation, where future tokens are unavailable by construction
(see [[autoregressive-models]]).

**Seq2seq (encoder–decoder):** an encoder RNN compresses the source sequence into a final
hidden state; a decoder RNN generates the target conditioned on it. [[attention-mechanism|Attention]]
was *invented* to fix exactly this bottleneck — letting the decoder look back at every encoder
hidden state instead of relying on one compressed vector (Bahdanau et al., 2014) — and that
insight was later generalized into the fully attention-based Transformer.

### Where Recurrence Still Matters

Despite being displaced from large-scale language modeling, recurrence persists in: streaming
and low-latency inference (constant per-step memory vs. a growing [[arch-kv-cache|KV cache]]),
small-scale time-series and control problems, and as a component inside larger architectures —
e.g. an RNN aggregator over neighbor messages in [[graph-neural-networks|graph neural networks]]
or an RNN text decoder in [[variational-autoencoders|VAEs]]. Recent state-space models (S4,
Mamba) revisit the recurrent formulation with parallelizable training, aiming to combine
RNN-style $O(1)$ inference with transformer-style training throughput.

---

*See also: [[backpropagation-advanced]] · [[attention-mechanism]] · [[activation-sigmoid-tanh]] · [[autoregressive-models]] · [[variational-autoencoders]] · [[graph-neural-networks]] · [[arch-kv-cache]]*
