---
title: Gradient Clipping
tags: [gradient-clipping, exploding-gradients, training-stability, rnn, optimization]
aliases: [gradient norm clipping, gradient value clipping, exploding gradients]
difficulty: 1
status: complete
related: [backpropagation, rnn-lstm, optimizer-sgd-momentum, optimizer-adam, loss-landscape, initialization]
depends_on: [backpropagation, optimizer-sgd-momentum, loss-landscape]
---

# Gradient Clipping

---

## Fundamental

### The Exploding Gradient Problem

During backpropagation through deep or recurrent networks, gradients can grow exponentially as they propagate back through many layers. For an RNN unrolled over $T$ steps, the gradient involves products of $T$ Jacobians — if the spectral radius of each exceeds 1, the product grows as $\rho^T$.

**Exploding gradients** cause:
- NaN/Inf values in weights after a large update
- Sudden loss spikes ("loss explosion")
- Training divergence or instability around sharp loss regions

### Gradient Norm Clipping

**Clip by norm:** scale the entire gradient vector if its $\ell_2$ norm exceeds a threshold $\tau$:

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 \leq \tau \\ \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|_2} & \text{otherwise} \end{cases}$$

where $\mathbf{g}$ = the concatenated gradient vector over all model parameters, $\|\mathbf{g}\|_2$ = the global gradient L2 norm (square root of sum of squared gradients across all parameters), $\tau > 0$ = the clipping threshold, and $\frac{\mathbf{g}}{\|\mathbf{g}\|_2}$ = the unit gradient vector (same direction, scaled to length 1).

This preserves the gradient direction while bounding its magnitude. All parameters share a single global norm computation.

**Clip by value:** independently clip each component: $g_i \leftarrow \text{clip}(g_i, -\tau, \tau)$. Simpler but changes the gradient direction — generally inferior to norm clipping.

---

## Intermediate

### Choosing the Threshold

Common values: $\tau = 1.0$ (default in many frameworks for RNNs), $\tau = 5.0$ (for transformers). Rule of thumb: run a few steps, observe the gradient norm distribution, set $\tau$ at the 95th–99th percentile of typical norms.

**Effect on training:**
- Too small $\tau$: gradient artificially small everywhere, slow convergence
- Too large $\tau$: clipping rarely activates, no protection
- Optimal $\tau$: clips the rare catastrophic spikes without affecting normal steps

### When Clipping Is Needed

| Model type | Clipping needed? |
|-----------|-----------------|
| RNN/LSTM long sequences | Yes — almost always |
| Transformer pre-LN | Rarely — LayerNorm stabilizes |
| Transformer post-LN (original) | Sometimes, especially early training |
| CNNs with BatchNorm | Rarely |
| Reinforcement learning (policy gradient) | Yes — unbounded returns cause spikes |

Modern transformers (GPT-2, LLaMA) use clip by norm with $\tau = 1.0$ as standard practice even if not strictly necessary — it prevents rare catastrophic spikes that can derail long training runs.

### Interaction with Adaptive Optimizers

With Adam, the adaptive learning rate already dampens large gradients via the second-moment estimate. However, clipping still helps because:
- Adam's second moment adapts slowly (exponential moving average)
- A sudden large gradient can corrupt the $v_t$ estimate, affecting many future steps
- Clipping prevents $v_t$ corruption before it can occur

**Implementation in PyTorch:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Call after loss.backward(), before optimizer.step()
```

---

## Advanced

### Gradient Norm as a Diagnostic

The gradient norm over training is a useful diagnostic:
- **Steady norm:** healthy training
- **Increasing norm:** possible instability, consider reducing LR or increasing clipping
- **Norm always at clip threshold:** $\tau$ is too tight; gradients are always being clipped — increase $\tau$ or reduce LR

**Gradient norm spike patterns:** spikes often coincide with "hard" batches (outliers, difficult examples). In language models, sequences containing rare tokens or long dependencies generate larger gradients.

### Global vs Per-Layer Clipping

Standard clipping computes a single global norm across all parameters. **Per-layer clipping** clips each parameter group independently — sometimes used in LoRA fine-tuning to avoid the small LoRA adapter gradients being dominated by the large base model gradients (or vice versa).

## Links

- [[backpropagation]] — gradient clipping rescales the gradient vector before the parameter update; it does not change the gradient direction, only its magnitude
- [[rnn-lstm]] — LSTMs with gradient clipping were the first practical solution to exploding gradients; the LSTM's gating already addresses vanishing gradients
- [[optimizer-sgd-momentum]] — gradient clipping caps the effective step size; combined with momentum, it prevents the momentum buffer from accumulating catastrophically large updates
- [[loss-landscape]] — exploding gradients arise from sharp regions (large Hessian eigenvalues) in the loss landscape; gradient clipping is a heuristic that prevents steps into these regions
- [[initialization]] — good initialization (Xavier, He) reduces the frequency of gradient explosions; gradient clipping is a safety net for pathological cases that slip through
