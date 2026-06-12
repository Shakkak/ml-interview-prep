---
title: Loss Landscape & Optimization Geometry
tags: [training, optimization, loss-landscape, saddle-points, sharpness]
aliases: [loss surface, loss landscape, flat minima, sharp minima, saddle points]
difficulty: 3
status: complete
related: [backpropagation, backpropagation-advanced, optimizer-adam, optimizer-sgd-momentum, optimizer-lr-schedules, normalization-layers, generalization-bounds, bias-variance-double-descent]
---

# Loss Landscape & Optimization Geometry

---

## Fundamental

### What the Loss Landscape Is

For a neural network with $p$ parameters, the loss $L : \mathbb{R}^p \to \mathbb{R}$ defines a surface in $(p+1)$-dimensional space. For a ResNet-50 with 25M parameters, this is a 25-million-dimensional surface that we navigate using gradient descent — a process no human intuition can directly visualize.

**Types of critical points** (where $\nabla L = 0$):
- **Global minimum:** $L(\theta) \leq L(\theta')$ for all $\theta'$ — the best possible solution.
- **Local minimum:** $L(\theta) \leq L(\theta')$ for all $\theta'$ in a neighborhood — a basin you can get stuck in.
- **Saddle point:** a minimum in some directions, maximum in others — gradient is zero but it's not a minimum. Exponentially more common than local minima in high dimensions.
- **Plateau:** flat region where $\|\nabla L\| \approx 0$ but $L$ is not at a minimum — slows down gradient-based methods.

**Practical observation:** for modern deep neural networks, most critical points encountered during training are saddle points, not local minima. The loss landscape of overparameterized networks has many paths to good (but not necessarily global) minima, and the main obstacle to training is saddle points and flat regions, not bad local minima.

### Flat vs Sharp Minima and Generalization

Two minima with the same training loss can have very different test performance. The key variable is the **sharpness** of the basin — how fast the loss rises as you move away from the minimum.

**Flat minimum:** loss stays low in a wide neighborhood. Small perturbations to weights don't change performance much — the solution is robust. **Sharp minimum:** loss rises steeply in a small neighborhood. The model relies on a precise weight configuration that may not generalize.

The **sharpness** is formally the largest eigenvalue $\lambda_\max$ of the Hessian $H = \nabla^2 L$ at the minimum. Flat: $\lambda_\max$ small. Sharp: $\lambda_\max$ large.

**Why flat minima generalize better:** consider perturbing weights by a random noise vector $\delta \sim \mathcal{N}(0, \sigma^2 I)$. The change in loss is approximately $\delta^\top H \delta / 2$. For a flat minimum (small $\lambda_\max$), this perturbation barely affects performance; for a sharp minimum, it can be catastrophic. Since test inputs differ from training inputs — effectively perturbing the distribution — flat minima are more robust to this distribution shift.

**Which optimizers find flatter minima:** SGD with small batch sizes and appropriate learning rate tends to find flatter minima than Adam. The stochastic gradient noise prevents the optimizer from settling into narrow sharp basins — see [[optimizer-sgd-momentum]].

---

## Intermediate

### Hessian Spectrum in Practice

For a typical neural network at a well-trained minimum, the Hessian has a characteristic **bulk-edge structure:**
- A large bulk of near-zero eigenvalues (many directions of near-zero curvature — the overparameterization manifold)
- A small number of large "outlier" eigenvalues corresponding to the most sensitive directions

The ratio $\kappa = \lambda_\max / \lambda_\min$ is the **condition number** — large $\kappa$ means the loss surface is elongated like a valley: flat in most directions, steep in a few. Gradient descent oscillates in the steep directions and crawls in the flat ones. This is why adaptive optimizers (Adam) help: they normalize by gradient magnitude, partially correcting for the poor conditioning.

**The largest eigenvalue as a training diagnostic:** $\lambda_\max$ can be estimated cheaply via the power iteration on the Hessian-vector product $Hv$ (obtainable in two backprop passes without forming $H$). Cohen et al. (2021) showed $\lambda_\max$ reliably tracks the "edge of stability" threshold $2/\eta$ during training — a useful signal for detecting when the learning rate is too high.

### Mode Connectivity

Perhaps the most surprising fact about deep network loss landscapes: **different minima are often connected by low-loss paths**. Garipov et al. (2018) showed that two independently trained models can be connected by a curved path through parameter space along which the loss stays near-minimum. **Why:** overparameterized networks have a vast solution manifold; the "minima" are actually all part of the same connected region, just accessed from different directions.

**Practical implications:**
1. **Snapshot ensembles:** train a single model, save snapshots along a cyclic LR schedule — each snapshot is at a different "mode." Ensemble these for free with no extra training cost (Huang et al., 2017).
2. **Loss barrier between two individually trained models:** there IS a high-loss barrier between the weight spaces of two separately trained models (because random symmetry permutations break the naive path). Git Re-Basin (Ainsworth et al., 2022) resolves this by matching neuron permutations before interpolating, enabling model merging along low-loss paths.
3. **Neural network merging (model soups, Wortsman et al., 2022):** averaging the weights of multiple fine-tuned models (from the same pretrained base) often outperforms any individual model — because the models share the same basin structure due to the common initialization, so weight interpolation stays in a low-loss region.

### Sharpness-Aware Minimization (SAM)

SAM (Foret et al., 2021) directly optimizes for flat minima by minimizing the maximum loss within a neighborhood:

$$\min_\theta \max_{\|\delta\| \leq \rho} L(\theta + \delta)$$

The inner maximization finds the worst perturbation $\delta^*$; the outer minimization moves to reduce even the worst-case perturbed loss. The first-order approximation: $\delta^* \approx \rho \frac{g}{\|g\|}$, so SAM's update computes the gradient at $\theta + \rho g / \|g\|$ (one extra forward+backward pass) and uses that for the descent step.

**SAM on ImageNet:** consistently 0.5–1.5% improvement in top-1 accuracy at the cost of 2× compute. Particularly effective for fine-tuning pretrained models where sharpness correlates strongly with overfitting.

### Loss Landscape Visualization

For networks with millions of parameters, standard 2D/3D visualization is impossible. Li et al. (2018) proposed plotting loss along two random or structured directions from a trained minimum:

$$f(\alpha, \beta) = L(\theta^* + \alpha d_1 + \beta d_2)$$

where $d_1, d_2$ are filter-normalized random directions (normalized to match each weight filter's magnitude). This removes the scale ambiguity that otherwise makes different architectures' landscapes incomparable.

**Key findings from landscape visualization:**
- ResNets (with skip connections) have dramatically smoother, more convex-looking landscapes than VGG (no skip connections)
- Wider networks generally have flatter, more benign landscapes
- BatchNorm smooths the landscape significantly (Santurkar et al., 2018)
- Very deep networks without skip connections develop landscapes with ridges and chaotic regions that are nearly impossible to optimize

---

## Advanced

### Double Descent and the Loss Landscape

The classical bias-variance tradeoff predicts that model performance degrades past the interpolation threshold (where training error = 0). The **double descent** phenomenon (Belkin et al., 2019; Nakkiran et al., 2020) shows that for overparameterized models, performance recovers and continues improving past this threshold — see [[bias-variance-double-descent]].

**Loss landscape interpretation:** the interpolation threshold corresponds to the phase transition where the loss landscape changes character:
- **Underparameterized regime:** the loss has a unique minimum (or very few) far from zero; the landscape is well-conditioned
- **At the threshold:** the system is critically parameterized; many near-degenerate minima exist; the landscape is ill-conditioned with many saddle points
- **Overparameterized regime:** the solution manifold is a continuous high-dimensional set of zero-loss solutions; gradient descent finds the minimum-norm solution via implicit bias of the optimizer

The minimum-norm solution in the overparameterized regime generalizes better than you'd expect — the implicit bias of gradient descent selects solutions that are flat in the overparameterization directions.

### Implicit Bias of Gradient Descent

Gradient descent doesn't just find any minimum — it has a systematic bias toward specific types of solutions. For linear models and infinitely wide networks (NTK regime):

- **Full-batch gradient descent** converges to the **minimum-norm** solution (minimum $\|\theta\|_2$) when starting from $\theta = 0$
- **SGD** is biased toward solutions that are robust to gradient noise — effectively toward flatter minima
- **SGD with small batch** finds flatter minima than large batch (verified empirically and theoretically via the Langevin diffusion approximation)
- **Adam** finds sharper minima than SGD+momentum due to coordinate-wise adaptation (Wilson et al., 2017)

The implicit bias is itself a form of regularization — the optimizer's convergence path induces a prior over solutions. Understanding this bias is key to understanding why certain architectural choices (normalization, residual connections) and optimizer choices work as well as they do.

### The Edge of Stability

Cohen et al. (2021) documented a surprising phenomenon in full-batch gradient descent training of deep networks: the **sharpness progressively increases during training** until it stabilizes at exactly $2/\eta$ — the theoretical threshold for divergence of gradient descent on quadratics ($\eta\lambda_\max < 2$ required for convergence). The name "edge of stability" reflects that training proceeds just at the boundary of divergence.

**Why does sharpness stabilize at $2/\eta$?** Near a minimum, gradient descent on a quadratic with curvature $\lambda$ follows $\theta_{t+1} = (1 - \eta\lambda)\theta_t$. For $\eta\lambda > 2$, this oscillates and diverges. For a neural network, the loss landscape is not quadratic, so training can persist at $\eta\lambda_\max = 2$ through a balancing act — the optimizer oscillates in the sharp directions while making progress in the flat ones.

**Implication for learning rate choice:** the maximum stable LR is $\eta < 2/\lambda_\max$. Since $\lambda_\max$ increases during training, a fixed LR eventually hits the edge of stability. LR schedules (warmup + decay) implicitly manage this by starting with a high LR (accepting EOS instability during warmup) and then decaying as the model approaches a minimum (stabilizing the sharp directions).

---

*See also: [[backpropagation-advanced]] · [[optimizer-sgd-momentum]] · [[optimizer-adam]] · [[optimizer-lr-schedules]] · [[generalization-bounds]] · [[bias-variance-double-descent]] · [[normalization-layers]]*
