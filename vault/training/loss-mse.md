---
title: Mean Squared Error (MSE) Loss
tags: [loss, regression, probability]
aliases: [MSE, L2 loss, squared error]
difficulty: 1
status: complete
depends_on: [statistical-inference-mle, distributions-gaussian]
related: [loss-huber, loss-cross-entropy, backpropagation, bayesian-inference]
---

# Mean Squared Error (MSE) Loss

---

## Fundamental

**Mean Squared Error (MSE)** is the standard loss function for regression. It measures the average squared difference between predictions and true values — a prediction twice as far from the target costs four times as much. Because squaring amplifies large errors, MSE encourages the model to avoid big mistakes, at the cost of being sensitive to outliers.

$$L_{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

where $N$ = number of samples in the batch, $y_i$ = true target value for sample $i$, and $\hat{y}_i$ = model's predicted value for sample $i$. The squared difference $(y_i - \hat{y}_i)^2$ is always non-negative and grows quadratically with the error.

Also written as $L = \frac{1}{2}\|y - \hat{y}\|^2$ (the $\frac{1}{2}$ is a convenience that cancels in the gradient).

**Gradient:** $\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)$ — proportional to the error. Larger errors produce larger gradients → faster correction for large mistakes.

**Probabilistic interpretation:** MSE is the [[statistical-inference-mle|MLE]] loss for a [[distributions-gaussian|Gaussian]] output distribution:
$$p(y | x; \theta) = \mathcal{N}(y; \hat{y}(x;\theta), \sigma^2)$$
$$\log p(y|x;\theta) = -\frac{(y - \hat{y})^2}{2\sigma^2} + \text{const}$$

where $\sigma^2$ = assumed noise variance (a constant), $\hat{y}(x;\theta)$ = model prediction, $(y-\hat{y})^2$ = squared error, negative log-likelihood proportional to MSE when $\sigma^2$ is fixed.

Maximizing log-likelihood = minimizing MSE. Choosing MSE as loss implicitly assumes Gaussian noise in the labels.

**When to use MSE:**
- Regression with unbounded real-valued output
- Noise is approximately Gaussian
- Large errors are disproportionately bad (quadratic penalty)
- Image reconstruction (pixel regression)

**Do not use MSE for classification** — MSE on logits or probabilities has gradient behavior that does not match the classification objective; use [[loss-cross-entropy|cross-entropy]] instead.

---

## Intermediate

**Gradient with respect to weights (linear model):** for $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$:
$$\nabla_\mathbf{w} L = \frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$$

where $\mathbf{X} \in \mathbb{R}^{n \times p}$ = data matrix ($n$ samples, $p$ features), $\mathbf{w} \in \mathbb{R}^p$ = weight vector, $\mathbf{y} \in \mathbb{R}^n$ = target vector, and $\mathbf{X}^\top$ = transpose of $\mathbf{X}$.

Setting to zero gives the **normal equations** (closed-form minimizer):
$$\mathbf{X}^\top\mathbf{X}\mathbf{w}^* = \mathbf{X}^\top\mathbf{y} \quad \Rightarrow \quad \mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

where $\mathbf{X}^\top\mathbf{X}$ = Gram matrix (must be invertible = full column rank), $\mathbf{X}^\top\mathbf{y}$ = cross-correlation between features and targets, $\mathbf{w}^*$ = least-squares solution (OLS estimator).

Requires $\mathbf{X}^\top\mathbf{X}$ to be invertible (full column rank, no perfectly correlated features).

**With L2 regularization (Ridge):**
$$\mathbf{w}^*_\text{ridge} = \left(\frac{1}{n}\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}\right)^{-1}\frac{1}{n}\mathbf{X}^\top\mathbf{y}$$

where $\lambda > 0$ = regularization strength and $\mathbf{I}$ = identity matrix (adds $\lambda$ to each diagonal entry of $\mathbf{X}^\top\mathbf{X}$, making it invertible even when features are collinear).

The $\lambda\mathbf{I}$ term ensures invertibility even when $\mathbf{X}^\top\mathbf{X}$ is rank-deficient (collinear features or $p > n$).

**MSE for image generation produces blurry outputs:** MSE averages over all plausible reconstructions. If multiple values of a pixel are plausible (e.g., texture, fine details), the MSE-optimal prediction is their mean — which is blurry. Perceptual losses (LPIPS), adversarial losses (GANs), and [[diffusion-models|diffusion model]] objectives produce sharper results by avoiding this averaging.

**Properties summary:**

| | MSE (L2) | L1 | [[loss-huber\|Huber]] |
|---|---|---|---|
| Gradient near 0 | Small (fast convergence) | Constant (slow) | Small (fast) |
| Gradient for outliers | Large (unstable) | Bounded | Bounded |
| Optimal under | Gaussian noise | Laplace noise | Mixed |
| Closed form | Yes (linear model) | No | No |

---

## Advanced

**MSE vs perceptual loss in neural image compression:** MSE optimizes peak signal-to-noise ratio (PSNR), which is $10\log_{10}(255^2/\text{MSE})$. However, human perception is not well-modeled by PSNR. Images with high PSNR can look blurry and perceptually worse than images with lower PSNR. The rate-distortion tradeoff changes depending on whether the distortion metric is MSE or LPIPS: at the same bitrate, LPIPS-optimized codecs produce sharper, perceptually better results with lower PSNR, while MSE-optimized codecs minimize pixel-wise error at the cost of sharpness. Blau & Michaeli (2018) formalized this as the perception-distortion tradeoff: better perceptual quality necessarily requires higher distortion under MSE.

**MSE and mode collapse in generative models:** for a conditional generative model $p(y|x)$ with multiple modes, MSE training minimizes $\mathbb{E}[(\hat{y} - y)^2]$, whose minimizer is the conditional mean $\hat{y}^* = \mathbb{E}[y|x]$. When $p(y|x)$ is multimodal, this mean lies between modes — a point of low probability under the true distribution. This is the formal explanation for MSE-trained image-to-image networks producing blurry outputs in ambiguous regions (e.g., predicting future video frames, colorization, super-resolution). Flow matching and diffusion models sidestep this by modeling the full conditional distribution.

**Connection to Gauss-Markov theorem:** the Gauss-Markov theorem states that the OLS estimator $\hat{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ is the Best Linear Unbiased Estimator (BLUE) when errors are uncorrelated, zero-mean, and homoscedastic. "Best" means minimum variance among all linear unbiased estimators. This is a strong theoretical justification for MSE in linear regression — but the conditions (homoscedasticity, no autocorrelation) often fail in practice, explaining why robust regression (Huber) and heteroscedastic regression (modeling variance as a function of $x$) are needed for real data.

**Heteroscedastic regression and learned uncertainty:** standard MSE assumes a fixed noise variance $\sigma^2$. For inputs where predictions are inherently more uncertain, a heteroscedastic model learns $\sigma^2(x)$ alongside $\hat{y}(x)$:
$$L = \frac{(y - \hat{y}(x))^2}{2\sigma^2(x)} + \frac{1}{2}\log\sigma^2(x)$$

where $\sigma^2(x)$ = predicted variance (function of input, learned alongside $\hat{y}$), first term = MSE scaled by predicted confidence, $\frac{1}{2}\log\sigma^2(x)$ = entropy regularizer preventing infinite variance prediction.

The second term regularizes against predicting infinite variance. This naturally down-weights noisy or ambiguous training examples and produces calibrated uncertainty estimates. Used in depth estimation, bounding box quality prediction, and any task with input-dependent noise. Kendall & Gal (2017) showed this aleatoric uncertainty estimation significantly improves semantic segmentation in ambiguous regions.

---

## Links

- [[statistical-inference-mle]] — MSE is the MLE loss when output noise is Gaussian; noise model determines loss function
- [[distributions-gaussian]] — MSE minimization is equivalent to maximizing the Gaussian log-likelihood
- [[loss-huber]] — Huber loss blends MSE near zero with L1 for large errors; the remedy for MSE's outlier sensitivity
- [[loss-cross-entropy]] — the classification counterpart; using MSE for classification is a common and consequential mistake
- [[bayesian-inference]] — Gaussian likelihood + Gaussian prior gives L2-regularized ridge regression as the MAP estimate
- [[diffusion-models]] — diffusion training uses MSE to predict noise; the blurriness problem of MSE is exactly what diffusion models solve for generation
