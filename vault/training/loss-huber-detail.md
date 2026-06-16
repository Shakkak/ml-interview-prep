---
title: Huber Loss, Quantile Regression, and Robust Losses
tags: [huber-loss, quantile-regression, pinball-loss, robust-regression, outlier-robust]
aliases: [Huber loss, pinball loss, quantile loss, MAE, robust regression, expectile]
difficulty: 1
status: complete
related: [loss-mse, loss-huber, loss-cross-entropy, statistical-inference-mle, gradient-boosting]
depends_on: [loss-mse, loss-huber, statistical-inference-mle]
---

# Huber Loss, Quantile Regression, and Robust Losses

---

## Fundamental

### Huber Loss

The **Huber loss** (see also [[loss-huber]]) combines MSE for small errors and MAE for large errors:

$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta(|r| - \frac{1}{2}\delta) & |r| > \delta \end{cases}$$

where $r = y - \hat{y}$ = residual (true minus predicted), $\delta > 0$ = threshold separating quadratic from linear regime, $\frac{1}{2}r^2$ = MSE for small errors, $\delta(|r| - \frac{1}{2}\delta)$ = scaled MAE for large errors.

**Choosing $\delta$:**
- $\delta \to 0$: approaches MAE (fully robust, non-smooth at zero)
- $\delta \to \infty$: approaches MSE (fully sensitive to outliers)
- Typical: $\delta = 1$ (residuals $< 1$ treated as quadratic, $> 1$ as linear)

The optimal $\delta$ depends on the scale of typical vs outlier residuals. Cross-validate $\delta$ or use domain knowledge about expected noise.

**Gradient:** $\nabla \mathcal{L}_\delta = r$ for $|r| \leq \delta$ (MSE gradient), $\nabla \mathcal{L}_\delta = \delta \cdot \text{sign}(r)$ for $|r| > \delta$ (clipped gradient). Gradient clipping = Huber loss optimization.

---

## Intermediate

### Quantile Regression

Instead of predicting the conditional mean $\mathbb{E}[y|x]$, **quantile regression** predicts the conditional quantile $Q_\tau[y|x]$ — the $\tau$-th percentile of $y$ given $x$.

**Pinball loss (quantile loss):**

$$\mathcal{L}_\tau(r) = \begin{cases} \tau \cdot r & r \geq 0 \\ (\tau - 1) \cdot r & r < 0 \end{cases} = r(\tau - \mathbf{1}[r < 0])$$

where $r = y - \hat{y}$ = residual, $\tau \in (0,1)$ = target quantile, $\mathbf{1}[r < 0]$ = 1 if underprediction, penalizes overprediction by $(1-\tau)$ and underprediction by $\tau$.

- Asymmetric: penalizes overprediction by $1-\tau$ and underprediction by $\tau$
- Minimizing expected pinball loss gives the $\tau$-th conditional quantile
- $\tau = 0.5$: median regression (equivalent to MAE up to scaling)
- $\tau = 0.9$: 90th percentile prediction — useful for upper confidence bounds

**Uncertainty estimation:** train models for $\tau \in \{0.1, 0.5, 0.9\}$ simultaneously → prediction intervals without Gaussian assumptions.

### Distribution-Free Prediction Intervals

**Conformal prediction with quantile regression:** train quantile models $\hat{q}_{0.05}(x)$ and $\hat{q}_{0.95}(x)$, calibrate on held-out data to ensure exactly $90\%$ empirical coverage. This provides distribution-free prediction intervals with guaranteed marginal coverage.

### Expectile Regression

**Expectiles** are to $\ell_2$ loss what quantiles are to $\ell_1$ loss. The $\tau$-expectile minimizes:

$$\mathcal{L}_\tau(r) = |\tau - \mathbf{1}[r < 0]| \cdot r^2$$

where $\tau \in (0,1)$ = target expectile, $|\tau - \mathbf{1}[r < 0]|$ = asymmetric weight ($\tau$ for positive residuals, $1-\tau$ for negative), $r^2$ = squared residual.

Asymmetric MSE — penalizes negative residuals by $(1-\tau)$ and positive residuals by $\tau$.

**Used in:** off-policy RL (IQN — Implicit Quantile Networks), option pricing, risk measures in finance (EVaR — expectile value-at-risk).

---

## Advanced

### Gradient Boosting with Quantile Loss

Gradient boosted trees (see [[gradient-boosting]]) with pinball loss trains directly to the $\tau$-th quantile:
- Pseudo-residuals: $r_i = \tau - \mathbf{1}[y_i < \hat{y}_i]$ — either $\tau$ or $\tau-1$
- Each tree splits to reduce pinball loss
- Natural prediction intervals: train three trees ($\tau = 0.1, 0.5, 0.9$) simultaneously

LightGBM and XGBoost both support quantile objectives natively.

### When to Use Each Loss

| Scenario | Recommended loss |
|----------|-----------------|
| Standard regression, Gaussian errors | MSE |
| Outliers present | Huber ($\delta$ tuned) or MAE |
| Asymmetric costs (false pos $\neq$ false neg) | Pinball / quantile |
| Prediction intervals needed | Quantile regression pair |
| Heavy-tailed errors | MAE or log-cosh |
| Reinforcement learning (value distribution) | Quantile / expectile |

## Links

- [[loss-mse]] — MSE = $\frac{1}{2}(y-\hat{y})^2$ is quadratic everywhere; Huber loss is quadratic for small errors and linear for large ones, reducing outlier influence
- [[loss-huber]] — this file extends the basic Huber loss file with quantile and expectile regression; quantile regression estimates $P(Y \leq \hat{y}|x) = q$ at a given quantile $q$
- [[statistical-inference-mle]] — quantile regression corresponds to MLE under an asymmetric Laplace distribution; the pinball loss is the negative log-likelihood of this distribution
- [[gradient-boosting]] — XGBoost and LightGBM support custom loss functions including Huber and quantile; gradient boosting can minimize any differentiable loss by fitting residuals
