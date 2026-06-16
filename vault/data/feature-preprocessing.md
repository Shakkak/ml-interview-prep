---
title: Feature Preprocessing and Engineering
tags: [preprocessing, feature-engineering, normalization, scaling, log-transform, missing-values, categorical]
aliases: [feature preprocessing, log transform, feature scaling, standardization, normalization, Box-Cox, target encoding]
difficulty: 1
status: complete
depends_on: [linear-algebra-fundamentals, distributions-overview]
related: [linear-algebra-fundamentals, distributions-overview, distributions-gaussian, statistical-inference-mle]
---

# Feature Preprocessing and Engineering

---

## Fundamental

Raw features are often on incompatible scales, skewed, or in representations the model cannot exploit directly. Preprocessing transforms features into a form that makes learning easier.

**Key rule:** tree-based models (Random Forest, XGBoost) are scale-invariant and skew-tolerant — they only care about the rank order of feature values. Linear models, neural networks, and SVMs are sensitive to both scale and distribution shape.

### Scaling Methods

**Standardization (Z-score):** $\tilde{x} = (x - \mu)/\sigma$. Output: mean 0, std 1. Does not bound the range — outliers survive in $z$-score space. Use when the feature is approximately [[distributions-gaussian|Gaussian]] or when you want to preserve outlier information.

**Min-max scaling:** $\tilde{x} = (x - x_{\min})/(x_{\max} - x_{\min}) \in [0, 1]$. Guaranteed bounded range. One extreme outlier compresses all other values into a tiny sub-range — fragile to outliers in the training set.

**Robust scaling:** $\tilde{x} = (x - \text{median})/\text{IQR}$. Uses the interquartile range instead of std — resistant to outliers. Use when the data has heavy-tailed distributions or known outliers to preserve.

### Log Transform for Skewed Features

Apply $\tilde{x} = \log(x+1)$ (for zeros) when a feature is strictly non-negative with a right-skewed distribution (long tail to the right). Examples: salary, house price, word frequency, API latency.

**Why it helps:** (1) compresses the right tail — $\log(10^6) = 6$ vs $\log(1) = 0$, range of 6 instead of $10^6$; (2) brings distribution toward Gaussian, satisfying linear model assumptions; (3) makes multiplicative relationships additive — $y = ax^b \Rightarrow \log y = \log a + b\log x$.

**When not to use:** tree-based models (monotonic transforms don't change split decisions), symmetric/Gaussian features, features where zero has specific meaning.

---

## Intermediate

### Box-Cox Transform

Generalization of the log transform to a family parametrized by $\lambda$:

$$\tilde{x} = \begin{cases} (x^\lambda - 1)/\lambda & \lambda \neq 0 \\ \log x & \lambda = 0 \end{cases}$$

where $x > 0$ = the original feature value (Box-Cox requires strictly positive input), $\lambda$ = the power parameter (found by MLE to maximize the likelihood that the transformed values are Gaussian; $\lambda = 0$ recovers log, $\lambda = 1$ is identity, $\lambda = 0.5$ is square-root), and $\tilde{x}$ = the transformed value. Find $\lambda$ that makes $\tilde{x}$ most Gaussian by [[statistical-inference-mle|MLE]] on training data. The log transform is the $\lambda = 0$ special case. Box-Cox requires $x > 0$; Yeo-Johnson extends it to negative values.

### Why Scaling Helps Gradient Descent

For a linear model with loss $\frac{1}{2}\|Xw - y\|^2$, the Hessian is $H = X^\top X$. Gradient descent convergence requires $O(\kappa(H))$ steps where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the [[eigenvalues-pca|condition number]]. If feature 1 has variance $10^6$ and feature 2 has variance 1, then $\kappa(H) \approx 10^6$ and convergence is extremely slow (elongated loss contours). After standardization, eigenvalues are equal, $\kappa \approx 1$, and gradient descent converges in far fewer steps.

### Categorical Features

| Strategy | When | Risk |
|---|---|---|
| One-hot encoding | Low cardinality ($\leq 20$) | Sparsity for high cardinality |
| Ordinal encoding | Ordered categories (small < medium < large) | Implies numerical distance |
| Target encoding | High cardinality (1000 cities) | Data leakage if naively applied |
| Learned embeddings | High cardinality in DL models | Requires enough data per category |

**Target encoding safety:** replace category $c$ with $\mathbb{E}[y \mid x=c]$ estimated using **[[cross-validation]]** — never compute on the same fold being predicted. Optionally add regularization: $\hat{\mu}_c = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}$ where $m$ is a smoothing factor (blends toward the global mean for rare categories).

### Missing Values

Missing-data mechanism (Rubin, 1976) determines the right strategy:

| Mechanism | Definition | Strategy |
|---|---|---|
| MCAR | Missing completely at random | Mean/median imputation |
| MAR | Missingness depends on other observed features | Model-based imputation (MissForest, mice) |
| MNAR | Missingness depends on the unobserved value itself | Indicator + impute; domain-specific handling |

**Always add a binary "was-missing" indicator** when missingness is MNAR or when you suspect it is informative — the fact that a value is missing may be the most predictive signal. Never impute and then pretend the value was originally observed.

---

## Advanced

### Leakage and Train/Test Consistency

All preprocessing statistics (mean, std, min, max, target encoding values, PCA components, imputation values) must be computed **only on the training set** and applied identically to the validation and test sets. Fitting on the combined dataset leaks test information into training — a common source of optimistic evaluation.

Correct workflow using scikit-learn pipelines:
```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X_train, y_train)        # scaler sees only training stats
pipe.score(X_test, y_test)        # test set transformed using train stats
```

A `ColumnTransformer` inside the pipeline handles mixed feature types (numerical + categorical) without leakage.

### Feature Interactions

Linear models cannot learn $x_1 \times x_2$ effects automatically. Adding explicit interaction features:

- **Polynomial:** $x_1^2, x_1 x_2, x_2^2$ — captures second-order effects. For $d$ features, $O(d^2)$ second-order terms. Regularize to prevent overfitting.
- **Ratio:** $x_1/x_2$ — often more informative than raw values (price per square meter, clicks per impression).
- **Difference:** $x_1 - x_2$ — e.g., days since last purchase, relative latency improvement.
- **Binning:** convert continuous to ordinal — useful when the relationship is piecewise constant (age groups, income brackets). Loses within-bin variation; gains robustness to outliers.

**Gradient boosted trees discover interactions automatically** — the key reason they dominate tabular benchmarks without explicit feature engineering. Feature engineering is primarily valuable for linear models or when domain knowledge constrains the search space.

### Connection to MLE and Log-Normal Regression

If you assume the target $y \mid x$ follows a log-normal distribution $\log y \mid x \sim \mathcal{N}(\mu(x), \sigma^2)$, then MLE under this model is equivalent to OLS on $\log y$. This provides the theoretical justification for log-transforming the target variable in regression when you believe the noise is **multiplicative** ($y = f(x) \cdot \epsilon$, $\epsilon > 0$) rather than additive ($y = f(x) + \epsilon$).

Salary prediction, house price prediction, and reaction time modeling all fit the multiplicative noise assumption — the same absolute error matters less when the underlying value is larger.

---

## Links

- [[linear-algebra-fundamentals]] — standardization projects features to a common scale; PCA whitening uses eigendecomposition to decorrelate and normalize simultaneously
- [[distributions-overview]] — the choice of transformation (log, Box-Cox, Yeo-Johnson) depends on the feature's distribution; log is for right-skewed, power transforms for arbitrary skew
- [[distributions-gaussian]] — standardization assumes approximately Gaussian features; robust scaling (median/IQR) is better for heavy-tailed distributions
- [[statistical-inference-mle]] — target encoding computes the conditional mean via MLE; Bayesian smoothing regularizes sparse categories toward the global mean
- [[eigenvalues-pca]] — PCA whitening is a preprocessing step that standardizes features while decorrelating them using the covariance matrix's eigendecomposition
- [[cross-validation]] — preprocessing fitted on training folds must be re-applied (not re-fitted) on test folds to prevent data leakage
