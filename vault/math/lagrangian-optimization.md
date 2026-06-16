---
title: Lagrangian Optimization and KKT Conditions
tags: [lagrangian, kkt-conditions, constrained-optimization, svm, duality, convex-optimization]
aliases: [Lagrange multipliers, KKT conditions, Karush-Kuhn-Tucker, constrained optimization, Lagrangian, primal-dual, SVM dual]
difficulty: 3
status: complete
depends_on: [matrix-calculus, math-convexity-jensen]
related: [math-convexity-jensen, kernel-methods, matrix-calculus, fisher-information, regularization-weight-decay]
---

# Lagrangian Optimization and KKT Conditions

---

## Fundamental

Most ML objectives are unconstrained (minimize $\mathcal{L}(\theta)$ over all $\theta \in \mathbb{R}^d$). Constrained optimization adds structure that often makes problems harder to solve directly — but the Lagrangian framework converts them into unconstrained problems.

**General constrained form:**
$$\min_x f(x) \quad \text{s.t.} \quad h_j(x) = 0 \text{ for } j=1,\ldots,p, \quad g_i(x) \leq 0 \text{ for } i=1,\ldots,m$$

where $x \in \mathbb{R}^n$ is the decision variable (e.g., model parameters), $f(x)$ is the objective to minimize (e.g., loss), $h_j(x) = 0$ are equality constraints (e.g., $\|w\|^2 = 1$), and $g_i(x) \leq 0$ are inequality constraints (e.g., $-w \leq 0$ meaning $w \geq 0$).

### Equality Constraints: Lagrange Multipliers

For $\min_x f(x)$ subject to $h(x) = 0$ (a single equality constraint):

**Geometric intuition:** at the optimal $x^*$, the gradient $\nabla f$ must be parallel to $\nabla h$. If not, there would exist a direction along the constraint surface (orthogonal to $\nabla h$) with negative $\nabla f$ component — we could decrease $f$ while staying feasible. So at optimality:
$$\nabla f(x^*) = -\lambda \nabla h(x^*)$$

**Lagrangian:** $\mathcal{L}(x, \lambda) = f(x) + \lambda h(x)$.

The optimum satisfies:
$$\nabla_x \mathcal{L} = 0 \quad\text{(stationarity)}, \qquad \nabla_\lambda \mathcal{L} = h(x) = 0 \quad\text{(feasibility)}$$

**Shadow price interpretation:** $\lambda$ measures how much the optimal value $f^*$ changes per unit relaxation of the constraint: $\frac{df^*}{dc}\big|_{h(x)=c, c=0} = -\lambda$. In economics, $\lambda$ is the marginal cost of the constraint.

**[[eigenvalues-pca|PCA]] example:** maximize $w^\top \Sigma w$ subject to $\|w\|^2 = 1$. Lagrangian: $w^\top \Sigma w - \lambda(w^\top w - 1)$. Stationarity: $\Sigma w = \lambda w$ — the optimal direction is an eigenvector of $\Sigma$ with eigenvalue $\lambda$ equal to the maximum variance achieved.

---

## Intermediate

### KKT Conditions for Inequality Constraints

For the general problem, the **KKT conditions** are necessary for optimality (and sufficient for [[math-convexity-jensen|convex]] problems):

**1. Stationarity:**
$$\nabla f(x^*) + \sum_i \mu_i \nabla g_i(x^*) + \sum_j \lambda_j \nabla h_j(x^*) = 0$$

where $\mu_i \geq 0$ = Lagrange multipliers for inequality constraints $g_i$ (also called KKT multipliers), and $\lambda_j$ = Lagrange multipliers for equality constraints $h_j$ (can be any sign). Stationarity says the gradient of the objective is canceled by the gradients of the active constraints.

**2. Primal feasibility:**
$$g_i(x^*) \leq 0, \quad h_j(x^*) = 0$$

All constraints are satisfied at the optimal point $x^*$.

**3. Dual feasibility:**
$$\mu_i \geq 0$$

Inequality multipliers are non-negative (ensures the constraint can only "push" against the feasible region, not "pull").

**4. Complementary slackness:**
$$\mu_i g_i(x^*) = 0 \text{ for all } i$$

**Complementary slackness** is the key condition: either the constraint is **active** ($g_i(x^*) = 0$, tight, $\mu_i \geq 0$) or **inactive** ($g_i(x^*) < 0$, slack, $\mu_i = 0$ — the constraint is not binding and has no effect on the solution).

### Strong Duality

The **Lagrangian dual function** $d(\mu, \lambda) = \inf_x \mathcal{L}(x, \mu, \lambda)$ is always a lower bound on $f^*$ (weak duality). The **duality gap** $f^* - d^*$ is the difference between primal and dual optima.

**Strong duality** (zero gap): for convex $f$, $g_i$ and linear $h_j$, strong duality holds under **Slater's condition**: there exists a strictly feasible point $\tilde{x}$ with $g_i(\tilde{x}) < 0$ for all $i$.

Strong duality lets you solve the dual problem (often easier) instead of the primal. This is the foundation of the SVM derivation.

### SVM Dual Derivation

**Hard-margin SVM primal:**
$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 \text{ for all } i$$

Equivalently $g_i(w,b) = 1 - y_i(w^\top x_i + b) \leq 0$. Lagrangian with $\alpha_i \geq 0$:
$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i [y_i(w^\top x_i + b) - 1]$$

where $w \in \mathbb{R}^d$ = weight vector (decision boundary normal), $b$ = bias, $\alpha_i \geq 0$ = Lagrange multipliers for the margin constraints, $y_i \in \{-1, +1\}$ = class labels, $x_i$ = training points.

Stationarity ($\nabla_w \mathcal{L} = 0$, $\nabla_b \mathcal{L} = 0$):
$$w = \sum_i \alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0$$

where $w = \sum_i \alpha_i y_i x_i$ says the optimal weight vector is a linear combination of the training points (weighted by their dual variables $\alpha_i$).

Substituting $w$ back into $\mathcal{L}$ and simplifying:
$$d(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j \,x_i^\top x_j$$

where $d(\alpha)$ is the dual objective (depends only on the dual variables $\alpha_i$ and the inner products $x_i^\top x_j$ between training examples, not on $w$ directly — enabling the [[kernel-methods|kernel trick]]).

**Dual problem:**
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j \,x_i^\top x_j \quad \text{s.t.} \quad \alpha_i \geq 0,\;\sum_i \alpha_i y_i = 0$$

Slater's condition holds for linearly separable data, so strong duality gives $d^* = f^*$.

**Three key insights from the dual:**
1. **[[kernel-methods|Kernel trick]]:** the objective depends only on inner products $x_i^\top x_j$ — replace with $k(x_i, x_j)$ to get nonlinear SVMs without changing the optimization structure.
2. **Support vectors via complementary slackness:** $\alpha_i g_i = 0$. Points inside the margin have $g_i < 0 \Rightarrow \alpha_i = 0$. Only points on the margin ($y_i(w^\top x_i + b) = 1$, $\alpha_i > 0$) determine $w = \sum \alpha_i y_i x_i$. These are the **support vectors**.
3. **Prediction:** $\hat{y} = \text{sign}\!\left(\sum_i \alpha_i y_i k(x_i, x) + b\right)$ — only the (sparse) support vectors contribute.

---

## Advanced

### Soft-Margin SVM and the Role of $C$

For non-separable data, introduce slack variables $\xi_i \geq 0$ (amount by which constraint $i$ is violated):
$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i,\;\xi_i \geq 0$$

Dual: same as hard-margin but with **box constraint** $0 \leq \alpha_i \leq C$. The upper bound $C$ prevents any single point from dominating.

$C$ is the regularization parameter:
- Large $C$: heavy penalty on slack → small margin, classify all points correctly if possible → overfitting risk
- Small $C$: allow more slack → wider margin, more misclassifications → underfitting risk

**Connection to hinge loss:** the soft-margin SVM is equivalent to minimizing $\frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1 - y_i(w^\top x_i + b))$. The $\max(0, \cdot)$ term is the **hinge loss**. The primal-dual equivalence shows that L2 regularization + hinge loss = maximum-margin classification with slack.

### ADMM and Distributed Optimization

When the primal problem is too large to solve directly, the **Alternating Direction Method of Multipliers (ADMM)** decomposes it:

$$\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c$$

where $x$ and $z$ = split decision variables (e.g., local model parameters on different nodes), $f$ and $g$ = separable objective functions (each depending on only one variable), $A$, $B$ = constraint matrices, $c$ = constraint target vector, and the linear constraint $Ax + Bz = c$ couples the two variables.

Augmented Lagrangian: $\mathcal{L}_\rho(x,z,y) = f(x) + g(z) + y^\top(Ax+Bz-c) + \frac{\rho}{2}\|Ax+Bz-c\|^2$.

ADMM alternates:
1. $x \leftarrow \arg\min_x \mathcal{L}_\rho(x, z, y)$ (minimize over $x$, others fixed)
2. $z \leftarrow \arg\min_z \mathcal{L}_\rho(x, z, y)$ (minimize over $z$, others fixed)
3. $y \leftarrow y + \rho(Ax + Bz - c)$ (dual ascent step)

ADMM is used for distributed machine learning (split the dataset across nodes, each node minimizes locally, nodes share dual variables), LASSO with proximal operators, and consensus optimization in federated learning.

### Duality Gap and KKT as Optimality Certificate

For convex problems, the KKT conditions are both necessary **and sufficient** for global optimality. A point $x^*$ is globally optimal iff there exist $\mu^*$, $\lambda^*$ satisfying all four KKT conditions.

This gives a practical **optimality certificate**: after running any optimization algorithm (e.g., SGD), check whether KKT violations are below tolerance. For the SVM dual, an $\alpha^*$ achieving dual $= $ primal (zero gap) is an exact certificate that the global optimum was found.

**Why neural network training is harder:** non-convex objectives violate the sufficient direction of KKT. A point satisfying the stationarity condition $\nabla_\theta L = 0$ may be a local minimum, saddle point, or local maximum. Recent theory (Kawaguchi 2016) shows that local minima of deep linear networks equal the global minimum, and wide networks have few "bad" local minima in practice — but no KKT-style optimality certificate exists.

---

## Links

- [[matrix-calculus]] — KKT conditions require computing gradients of the Lagrangian; the stationarity condition is $\nabla_x \mathcal{L} = 0$
- [[math-convexity-jensen]] — strong duality (primal = dual) holds when the problem is convex; Jensen's inequality is used in the proof of convex duality
- [[kernel-methods]] — the SVM dual is derived by applying Lagrange multipliers to the margin-maximization problem; kernelization happens in the dual objective
- [[regularization-weight-decay]] — L2 regularization can be derived as a Lagrangian relaxation of a norm constraint; the regularization strength $\lambda$ is the Lagrange multiplier
- [[eigenvalues-pca]] — PCA's variance-maximization under unit-norm constraint is solved via Lagrange multipliers; the solution is the eigenvector equation
- [[fisher-information]] — the natural gradient is derived by minimizing loss subject to a KL constraint; the Lagrange multiplier leads to $F^{-1}$ preconditioning
