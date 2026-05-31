---
title: Causal Inference
tags: [statistics, causal-inference, confounding]
aliases: [causal inference, causality, confounding, RCT, do-calculus]
difficulty: 3
status: complete
related: [statistical-inference-mle, distributions-overview, bayesian-inference]
---

# Causal Inference

---

## Fundamental

**Correlation vs Causation:**

Correlation: $X$ and $Y$ move together in the data. Causation: changing $X$ (while holding everything else fixed) changes $Y$. The fundamental challenge: observed data reflects correlation, not causation. You cannot tell from $P(Y|X)$ alone whether $X$ causes $Y$.

**The three structures of correlation:**

Chain (Mediation): $X \to Z \to Y$. $X$ causes $Z$ causes $Y$. $X$ and $Y$ are correlated; conditioning on $Z$ blocks the path.

Fork (Confounding): $X \leftarrow Z \to Y$. $Z$ causes both $X$ and $Y$. $X$ and $Y$ are correlated but neither causes the other. Conditioning on $Z$ removes the correlation.

Collider: $X \to Z \leftarrow Y$. $X$ and $Y$ are independent. But conditioning on $Z$ *creates* a spurious correlation between $X$ and $Y$. Counter-intuitive and a common source of selection bias.

**Simpson's Paradox:** a relationship in the overall population reverses within every subgroup. Resolution: condition on the confounding variable. The within-group rates are the causal estimates.

Numerical example (drug trial where severe patients disproportionately receive the drug):

| Group | Drug | No Drug |
|-------|------|---------|
| Mild | 93% | 87% |
| Severe | 73% | 69% |
| **Overall** | **81%** | **83%** |

Drug looks *worse* overall but is *better* within every severity subgroup. Severity is the confounder — severe patients are more likely to receive the drug, dragging down its average. Conditioning on severity: drug is better.

**Randomized Controlled Trials (RCT):** randomly assign treatment. Randomization ensures treatment is independent of all confounders — both observed and unobserved.

$$P(Y | \text{do}(X=x)) = P(Y|X=x) \quad \text{(in an RCT)}$$

This is the gold standard. But RCTs are expensive, sometimes unethical, or impossible (can't randomize gender, nationality, etc.).

---

## Intermediate

**The do-operator:** $P(Y|\text{do}(X=x))$ means "intervene to set $X=x$" — different from conditioning on observing $X=x$. Conditioning is passive (select a subpopulation); intervening is active (override the mechanism determining $X$).

**Backdoor adjustment:** if you can identify and measure all confounders $Z$:

$$P(Y|\text{do}(X=x)) = \sum_z P(Y|X=x, Z=z)P(Z=z)$$

Numerical example: using population severity split (60% mild, 40% severe) from the Simpson's paradox example:

$$P(\text{recovery}|\text{do}(\text{drug})) = 0.6 \times 0.93 + 0.4 \times 0.73 = 0.85$$
$$P(\text{recovery}|\text{do}(\text{no drug})) = 0.6 \times 0.87 + 0.4 \times 0.69 = 0.798$$

Drug is better: 85% vs 79.8%. The backdoor adjustment recovers the true causal effect.

**Instrumental Variables (IV):** when not all confounders are measurable, use a variable $Z$ (instrument) that: (1) causes $X$ (relevance), (2) affects $Y$ only through $X$ (exclusion restriction), (3) is independent of confounders. Example: distance to college as instrument for education (affects education attainment, but doesn't directly affect wages except through education).

**Propensity Score Matching:** estimate $P(X=1|Z)$ (propensity score) for each unit. Match treated and control units with similar propensity scores. Compare outcomes within matched pairs. Limitation: only controls for *observed* confounders.

**Difference-in-Differences:** compare the change in outcomes over time between treated and control groups:

$$\hat{\tau} = (Y_{T,\text{after}} - Y_{T,\text{before}}) - (Y_{C,\text{after}} - Y_{C,\text{before}})$$

Removes time-invariant confounders and common time trends. Assumption: parallel trends — in the absence of treatment, treated and control groups would have evolved similarly.

**Collider bias example:** among hospitalized patients, there is a negative correlation between disease severity and injury severity. Does severe disease protect against injury? No — hospitalization is a collider. Both severe disease and severe injury cause hospitalization. Conditioning on being hospitalized creates a spurious negative correlation.

**ML selection bias example:** if training data is collected by filtering on a collider (e.g., only users who stayed engaged with the product), features and labels may be spuriously correlated in ways that don't hold on the full population.

---

## Advanced

**Do-calculus (Pearl, 1995):** a complete formal system for deriving causal effects from observational data using a causal graph (DAG). Three rules transform expressions with do-operators into observational distributions:

- Rule 1 (Insertion/deletion of observations): $P(y|\text{do}(x), z, w) = P(y|\text{do}(x), w)$ when $Z$ is independent of $Y$ given $X,W$ in the mutilated graph $G_{\overline{X}}$.
- Rule 2 (Action/observation exchange): replaces do-operators with conditioning when the variable has no confounders.
- Rule 3 (Deletion of actions): removes do-operators when the variable is not an ancestor of any confounder.

Do-calculus is **complete**: if a causal effect is identifiable from the DAG and data, do-calculus will find it. Non-identifiable effects (e.g., when unmeasured confounders exist without instruments) cannot be estimated, and do-calculus proves this impossibility.

**Invariant Risk Minimization (Arjovsky et al., 2019):** learn a model that performs equally well across multiple environments (different hospitals, time periods, countries). Features that vary between environments are likely spurious correlations; features that work everywhere are likely causal. The IRM objective:

$$\min_\Phi \sum_e \mathcal{L}^e(\Phi) \quad \text{s.t.} \quad \Phi \in \arg\min_{\bar{\Phi}} \mathcal{L}^e(\bar{\Phi}) \text{ for all } e$$

The inner constraint forces the same classifier to be optimal across all environments, identifying features invariant to environment shifts.

**Counterfactual reasoning and the Rubin causal model:** the fundamental problem of causal inference is that for each unit $i$, only one potential outcome $Y_i^{(t)}$ is observed for the treatment actually received. The counterfactual $Y_i^{(1-t)}$ is never observed. The Average Treatment Effect (ATE) is:

$$\text{ATE} = \mathbb{E}[Y^{(1)} - Y^{(0)}]$$

This requires the **Stable Unit Treatment Value Assumption (SUTVA)**: each unit's outcome is unaffected by other units' treatments (no interference). In social networks or epidemics, SUTVA is violated — individuals affect each other.

**Double machine learning (Chernozhukov et al., 2018):** estimates treatment effects while controlling for high-dimensional confounders. Two stages: (1) use an ML model to predict $X$ from $Z$ (confounders) and obtain residuals $\tilde{X}$; (2) use another ML model to predict $Y$ from $Z$ and obtain residuals $\tilde{Y}$; (3) regress $\tilde{Y}$ on $\tilde{X}$. The Frisch-Waugh-Lovell theorem guarantees that this partialing-out procedure recovers the causal effect even with non-parametric ML estimators, under the Neyman orthogonality condition.

**Causal discovery:** learning the DAG structure from observational data. PC algorithm: start with a complete graph; remove edges by testing conditional independence; orient remaining edges. FCI algorithm: handles latent confounders. Key limitation: Markov equivalence classes — multiple DAGs produce the same conditional independence structure and are indistinguishable from observational data alone. Interventional data breaks equivalence classes.

---

*See also: [[statistical-inference-mle]] · [[distributions-overview]] · [[bayesian-inference]]*
