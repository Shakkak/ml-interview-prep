---
title: Data Leakage
tags: [data-leakage, target-leakage, temporal-leakage, train-test-contamination, overfitting]
aliases: [data leakage, target leakage, temporal leakage, train-test contamination, feature leakage]
difficulty: 1
status: complete
related: [cross-validation, feature-preprocessing, bias-variance-double-descent, evaluation-metrics-guide, overfitting]
---

# Data Leakage

---

## Fundamental

### What Is Data Leakage?

**Data leakage** occurs when information from outside the training set reaches the model in a way that inflates measured performance but degrades real-world performance. The model learns to exploit shortcuts that won't be available at deployment.

**Two main types:**

1. **Target leakage:** a feature used at training time is derived from (or reveals) the target variable, but would not be available at prediction time
2. **Train-test contamination:** test data influences preprocessing, feature selection, or hyperparameter tuning — the test set is no longer truly held-out

Both lead to overoptimistic evaluation metrics.

### Target Leakage Examples

| Feature | Target | Leakage mechanism |
|---------|--------|------------------|
| Antibiotic prescription | Has infection | Antibiotic is prescribed *because* of infection |
| Credit limit increase | Default | Limit is reduced *because* default risk increased |
| Discharge date | Hospital stay length | Discharge date determines stay length |
| Future sales data | Current month sales | Data from the future included in training features |

**Rule:** a feature has target leakage if it is causally downstream of the target or temporally after the prediction point.

---

## Intermediate

### Train-Test Contamination

**Scaling with test data:** fitting a StandardScaler or MinMaxScaler on the combined train+test set before splitting. The scaler uses test statistics (mean, std) — information from the test set contaminates training features.

**Correct approach:** fit the scaler on training data only; transform both train and test using those statistics.

**Feature selection leakage:** selecting features by correlation with the target on the full dataset (before splitting) — the selected features are "lucky" for the test set specifically, inflating performance.

**Imputation leakage:** filling missing values using the mean of the full dataset. The test mean slightly influences training features.

**Cross-validation leakage:** applying SMOTE or other data augmentation before splitting CV folds. The synthetic samples in the training fold overlap in feature space with the validation fold examples they were generated from.

### Temporal Leakage

Time-series data has an additional dimension: **temporal order**. Using future observations to predict the past.

**Examples:**
- Training a stock price predictor with normalized prices (normalization uses future prices)
- Grouping temporal events before splitting (events from the same "episode" end up in both train and test)
- Using lag features calculated over the full time series

**Temporal CV (time series split):** always validate on future data relative to training. Never shuffle time-series data before splitting.

---

## Advanced

### Detecting Leakage

1. **Suspiciously high performance:** AUC > 0.99 on a hard problem, accuracy > 95% on noisy data — suspect leakage
2. **Feature importance analysis:** if one feature dominates all others on a complex task, verify it isn't leaking target information
3. **Temporal consistency check:** verify features are available at the prediction point in the real workflow
4. **Hold-out test set check:** if performance drops dramatically from validation to a true held-out test, leakage likely inflated the validation metrics

### Leakage in LLM Evaluation

Data leakage is increasingly concerning for LLM benchmarks:
- **Training set contamination:** test examples from established benchmarks (MMLU, GSM8K) appear in web-scale pretraining data
- **N-gram overlap detection:** checking for n-gram matches between training data and benchmark examples
- **Canonical contamination:** prompting models to "continue" a benchmark example to detect memorization

**Mitigation:** use time-based splits (release date of benchmark vs training cutoff), use private evaluation sets, or use LLM-generated benchmarks with controlled creation date.

*See also: [[cross-validation]] · [[feature-preprocessing]] · [[evaluation-metrics-guide]] · [[bias-variance-double-descent]]*
