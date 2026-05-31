---
title: Generative vs Discriminative Models
tags: [statistics, generative-models, discriminative-models, probability, machine-learning]
aliases: [generative discriminative, generative model, discriminative model, p(y|x), p(x,y)]
difficulty: 2
status: complete
related: [statistical-inference-mle, bayesian-inference, distributions-gaussian, variational-autoencoders, diffusion-models]
---

# Generative vs Discriminative Models

---

## Fundamental

**Discriminative model:** learns $p(y | x)$ directly — given input $x$, what is the probability of label $y$? Models the decision boundary between classes.

**Generative model:** learns $p(x, y) = p(x | y)\,p(y)$ — how is data generated? Can derive $p(y|x)$ via Bayes' theorem if needed:

$$p(y | x) = \frac{p(x | y)\,p(y)}{p(x)}$$

**Discriminative examples:** logistic regression, SVM, neural network classifiers, CRF.

**Generative examples:** Naive Bayes, Gaussian Discriminant Analysis, GMM, HMM, VAE, GAN, Normalizing Flow, Diffusion Model.

**Discriminative advantages:**
- Solve a simpler problem — no need to model the full input distribution
- Higher accuracy for classification given enough data (fewer wasted parameters on modeling input noise)
- Asymptotically optimal when $n \to \infty$

**Discriminative disadvantages:**
- Cannot generate new samples
- Cannot handle missing features naturally
- No explicit model of $p(x)$ → cannot detect OOD inputs or anomalies

**Generative advantages:**
- Can generate new samples ($x \sim p(x|y)$)
- Handle missing features: $p(x_1 | x_2, y)$ via marginalization
- Anomaly detection: flag low $p(x)$ inputs
- Semi-supervised learning: use unlabeled data to learn $p(x)$
- Data augmentation: generate synthetic training examples

**Generative disadvantages:**
- Harder to train (must model all of $p(x)$, including irrelevant variation)
- Often lower classification accuracy than discriminative models
- May make incorrect distributional assumptions

**Logistic regression as discriminative:** models $p(y=1 | x) = \sigma(w^\top x + b)$ directly. No assumption about $p(x)$.

**Gaussian Discriminant Analysis (GDA) as generative:** assumes $p(x | y=k) = \mathcal{N}(x; \mu_k, \Sigma)$. Estimates $\mu_k$, $\Sigma$, and $p(y)$ from training data. Uses Bayes' theorem to classify.

**The GDA-logistic connection:** if $p(x|y)$ is Gaussian and all classes share $\Sigma$, the Bayes-optimal classifier from GDA is logistic regression. GDA and logistic regression converge to the same boundary with infinite data — but GDA makes more assumptions and converges faster with small data.

---

## Intermediate

**When to use each:**

| Situation | Prefer |
|-----------|--------|
| Large labeled dataset | Discriminative |
| Small labeled dataset, strong domain knowledge | Generative |
| Need to generate new samples | Generative |
| Handle missing features at inference | Generative |
| Anomaly / OOD detection | Generative (or hybrid) |
| Semi-supervised learning | Generative |
| Speed and simplicity | Discriminative |
| Calibrated probabilities | Discriminative (with calibration) |

**Naive Bayes:** assumes features are conditionally independent given the label: $p(x|y) = \prod_i p(x_i | y)$. Almost never literally true, but the decision boundary can be correct even when the generative model is wrong. Works surprisingly well for text classification despite strong independence assumption — individual words are not independent, but their marginal contributions to classification often are.

**Modern deep generative models:**

- **VAE:** models $p(x) = \int p(x|z)p(z)dz$ with encoder $q(z|x)$ and decoder $p(x|z)$
- **GAN:** implicit model of $p(x)$ via generator. Does not give explicit $p(x)$ — cannot evaluate likelihood
- **Diffusion Models:** model $p(x)$ by learning to reverse a noise-adding process. Current state of the art for image generation
- **Normalizing Flows:** exact $p(x)$ via invertible transformations. Used in anomaly detection and density estimation

**Covariate shift sensitivity:** discriminative models that model $p(y|x)$ are more vulnerable to covariate shift ($P_S(x) \neq P_T(x)$) than generative models — a discriminative model trained on cats from one distribution may learn $x$-specific shortcuts rather than generalizable features. A generative model that explicitly models $p(x|y)$ can at least detect when $p_T(x)$ differs from $p_S(x)$ and flag inputs as OOD.

---

## Advanced

**The generative-discriminative tradeoff quantified (Ng & Jordan, 2002):** for Naive Bayes vs logistic regression, Naive Bayes achieves lower test error with $O(\log d)$ samples; logistic regression achieves lower test error with $O(d)$ samples. Generative models are more data-efficient with few samples; discriminative models are asymptotically better with many samples. The crossover occurs at roughly $d$ samples where $d$ is the feature dimension.

**The generative-discriminative spectrum — modern models blur the boundary:**

- **CLIP:** jointly trained vision-language model; discriminative (contrastive) but enables generation by pairing with a decoder
- **Diffusion classifiers:** use $p(x|y)$ from a diffusion model to classify — a generative model used discriminatively
- **Self-supervised pretraining (MAE, BERT):** learns $p(x)$ structure (generative pretext) but is used discriminatively after fine-tuning
- **Score-based models:** learn $\nabla_x \log p(x)$ (the score function) rather than $p(x)$ directly; enables generation without computing intractable normalizing constants

**Energy-based models (EBMs):** define $p(x) \propto \exp(-E_\theta(x))$ where $E_\theta$ is a neural energy function. The normalization constant $Z = \int \exp(-E_\theta(x))dx$ is intractable. Training uses contrastive divergence (CD): sample from the model via MCMC to approximate the partition function gradient. EBMs are the most general framework but the hardest to train. Connection: logistic regression is a linear EBM; the classification boundary is a level set of the energy.

**Semi-supervised generative models (Kingma et al., 2014):** VAEs can incorporate label information semi-supervisedly. The ELBO objective can be decomposed into labeled and unlabeled terms. For labeled data: $\log p(x, y) \geq \mathbb{E}[\log p(y|x)] + \text{ELBO}_{xy}$. For unlabeled data: marginalize over $y$. The model jointly learns the generative distribution $p(x|y)$ and the classifier $p(y|x)$. Even with few labeled examples, the generative structure provides regularization through the learned $p(x)$.

**Why discriminative models don't generalize under distribution shift despite being "simpler":** the "simpler problem" claim assumes the test distribution matches the training distribution. Under covariate shift, discriminative models may learn spurious correlations in $p(y|x)$ that are specific to $P_S(x)$. A generative model with the correct $p(x|y)$ can decouple feature relevance from feature prevalence. This is the theoretical basis for why generative pretraining followed by discriminative fine-tuning (the BERT paradigm) outperforms pure discriminative training in low-data regimes.

---

*See also: [[statistical-inference-mle]] · [[bayesian-inference]] · [[distributions-gaussian]] · [[variational-autoencoders]] · [[diffusion-models]]*
