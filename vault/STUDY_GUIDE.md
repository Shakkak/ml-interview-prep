# ML Interview Study Guide

> **How to use:** Pick your track(s), start at Tier 1, work down. Score = importance for *general* ML interviews (1–10). Track tags show which scenarios value each topic most.
>
> **Tracks:** `GEN` General ML · `DL` Deep Learning Eng · `LLM` NLP/LLM · `CV` Computer Vision · `GEN-AI` Generative AI · `THEORY` Research/Theory

---

## Track Cheat Sheet

| Track | Who | Must-reads first |
|-------|-----|-----------------|
| **GEN** — General ML Scientist | Breadth-first, classical + DL | Linear/logistic regression, bias-variance, cross-validation, backprop, attention |
| **DL** — Deep Learning Engineer | Systems + architecture depth | Backprop, normalization, optimizers, attention, residual blocks, mixed precision |
| **LLM** — NLP / LLM Engineer | Transformers + training at scale | Attention, KV cache, LoRA, RLHF, tokenization, positional encoding, speculative decoding |
| **CV** — Computer Vision | Image models + detection | CNNs, ViT, FPN, IoU/NMS, U-Net, data augmentation, contrastive learning |
| **GEN-AI** — Generative AI | Generative models depth | Diffusion, VAE, GAN, flow matching, CLIP, autoregressive models, RLHF |
| **THEORY** — Research Scientist | Statistical learning theory | PAC learning, Rademacher, generalization bounds, Fisher info, information geometry |

---

## Tier 1 — Universal (Score 9–10)

**Read these before any ML interview. No exceptions.**

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 10 | [[backpropagation]] | ALL | Chain rule, gradient flow — the single most-tested topic |
| 10 | [[attention-mechanism]] | ALL | Self-attention, scaled dot-product, multi-head — foundational for all modern ML |
| 9 | [[normalization-layers]] | ALL | BatchNorm, LayerNorm, GroupNorm — asked in virtually every DL interview |
| 9 | [[optimizer-adam]] | ALL | Adam, moment estimates, bias correction — everyone needs this |
| 9 | [[optimizer-sgd-momentum]] | ALL | SGD + momentum as baseline; compare/contrast with Adam |
| 9 | [[regularization-dropout]] | ALL | Dropout as regularizer + test-time behavior |
| 9 | [[loss-cross-entropy]] | ALL | CE loss, log-prob, connection to MLE and KL divergence |
| 9 | [[arch-residual-block]] | DL CV GEN-AI | Skip connections, vanishing gradient fix — cited in almost every architecture question |
| 9 | [[arch-positional-encoding]] | LLM DL | Sinusoidal, learned, relative — before any transformer discussion |
| 9 | [[bias-variance-double-descent]] | ALL | Bias-variance tradeoff + modern double descent — classic interview anchor |
| 9 | [[cross-validation]] | GEN | CV, k-fold, stratified — ML fundamentals staple |
| 9 | [[linear-regression]] | GEN | OLS, normal equations, assumptions — often the warm-up question |
| 9 | [[logistic-regression]] | GEN | Binary/multinomial, gradient, probabilistic interpretation |
| 9 | [[gradient-boosting]] | GEN | XGBoost/LightGBM — top classical ML topic |
| 9 | [[matrix-calculus]] | ALL | Jacobians, chain rule in matrix form — backprop prereq |
| 9 | [[linear-algebra-fundamentals]] | ALL | Eigendecomposition, rank, projections — universal |
| 9 | [[eigenvalues-pca]] | ALL | PCA derivation, EVD, covariance — always tested |
| 9 | [[distributions-gaussian]] | ALL | Gaussian, MGF, affine transforms — building block |
| 9 | [[statistical-inference-mle]] | ALL | MLE, likelihood, sufficient statistics — theoretical backbone |
| 9 | [[entropy-mutual-info]] | ALL | Shannon entropy, MI, KL — info theory for ML |
| 9 | [[bayesian-inference]] | THEORY GEN | Bayes theorem, conjugates, posterior — Bayesian vs frequentist |
| 9 | [[evaluation-metrics-guide]] | ALL | Precision/recall/F1/AUC/mAP — can't interview without this |
| 9 | [[lora-quantization]] | LLM DL | LoRA (extremely hot) + quantization basics |
| 9 | [[transfer-learning]] | ALL | Fine-tuning, feature extraction, domain shift |
| 9 | [[bert-mlm]] | LLM | MLM, NSP, BERT pretraining — still the reference SSL model |
| 9 | [[clip]] | CV LLM GEN-AI | Contrastive image-text, zero-shot classification |
| 9 | [[arch-kv-cache]] | LLM | KV cache, memory layout, paged attention — critical for LLM serving |
| 9 | [[autoregressive-models]] | LLM GEN-AI | AR generation, teacher forcing, sampling strategies |
| 9 | [[diffusion-models]] | GEN-AI | DDPM, score matching, DDIM — most-asked GenAI topic |

---

## Tier 2 — Core (Score 7–8)

**Should know before interviews. Covers the bulk of real questions.**

### Training & Optimization

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[backpropagation-advanced]] | DL | Computational graphs, auto-diff, second-order intuition |
| 8 | [[initialization]] | DL | Xavier/He init — critical for stable training |
| 8 | [[mixed-precision]] | DL LLM | FP16/BF16, loss scaling, gradient underflow |
| 8 | [[loss-kl-divergence]] | ALL | KL divergence form, ELBO, VAE connection |
| 8 | [[loss-mse]] | GEN | MSE/L2, its assumptions, when to prefer L1/Huber |
| 8 | [[loss-landscape]] | DL THEORY | Flat minima, SAM, loss surface geometry — commonly discussed in DL research |
| 8 | [[loss-focal]] | CV | RetinaNet, $(1-p_t)^\gamma$ weighting — critical for class imbalance in detection |
| 8 | [[loss-triplet]] | CV DL | Metric learning, siamese networks, face recognition |
| 8 | [[regularization-weight-decay]] | ALL | L2 reg, AdamW decoupled weight decay |
| 8 | [[optimizer-lr-schedules]] | DL LLM | Warmup, decay, cyclical — LLM training standard |
| 8 | [[gradient-clipping]] | DL LLM | Essential for RNN/transformer stability — always asked in training discussions |
| 8 | [[model-calibration]] | GEN DL | Temperature scaling, ECE, reliability diagrams — common in production ML |
| 8 | [[data-augmentation]] | CV | Standard augmentations, RandAugment, geometric transforms |
| 8 | [[math-convexity-jensen]] | THEORY | Convex functions, Jensen's inequality, ELBO derivation |

### Architectures

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[vision-transformer]] | CV DL LLM | ViT — patch embedding, class token, scaling |
| 8 | [[rnn-lstm]] | LLM DL | LSTM gates, vanishing gradients — still tested |
| 8 | [[cnn-architectures-guide]] | CV | AlexNet → ResNet → EfficientNet progression |
| 8 | [[convolution-math]] | CV DL | Convolution operation, receptive field, output size formula |
| 8 | [[arch-depthwise-separable]] | CV | MobileNet, factored convolutions, FLOPs math |
| 8 | [[tokenization]] | LLM | BPE, WordPiece, SentencePiece — often tested |
| 8 | [[flash-attention]] | LLM DL | IO-aware attention, tiling, memory complexity |
| 8 | [[mixture-of-experts]] | LLM | Sparse MoE, routing, load balancing — Mixtral/GPT-4 style |
| 8 | [[rotary-embeddings]] | LLM | RoPE — standard in modern LLMs (LLaMA, Mistral) |
| 8 | [[cross-attention]] | DL LLM CV | Encoder-decoder, Q from one stream, K/V from another |
| 8 | [[speculative-decoding]] | LLM | Draft-then-verify, throughput gains — very hot LLM inference topic |
| 8 | [[state-space-models]] | LLM DL | Mamba/S4, linear-time sequence modeling — genuinely important now |
| 8 | [[grouped-query-attention]] | LLM | GQA/MQA — in LLaMA2+, Mistral, Gemma; always asked for LLM roles |
| 8 | [[graph-neural-networks]] | THEORY | Message passing, GCN, GAT — molecules, recommenders, social networks |
| 8 | [[iou-nms]] | CV | IoU computation, NMS, soft-NMS — core for every object detection interview |
| 8 | [[activation-relu-variants]] | DL | ReLU, LeakyReLU, dying ReLU — common architecture question |
| 8 | [[activation-softmax]] | DL | Log-sum-exp trick, numerical stability, attention weights |
| 8 | [[activation-sigmoid-tanh]] | DL | LSTM gating, saturation, vanishing gradients |
| 8 | [[vlm-architectures]] | CV LLM | LLaVA, Flamingo, PaliGemma — multimodal trend |
| 8 | [[word-embeddings]] | LLM | Word2Vec, GloVe, contextual vs static |
| 8 | [[unet]] | CV GEN-AI | Skip connections + upsampling — segmentation + diffusion |

### Theory & Classical ML

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[ensemble-methods]] | GEN | Bagging, boosting, stacking, random forests |
| 8 | [[decision-trees]] | GEN | Splits, information gain, Gini — gradient boosting foundation |
| 8 | [[support-vector-machines]] | GEN THEORY | SVM, margin, kernel trick, dual form |
| 8 | [[curse-of-dimensionality]] | GEN | High-dim geometry, concentration, distance metrics |
| 8 | [[scaling-laws]] | LLM | Chinchilla, compute-optimal training — critical for LLM discussions |
| 8 | [[model-compression]] | DL LLM | Pruning/quant/distill/low-rank overview — common system question |
| 8 | [[causal-inference]] | GEN THEORY | A/B testing, do-calculus, counterfactuals — increasingly asked in industry ML |
| 8 | [[feature-preprocessing]] | GEN | Normalization, standardization, missing data |
| 8 | [[data-leakage]] | GEN | Train-test leakage, temporal leakage — practical gotcha |

### Probability & Math

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[distributions-overview]] | GEN | All major distributions, their relationships, when to use each |
| 8 | [[expectation-maximization]] | THEORY GEN | EM algorithm, GMM training, latent variables |
| 8 | [[gaussian-mixture-models]] | GEN | GMM, soft clustering, EM for GMM |
| 8 | [[hypothesis-testing]] | GEN | p-values, t-test, type I/II errors |
| 8 | [[central-limit-theorem]] | GEN | CLT, sampling distributions, when it applies |
| 8 | [[math-svd]] | ALL | SVD, low-rank approx, PCA connection |
| 8 | [[lagrangian-optimization]] | THEORY | Lagrange multipliers, KKT conditions, SVM dual |
| 8 | [[taylor-series]] | ALL | Newton's method derivation, loss approximations, understanding activations |
| 8 | [[markov-chains]] | THEORY | Stationary distribution, mixing — foundation for RL, MCMC, HMMs |
| 8 | [[monte-carlo-methods]] | THEORY | MCMC, integration, simulation — pervasive across Bayesian ML and RL |
| 8 | [[sampling-methods]] | THEORY | Rejection sampling, Metropolis-Hastings, Gibbs — core Bayesian inference skill |
| 8 | [[variational-inference]] | THEORY GEN-AI | ELBO, mean-field VI — foundation of VAE and Bayesian DL |
| 8 | [[exponential-family]] | THEORY | Connects all distributions, GLMs, sufficient stats, natural params |
| 8 | [[gaussian-processes]] | THEORY | Bayesian optimization, uncertainty quantification, kernel regression |
| 8 | [[hessian-curvature]] | THEORY | Saddle points, second-order information, loss landscape geometry |

### Self-supervised & Adaptation

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[contrastive-learning]] | CV DL | SimCLR, InfoNCE, positives/negatives |
| 8 | [[masked-autoencoders]] | CV DL | MAE, high masking ratio, patch reconstruction |
| 8 | [[knowledge-distillation]] | DL | Soft labels, temperature, teacher-student |
| 8 | [[self-supervised-overview]] | ALL | Taxonomy of SSL — generative/discriminative/contrastive |
| 8 | [[rlhf]] | LLM GEN-AI | Reward model, PPO, preference learning |
| 8 | [[instruction-tuning]] | LLM | SFT, instruction format, FLAN, chat fine-tuning |
| 8 | [[dpo-preference]] | LLM | DPO — RL-free alternative to RLHF, very current |
| 7 | [[prompt-engineering]] | LLM | CoT, few-shot, system prompts — practical but often tested |

### Generative Models

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[variational-autoencoders]] | GEN-AI THEORY | VAE, ELBO, reparameterization trick |
| 8 | [[generative-adversarial-networks]] | GEN-AI | GAN loss, mode collapse, Wasserstein GAN |

---

## Tier 3 — Good to Know (Score 6–7)

**Read these if you have time or the role matches.**

### Training

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 7 | [[cosine-annealing]] | DL LLM | SGDR, OneCycleLR, warm restarts |
| 7 | [[gradient-accumulation]] | DL LLM | Simulate large batches on limited VRAM |
| 7 | [[gradient-checkpointing]] | DL LLM | O(√L) memory trade-off — essential for long-context training |
| 7 | [[mixup-cutmix]] | CV | Data mixing strategies, decision boundary smoothing |
| 7 | [[loss-nt-xent]] | DL | SimCLR loss, temperature-scaled contrastive |
| 7 | [[large-batch-training]] | DL LLM | Linear LR scaling, warmup, sharp vs flat minima |
| 7 | [[regularization-early-stopping]] | GEN | Validation-based stopping, implicit regularization |
| 7 | [[regularization-label-smoothing]] | DL | Over-confidence prevention, soft targets |
| 7 | [[second-order-optimization]] | THEORY | K-FAC, natural gradient, Newton's method |
| 7 | [[optimizer-rmsprop-adagrad]] | DL | Adaptive LR history; precursor to Adam |
| 7 | [[loss-huber]] | GEN | Robust regression, outlier handling |
| 7 | [[loss-hinge]] | GEN THEORY | SVM loss, margin-based learning |
| 7 | [[loss-ranking]] | LLM | RankNet, LambdaRank — RLHF and IR use ranking losses |

### Architectures

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 7 | [[feature-pyramid-networks]] | CV | Multi-scale feature hierarchy, FPN neck |
| 7 | [[sparse-attention]] | LLM | Local/strided patterns, Longformer, BigBird |
| 7 | [[dilated-convolution]] | CV | Expanded receptive field without stride, DeepLab |
| 7 | [[arch-bottleneck-1x1]] | CV | Channel compression, ResNet v2 design |
| 7 | [[pooling-operations]] | CV | Global average pooling, max pooling, spatial pyramid |
| 7 | [[vit-training-recipe]] | CV DL | DeiT, AugReg, training tricks for ViT |
| 7 | [[activation-gelu-swish]] | DL LLM | GELU in transformers, SiLU/Swish in LLaMA FFN |
| 7 | [[flash-decoding]] | LLM | Parallel KV reduction for long contexts |
| 7 | [[anchor-free-detection]] | CV | FCOS, CenterNet, DETR — modern detection paradigm |
| 7 | [[arch-roi-align]] | CV | Bilinear interpolation vs quantization, Mask R-CNN |
| 7 | [[grouped-convolution]] | CV DL | FLOPs/parameter reduction, ResNeXt cardinality |
| 7 | [[squeeze-excitation]] | CV | Channel-wise recalibration, SE blocks |

### Theory & Data

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 7 | [[hyperparameter-optimization]] | GEN | Grid/random/Bayesian search, Hyperband |
| 7 | [[kernel-methods]] | THEORY | Kernel trick, Mercer's theorem, kernel PCA |
| 7 | [[emergent-abilities]] | LLM | Phase transitions, scale-dependent capabilities |
| 7 | [[class-imbalance]] | GEN | Focal loss, class weights, threshold tuning |
| 7 | [[clustering]] | GEN | K-means, DBSCAN, hierarchical, evaluation |
| 7 | [[generative-vs-discriminative]] | GEN THEORY | Naive Bayes vs logistic regression, density estimation |
| 7 | [[categorical-encoding]] | GEN | One-hot, target encoding, embeddings |
| 7 | [[imbalanced-sampling]] | GEN | SMOTE, ADASYN, Tomek links |
| 7 | [[test-time-strategies]] | DL | TTA, ensembling, MC dropout for uncertainty |
| 7 | [[domain-adaptation]] | DL | Covariate shift, distribution alignment, DANN |
| 7 | [[continual-learning]] | DL | Catastrophic forgetting, EWC, replay — common question about LLM updates |
| 7 | [[meta-learning]] | THEORY | MAML, prototypical networks, few-shot learning |
| 7 | [[pruning-sparsity]] | DL | Magnitude/structured pruning, SparseGPT, lottery ticket |
| 7 | [[anomaly-detection]] | GEN | IsolationForest, reconstruction error, one-class SVM |
| 7 | [[generalization-bounds]] | THEORY | VC dimension, sample complexity, PAC framework |
| 7 | [[label-noise]] | GEN | Noisy labels, Co-teaching, confident learning |
| 7 | [[outlier-detection]] | GEN | Statistical tests, DBSCAN, distance-based |
| 7 | [[vlm-explainability]] | CV LLM | Grad-CAM, attention rollout, saliency maps |

### Probability & Math

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 7 | [[bootstrap]] | GEN | Resampling, confidence intervals, bagging connection |
| 7 | [[fourier-transform]] | ALL | Convolution theorem, audio ML, spectral methods, random Fourier features |
| 7 | [[manifold-learning]] | THEORY | Manifold hypothesis, t-SNE, UMAP — visualization frequently asked |
| 7 | [[optimal-transport]] | THEORY GEN-AI | Wasserstein distance, Earth mover's, used in GANs and domain adaptation |
| 7 | [[fisher-information]] | THEORY | Natural gradient, Cramér-Rao bound, FIM in Bayesian ML |
| 7 | [[change-of-variables]] | THEORY | Reparameterization trick, normalizing flow foundation |
| 7 | [[hidden-markov-models]] | THEORY | Viterbi, forward-backward, sequence labeling history |
| 7 | [[importance-sampling]] | THEORY | Off-policy RL evaluation, variance reduction, MCMC |
| 7 | [[dirichlet-distribution]] | THEORY | LDA topic models, Bayesian priors, concentration parameter |

### Self-supervised & Adaptation

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 7 | [[byol-simsiam]] | CV | SSL without negatives, stop-gradient trick |
| 7 | [[dino-dinov2]] | CV | Self-distillation, patch-level features, strong backbone |
| 7 | [[speculative-pretraining]] | LLM | T5 span masking, UL2, ELECTRA, FIM — beyond BERT/GPT |
| 7 | [[quantization-gguf]] | LLM DL | k-quants, Q4_K_M, GGUF format — LLM deployment standard |
| 6 | [[adapter-layers]] | LLM DL | Bottleneck adapters, parameter-efficient alternative to LoRA |
| 6 | [[prefix-tuning]] | LLM | Soft prompt tokens prepended to each layer |

### Generative

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 7 | [[autoencoder-variants]] | GEN-AI | DAE, SAE, CAE — from compression to LLM interpretability |
| 7 | [[diffusion-guidance]] | GEN-AI | Classifier guidance, CFG, negative prompts |
| 7 | [[flow-matching]] | GEN-AI | Continuous normalizing flows, OT paths — surpassing diffusion |
| 7 | [[normalizing-flows]] | GEN-AI THEORY | Exact likelihood, invertible transforms, RealNVP |
| 6 | [[energy-based-models]] | THEORY | Unnormalized densities, contrastive divergence |

---

## Tier 4 — Specialized (Score 1–6)

**Only if the role specifically demands it, or you have extra time.**

| Score | Topic | Why it's low |
|-------|-------|-------------|
| 6 | [[adversarial-training]] | DL — robustness to perturbations, PGD, FGSM |
| 6 | [[loss-dice]] | CV — segmentation-specific F1 loss |
| 6 | [[loss-huber-detail]] | GEN — quantile/pinball regression, expectile |
| 6 | [[stochastic-depth]] | CV DL — random layer dropping, scaling connection |
| 6 | [[parametric-nonparametric]] | GEN — comparing model families |
| 6 | [[credible-intervals]] | GEN — Bayesian CI vs frequentist CI |
| 6 | [[blip]] | CV LLM — VLP pretraining with bootstrapping |
| 6 | [[neural-tangent-kernel]] | THEORY — infinite-width networks, kernel regime |
| 6 | [[spectral-bias]] | THEORY — networks learn low-frequency first, NTK connection |
| 6 | [[coordinate-descent]] | THEORY — LASSO coordinate update, BCD, SMO |
| 6 | [[information-geometry]] | THEORY — natural gradient, statistical manifolds |
| 5 | [[grokking]] | Niche research topic |
| 5 | [[lottery-ticket-hypothesis]] | Research curiosity, rarely tested |
| 5 | [[model-merging]] | Emerging technique, not yet standard |
| 5 | [[lookahead-optimizer]] | Niche optimizer, Ranger = Lookahead + RAdam |
| 5 | [[generating-functions]] | MGF, cumulants — statistics depth |
| 5 | [[numerical-methods]] | Numerical stability, floating point — useful for impl |
| 5 | [[maximum-entropy-principle]] | Deep theory, rarely directly tested |
| 5 | [[sufficient-statistics]] | Statistics depth, useful for THEORY track |
| 5 | [[data2vec]] | Niche SSL model |
| 5 | [[energy-based-models]] | Contrastive divergence, niche |
| 5 | [[pac-learning]] | Pure theory (PAC, VC dim) — important only for research roles |
| 5 | [[rademacher-complexity]] | Pure theory — complement to generalization-bounds |
| 5 | [[kronecker-products]] | K-FAC / optimization research |
| 4 | [[copulas]] | Statistics specialty, rare in ML interviews |
| 4 | [[point-processes]] | Hawkes process — event sequence specialty |
| 4 | [[gram-matrix-theory]] | Style transfer / kernel niche |
| 4 | [[mixture-of-depths]] | Very new, not yet standard in interviews |
| 4 | [[multi-source-fusion]] | Domain-specific |
| 4 | [[visualization-guide]] | Practical tool, not interview theory |
| 3 | [[remote-sensing-foundation-models]] | Domain-specific (EO) |
| 3 | [[earth-observation-fundamentals]] | Domain-specific (EO) |
| 2 | [[biodiversity-ml]] | Very domain-specific |
| 2 | [[satellite-imagery-preprocessing]] | Very domain-specific |

---

## Recommended Reading Orders by Track

### General ML (40 files, ~3 weeks)
Tier 1 universals → [[decision-trees]] · [[ensemble-methods]] · [[clustering]] · [[support-vector-machines]] · [[cross-validation]] · [[hyperparameter-optimization]] · [[evaluation-metrics-guide]] · [[feature-preprocessing]] · [[data-leakage]] · [[distributions-overview]] · [[hypothesis-testing]] · [[expectation-maximization]] · [[gaussian-mixture-models]] · [[causal-inference]] · [[model-calibration]]

### Deep Learning Engineer (50 files, ~4 weeks)
Tier 1 → all DL-tagged Tier 2 → [[mixed-precision]] · [[gradient-checkpointing]] · [[gradient-clipping]] · [[model-compression]] · [[pruning-sparsity]] · [[backpropagation-advanced]] · [[initialization]] · [[large-batch-training]] · [[loss-landscape]] · [[second-order-optimization]] · [[state-space-models]]

### NLP / LLM (45 files, ~3 weeks)
[[attention-mechanism]] · [[arch-kv-cache]] · [[arch-positional-encoding]] · [[tokenization]] · [[bert-mlm]] → [[rotary-embeddings]] · [[grouped-query-attention]] · [[sparse-attention]] · [[speculative-decoding]] · [[flash-attention]] · [[flash-decoding]] · [[lora-quantization]] · [[rlhf]] · [[dpo-preference]] · [[instruction-tuning]] · [[scaling-laws]] · [[emergent-abilities]] · [[speculative-pretraining]] · [[mixture-of-experts]] · [[autoregressive-models]] · [[quantization-gguf]]

### Computer Vision (45 files, ~3 weeks)
[[vision-transformer]] · [[arch-residual-block]] · [[clip]] → [[cnn-architectures-guide]] · [[convolution-math]] · [[arch-depthwise-separable]] · [[feature-pyramid-networks]] · [[unet]] · [[iou-nms]] · [[arch-roi-align]] · [[dilated-convolution]] · [[data-augmentation]] · [[mixup-cutmix]] · [[contrastive-learning]] · [[masked-autoencoders]] · [[dino-dinov2]] · [[anchor-free-detection]] · [[vlm-architectures]] · [[loss-focal]]

### Generative AI (35 files, ~2.5 weeks)
[[diffusion-models]] → [[variational-autoencoders]] → [[autoregressive-models]] → [[generative-adversarial-networks]] → [[clip]] → [[rlhf]] → [[dpo-preference]] → [[diffusion-guidance]] → [[flow-matching]] → [[normalizing-flows]] → [[autoencoder-variants]] → [[vlm-architectures]] → [[optimal-transport]]

### Research / Theory (35 files, ~3 weeks)
Tier 1 math/probability → [[pac-learning]] · [[rademacher-complexity]] · [[generalization-bounds]] · [[neural-tangent-kernel]] · [[information-geometry]] · [[fisher-information]] · [[variational-inference]] · [[gaussian-processes]] · [[kernel-methods]] · [[optimal-transport]] · [[causal-inference]] · [[spectral-bias]] · [[second-order-optimization]] · [[exponential-family]] · [[markov-chains]] · [[monte-carlo-methods]]

---

## Quick-Score Summary

| Score | Count | Summary |
|-------|-------|---------|
| 10 | 2 | Absolute must: [[backpropagation]] + [[attention-mechanism]] |
| 9 | 27 | Core fundamentals — cover these first |
| 8 | ~55 | Strong preparation — most interviews touch these |
| 7 | ~55 | Good depth — differentiates candidates |
| 6 | ~12 | Role-specific or supplementary |
| 5 | ~10 | Niche / research interest |
| 4 | 6 | Specialty only |
| 2–3 | 4 | Domain-specific (EO/biodiversity) — skip unless applying to remote sensing roles |

**Minimum viable prep (score ≥ 9):** 29 files — covers ~80% of interview questions for most DL/ML roles.  
**Solid prep (score ≥ 8):** ~84 files — strong across all tracks.  
**Comprehensive (score ≥ 7):** ~139 files — ready for senior/research roles.
