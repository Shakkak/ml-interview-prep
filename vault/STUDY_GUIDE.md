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
| 8 | [[regularization-weight-decay]] | ALL | L2 reg, AdamW decoupled weight decay |
| 8 | [[optimizer-lr-schedules]] | DL LLM | Warmup, decay, cyclical — LLM training standard |
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
| 8 | [[vlm-architectures]] | CV LLM | LLaVA, Flamingo, PaliGemma — multimodal trend |
| 8 | [[word-embeddings]] | LLM | Word2Vec, GloVe, contextual vs static |
| 8 | [[unet]] | CV GEN-AI | Skip connections + upsampling — segmentation + diffusion |
| 8 | [[activation-relu-variants]] | DL | ReLU, LeakyReLU, dying ReLU — common architecture question |

### Theory & Classical ML

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[ensemble-methods]] | GEN | Bagging, boosting, stacking, random forests |
| 8 | [[decision-trees]] | GEN | Splits, information gain, Gini — gradient boosting foundation |
| 8 | [[support-vector-machines]] | GEN THEORY | SVM, margin, kernel trick, dual form |
| 8 | [[curse-of-dimensionality]] | GEN | High-dim geometry, concentration, distance metrics |
| 8 | [[scaling-laws]] | LLM | Chinchilla, compute-optimal training — critical for LLM discussions |
| 8 | [[model-compression]] | DL LLM | Pruning/quant/distill/low-rank overview — common system question |
| 8 | [[feature-preprocessing]] | GEN | Normalization, standardization, missing data |
| 8 | [[data-leakage]] | GEN | Train-test leakage, temporal leakage — practical gotcha |

### Probability & Math

| Score | Topic | Tags | One-line |
|-------|-------|------|---------|
| 8 | [[distributions-overview]] | GEN | Exponential family, common distributions |
| 8 | [[expectation-maximization]] | THEORY GEN | EM algorithm, GMM training, latent variables |
| 8 | [[gaussian-mixture-models]] | GEN | GMM, soft clustering, EM for GMM |
| 8 | [[hypothesis-testing]] | GEN | p-values, t-test, type I/II errors |
| 8 | [[central-limit-theorem]] | GEN | CLT, sampling distributions, when it applies |
| 8 | [[math-svd]] | ALL | SVD, low-rank approx, PCA connection |
| 8 | [[lagrangian-optimization]] | THEORY | Lagrange multipliers, KKT conditions, SVM dual |

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

| Score | Topic | Tags |
|-------|-------|------|
| 7 | [[cosine-annealing]] | DL LLM |
| 7 | [[gradient-accumulation]] | DL LLM |
| 7 | [[gradient-checkpointing]] | DL LLM |
| 7 | [[gradient-clipping]] | DL |
| 7 | [[mixup-cutmix]] | CV |
| 7 | [[loss-triplet]] | CV DL |
| 7 | [[loss-landscape]] | THEORY |
| 7 | [[loss-focal]] | CV |
| 7 | [[loss-nt-xent]] | DL |
| 7 | [[large-batch-training]] | DL LLM |
| 7 | [[regularization-early-stopping]] | GEN |
| 7 | [[regularization-label-smoothing]] | DL |
| 7 | [[second-order-optimization]] | THEORY |
| 7 | [[optimizer-rmsprop-adagrad]] | DL |

### Architectures

| Score | Topic | Tags |
|-------|-------|------|
| 7 | [[feature-pyramid-networks]] | CV |
| 7 | [[grouped-query-attention]] | LLM |
| 7 | [[sparse-attention]] | LLM |
| 7 | [[speculative-decoding]] | LLM |
| 7 | [[state-space-models]] | LLM DL |
| 7 | [[dilated-convolution]] | CV |
| 7 | [[graph-neural-networks]] | THEORY |
| 7 | [[arch-bottleneck-1x1]] | CV |
| 7 | [[pooling-operations]] | CV |
| 7 | [[iou-nms]] | CV |
| 7 | [[vit-training-recipe]] | CV DL |
| 7 | [[activation-gelu-swish]] | DL LLM |
| 7 | [[activation-sigmoid-tanh]] | DL |
| 7 | [[activation-softmax]] | DL |

### Theory & Data

| Score | Topic | Tags |
|-------|-------|------|
| 7 | [[hyperparameter-optimization]] | GEN |
| 7 | [[kernel-methods]] | THEORY |
| 7 | [[model-calibration]] | GEN |
| 7 | [[emergent-abilities]] | LLM |
| 7 | [[causal-inference]] | GEN THEORY |
| 7 | [[class-imbalance]] | GEN |
| 7 | [[clustering]] | GEN |
| 7 | [[generative-vs-discriminative]] | GEN THEORY |
| 7 | [[categorical-encoding]] | GEN |
| 7 | [[imbalanced-sampling]] | GEN |
| 7 | [[test-time-strategies]] | DL |

### Probability & Math

| Score | Topic | Tags |
|-------|-------|------|
| 7 | [[bootstrap]] | GEN |
| 7 | [[exponential-family]] | THEORY |
| 7 | [[gaussian-processes]] | THEORY |
| 7 | [[markov-chains]] | THEORY |
| 7 | [[monte-carlo-methods]] | THEORY |
| 7 | [[sampling-methods]] | THEORY |
| 7 | [[variational-inference]] | THEORY GEN-AI |
| 7 | [[hessian-curvature]] | THEORY |
| 7 | [[taylor-series]] | THEORY |

### Self-supervised & Adaptation

| Score | Topic | Tags |
|-------|-------|------|
| 7 | [[byol-simsiam]] | CV |
| 7 | [[dino-dinov2]] | CV |
| 7 | [[speculative-pretraining]] | LLM |
| 6 | [[adapter-layers]] | LLM DL |
| 6 | [[prefix-tuning]] | LLM |

### Generative

| Score | Topic | Tags |
|-------|-------|------|
| 7 | [[autoencoder-variants]] | GEN-AI |
| 7 | [[diffusion-guidance]] | GEN-AI |
| 6 | [[flow-matching]] | GEN-AI |
| 6 | [[normalizing-flows]] | GEN-AI THEORY |
| 5 | [[energy-based-models]] | THEORY |

### Remaining Score-6 Topics

| Score | Topic | Tags |
|-------|-------|------|
| 6 | [[anchor-free-detection]] | CV |
| 6 | [[arch-roi-align]] | CV |
| 6 | [[flash-decoding]] | LLM |
| 6 | [[grouped-convolution]] | CV DL |
| 6 | [[squeeze-excitation]] | CV |
| 6 | [[adversarial-training]] | DL |
| 6 | [[loss-dice]] | CV |
| 6 | [[loss-hinge]] | GEN |
| 6 | [[loss-huber]] | GEN |
| 6 | [[loss-huber-detail]] | GEN |
| 6 | [[loss-ranking]] | LLM |
| 6 | [[stochastic-depth]] | CV DL |
| 6 | [[anomaly-detection]] | GEN |
| 6 | [[continual-learning]] | DL |
| 6 | [[domain-adaptation]] | DL |
| 6 | [[generalization-bounds]] | THEORY |
| 6 | [[meta-learning]] | THEORY |
| 6 | [[parametric-nonparametric]] | GEN |
| 6 | [[pruning-sparsity]] | DL |
| 6 | [[vlm-explainability]] | CV LLM |
| 6 | [[active-learning]] | GEN |
| 6 | [[label-noise]] | GEN |
| 6 | [[outlier-detection]] | GEN |
| 6 | [[change-of-variables]] | THEORY |
| 6 | [[credible-intervals]] | GEN |
| 6 | [[dirichlet-distribution]] | THEORY |
| 6 | [[fisher-information]] | THEORY |
| 6 | [[hidden-markov-models]] | THEORY |
| 6 | [[importance-sampling]] | THEORY |
| 6 | [[quantization-gguf]] | LLM DL |
| 6 | [[blip]] | CV LLM |
| 6 | [[fourier-transform]] | THEORY |
| 6 | [[manifold-learning]] | THEORY |
| 6 | [[optimal-transport]] | THEORY GEN-AI |

---

## Tier 4 — Specialized (Score 1–5)

**Only if the role specifically demands it, or you have extra time.**

| Score | Topic | Why it's low |
|-------|-------|-------------|
| 5 | [[grokking]] | Niche research topic |
| 5 | [[lottery-ticket-hypothesis]] | Research curiosity, rarely tested |
| 5 | [[model-merging]] | Emerging technique |
| 5 | [[neural-tangent-kernel]] | Deep theory, rarely in applied interviews |
| 5 | [[spectral-bias]] | Research-level |
| 5 | [[coordinate-descent]] | Rarely tested directly |
| 5 | [[lookahead-optimizer]] | Niche optimizer |
| 5 | [[generating-functions]] | Statistical theory depth |
| 5 | [[information-geometry]] | Very advanced THEORY |
| 5 | [[numerical-methods]] | Numerical computing context |
| 5 | [[maximum-entropy-principle]] | Deep theory |
| 5 | [[sufficient-statistics]] | Statistics depth |
| 5 | [[data2vec]] | Niche SSL model |
| 4 | [[pac-learning]] | Pure theory (PAC, VC dim) — only for research roles |
| 4 | [[rademacher-complexity]] | Pure theory |
| 4 | [[copulas]] | Statistics specialty |
| 4 | [[point-processes]] | Specialty (event sequences, Hawkes) |
| 4 | [[kronecker-products]] | K-FAC / optimization research |
| 4 | [[gram-matrix-theory]] | Style transfer / kernel methods niche |
| 4 | [[mixture-of-depths]] | Very new, not yet standard |
| 4 | [[multi-source-fusion]] | Domain-specific |
| 4 | [[visualization-guide]] | Practical tool, not theory |
| 3 | [[remote-sensing-foundation-models]] | Domain-specific (EO) |
| 3 | [[earth-observation-fundamentals]] | Domain-specific (EO) |
| 2 | [[biodiversity-ml]] | Very domain-specific |
| 2 | [[satellite-imagery-preprocessing]] | Very domain-specific |

---

## Recommended Reading Orders by Track

### General ML (40 files, ~3 weeks)
Tier 1 universals → [[decision-trees]] · [[ensemble-methods]] · [[clustering]] · [[support-vector-machines]] · [[cross-validation]] · [[hyperparameter-optimization]] · [[evaluation-metrics-guide]] · [[feature-preprocessing]] · [[data-leakage]] · [[distributions-overview]] · [[hypothesis-testing]] · [[expectation-maximization]] · [[gaussian-mixture-models]]

### Deep Learning Engineer (50 files, ~4 weeks)
Tier 1 → all DL-tagged Tier 2 → [[mixed-precision]] · [[gradient-checkpointing]] · [[model-compression]] · [[pruning-sparsity]] · [[backpropagation-advanced]] · [[initialization]] · [[large-batch-training]] · [[second-order-optimization]]

### NLP / LLM (45 files, ~3 weeks)
[[attention-mechanism]] · [[arch-kv-cache]] · [[arch-positional-encoding]] · [[tokenization]] · [[bert-mlm]] → [[rotary-embeddings]] · [[grouped-query-attention]] · [[sparse-attention]] · [[speculative-decoding]] · [[flash-attention]] · [[flash-decoding]] · [[lora-quantization]] · [[rlhf]] · [[dpo-preference]] · [[instruction-tuning]] · [[scaling-laws]] · [[emergent-abilities]] · [[speculative-pretraining]] · [[mixture-of-experts]] · [[autoregressive-models]]

### Computer Vision (45 files, ~3 weeks)
[[vision-transformer]] · [[arch-residual-block]] · [[clip]] → [[cnn-architectures-guide]] · [[convolution-math]] · [[arch-depthwise-separable]] · [[feature-pyramid-networks]] · [[unet]] · [[iou-nms]] · [[arch-roi-align]] · [[dilated-convolution]] · [[data-augmentation]] · [[mixup-cutmix]] · [[contrastive-learning]] · [[masked-autoencoders]] · [[dino-dinov2]] · [[anchor-free-detection]] · [[vlm-architectures]]

### Generative AI (35 files, ~2.5 weeks)
[[diffusion-models]] → [[variational-autoencoders]] → [[autoregressive-models]] → [[generative-adversarial-networks]] → [[clip]] → [[rlhf]] → [[dpo-preference]] → [[diffusion-guidance]] → [[flow-matching]] → [[normalizing-flows]] → [[autoencoder-variants]] → [[vlm-architectures]]

### Research / Theory (35 files, ~3 weeks)
Tier 1 math/probability → [[pac-learning]] · [[rademacher-complexity]] · [[generalization-bounds]] · [[neural-tangent-kernel]] · [[information-geometry]] · [[fisher-information]] · [[variational-inference]] · [[gaussian-processes]] · [[kernel-methods]] · [[optimal-transport]] · [[causal-inference]] · [[spectral-bias]] · [[second-order-optimization]]

---

## Quick-Score Summary

| Score | Count | Summary |
|-------|-------|---------|
| 10 | 2 | Absolute must: [[backpropagation]] + [[attention-mechanism]] |
| 9 | 27 | Core fundamentals — cover these first |
| 8 | 36 | Strong preparation — most interviews touch these |
| 7 | 35 | Good depth — differentiates candidates |
| 6 | 36 | Role-specific or supplementary |
| 5 | 13 | Niche / research interest |
| 4 | 8 | Specialty only |
| 2–3 | 4 | Domain-specific (EO/biodiversity) — skip unless applying to remote sensing roles |

**Minimum viable prep (score ≥ 9):** 29 files — covers ~80% of interview questions for most DL/ML roles.  
**Solid prep (score ≥ 8):** 65 files — strong across all tracks.  
**Comprehensive (score ≥ 7):** 100 files — ready for senior/research roles.
