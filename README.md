# ml-interview-prep

An Anki deck and Obsidian vault for ML interview preparation — from core fundamentals to advanced research depth. Covers deep learning, statistics, computer vision, transformers, generative models, self-supervised learning, and Earth observation.

---

## Contents

| | |
|---|---|
| **Anki deck** | 624 cards across 21 sub-decks |
| **Obsidian vault** | 109 concept notes across 9 folders |

---

## Anki Deck

Download the latest `interview-prep.apkg` from the [Releases page](https://github.com/Shakkak/ml-interview-prep/releases) and import into Anki via **File → Import**.

### Sub-decks

| # | Sub-deck | Cards | Topics |
|---|----------|------:|--------|
| 01 | Math Foundations | 34 | Loss functions, backpropagation, KL divergence, entropy, convexity, Jensen's inequality |
| 02 | Linear Algebra | 24 | Subspaces, PCA, SVD, norms, Jacobian, Hessian, Woodbury identity |
| 03 | Probability & Distributions | 38 | Gaussian, Beta, Dirichlet, exponential family, CLT, MCMC, HMC |
| 04 | Statistical Inference | 33 | MLE, MAP, hypothesis testing, bootstrap, causal inference, AIC/BIC |
| 05 | Training Fundamentals | 49 | Vanishing gradients, weight init, dropout, batch norm, Adam vs SGD, stochastic depth, LLRD |
| 06 | Evaluation Metrics | 17 | Precision/recall, ROC/AUC, mAP, IoU, calibration, SSIM/PSNR |
| 07 | Data Preprocessing | 36 | Normalisation, augmentation, Fourier transforms, wavelet analysis, spectral bias |
| 08 | Data Visualisation | 18 | Histogram, scatter, boxplot, heatmap, Tufte principles, chart selection |
| 09 | CNN Architectures | 21 | VGG, ResNet, Inception, MobileNet, EfficientNet, FPN, U-Net, SE networks, ConvNeXt |
| 10 | Transformers & Attention | 43 | QKV, multi-head attention, RoPE, Flash Attention, GQA, Swin, ViT training recipe, LLRD |
| 11 | Self-Supervised Learning | 14 | SimCLR, MoCo, BYOL, MAE, BERT, DINO, DINOv2, BEiT, backbone selection |
| 12 | Fine-Tuning & Adaptation | 24 | LoRA, QLoRA, RLHF, DPO, instruction tuning, knowledge distillation |
| 13 | Advanced ML Theory | 52 | Spectral bias, NTK, information bottleneck, optimal transport, generalisation bounds |
| 14 | Generative Models | 38 | VAEs, GANs, WGAN, diffusion (DDPM/DDIM/CFG/LDM), normalising flows, autoregressive |
| 15 | Vision-Language Models | 23 | CLIP, SigLIP, BLIP-2, LLaVA, Flamingo, Q-Former, hallucination, benchmarks (MMMU/GQA) |
| 16 | CV Interview Essentials | 36 | Detection (Faster R-CNN, YOLO, FCOS), clustering (k-means, DBSCAN, GMM), inductive bias |
| 17 | Segmentation | 29 | Mask R-CNN, DeepLab, FCN, Mask2Former, Otsu, watershed |
| 18 | Cross-Topic Synthesis | 55 | Questions that connect concepts across multiple areas |
| 20 | Graph Neural Networks | 5 | Message passing, GCN, GAT, spectral vs spatial |
| 21 | Earth Observation & Remote Sensing | 18 | Spectral bands, NDVI, SAR, GeoTIFF, Prithvi, RemoteCLIP |
| 22 | Biodiversity & Ecology ML | 17 | Species distribution modelling, MaxEnt, GBIF, long-tail recognition, BioCLIP |

### Card structure

Each card has:

- **Question** — interview-style, with optional `[Junior]` / `[Mid-level]` / `[Senior]` / `[Paper]` prefix
- **Answer** — what a strong candidate says in 2–4 sentences
- **Steps** — deeper follow-up: derivations, worked examples, comparison tables, edge cases
- **Hint** — first move for non-obvious proofs (omitted when the question is self-explanatory)

### Card tags

Cards carry two kinds of tags usable in Anki's browser to filter by role, domain, or question type:

| Prefix | Values |
|--------|--------|
| `role:` | `mle`, `ds`, `cv-engineer`, `nlp-engineer`, `research`, `eo-specialist` |
| `domain:` | `ml-core`, `deep-learning`, `computer-vision`, `transformers-nlp`, `self-supervised`, `generative-models`, `statistics`, `math`, `production-ml`, `vision-language`, `earth-observation`, `biodiversity-ml` |
| `style:` | `conceptual`, `derivation`, `problem-solving`, `practical`, `tradeoffs`, `debugging` |

Example: filter `role:mle domain:transformers-nlp` in Anki's browser to study only transformer cards relevant to an MLE role.

---

## Obsidian Vault

Clone the repo and open the `vault/` folder as an Obsidian vault.

109 concept notes across 9 folders. Each note has:
- **Fundamental / Intermediate / Advanced** sections
- `[[wikilinks]]` to related concepts (visible in the graph view)
- Key equations, worked examples, and `[!tip]` / `[!warning]` callouts for non-obvious results

### Folder overview

| Folder | Notes | Concepts |
|--------|------:|---------|
| `architectures/` | 23 | Attention, ViT, Swin, Flash Attention, RNN/LSTM, CNNs, U-Net, VLM architectures, open-vocabulary detection, positional encodings, 1×1 convolutions, depthwise separable convs, ViT training recipe |
| `training/` | 21 | Backpropagation, optimisers (Adam, SGD, RMSProp), batch norm, dropout, loss functions, data augmentation, mixed precision, LR schedules, regularisation |
| `theory/` | 19 | Bias-variance, spectral bias, NTK, kernel methods, decision trees, ensemble methods, calibration, anomaly detection, clustering, causal inference, domain adaptation |
| `probability/` | 11 | Bayesian inference, MLE, entropy, Gaussian, distributions, hypothesis testing, MCMC/HMC, bootstrap, Fisher information |
| `math/` | 9 | Linear algebra, SVD, PCA, Fourier transform, matrix calculus, convexity, optimal transport, Lagrangian optimisation, means (AM/GM/HM) |
| `self-supervised/` | 8 | CLIP, BLIP, BERT, SimCLR/MoCo/BYOL, DINO/DINOv2, knowledge distillation, remote sensing foundation models |
| `adaptation/` | 5 | LoRA/QLoRA, RLHF, instruction tuning, prompt engineering, transfer learning |
| `generative/` | 5 | VAEs, GANs, diffusion models (DDPM/DDIM/CFG), normalising flows, autoregressive models |
| `data/` | 8 | Evaluation metrics, feature preprocessing, earth observation, satellite imagery, multi-source fusion, visualisation |

---

## License

[CC BY-NC-SA 4.0](LICENSE) — free to use and adapt for non-commercial purposes with attribution.
