# ml-interview-prep

An Anki deck and Obsidian vault for ML interview preparation. Covers deep learning, statistics, NLP, computer vision, generative models, and Earth observation — from core fundamentals to advanced research-level depth.

---

## Anki Deck

Download `interview-prep.apkg` and import into Anki via **File → Import**.

The deck is organised into 22 sub-decks:

| Sub-deck | Topics |
|---|---|
| Mathematical Foundations | Loss functions, gradient descent, Adam, backpropagation, KL divergence, entropy |
| Linear Algebra | Subspaces, PCA, SVD, norms, condition number, Jacobian, Hessian, Woodbury |
| Probability & Distributions | Gaussian, Poisson, Beta, Dirichlet, exponential family, CLT, MCMC, HMC |
| Statistical Inference | MLE, MAP, hypothesis testing, bootstrap, causal inference, AIC/BIC, logistic regression |
| Training Fundamentals | Vanishing gradients, weight initialisation, dropout, batch normalisation, Adam vs SGD, mixed precision |
| Evaluation Metrics | Precision/recall, ROC/AUC, mAP, IoU, calibration, cross-validation |
| Data Preprocessing | Normalisation, augmentation, Fourier transforms, bilateral filtering, wavelet analysis |
| CNN Architectures | VGG, ResNet, Inception, MobileNet, EfficientNet, FPN, U-Net, SE networks, ConvNeXt |
| Transformers & Attention | QKV, multi-head attention, positional encodings, RoPE, Flash Attention, GQA, Swin |
| Self-Supervised Learning | SimCLR, MoCo, BYOL, MAE, BERT, DeBERTa, contrastive learning |
| Fine-Tuning & Adaptation | LoRA, QLoRA, full fine-tuning, RLHF, DPO, instruction tuning, knowledge distillation |
| Generative Models | VAEs, GANs, WGAN, diffusion models (DDPM/DDIM/CFG/LDM), normalising flows, autoregressive |
| Vision-Language Models | CLIP, BLIP-2, LLaVA, Flamingo, InstructBLIP, Q-Former, grounding |
| Computer Vision Interview | Detection (Faster R-CNN, YOLO, FCOS), segmentation, anchors, NMS |
| Segmentation | Classical pipelines, Otsu, watershed, Mask R-CNN, DeepLab, FCN, Mask2Former |
| Advanced ML Reasoning | Spectral bias, NTK, information bottleneck, optimal transport, generalisation bounds |
| Earth Observation | Spectral bands, NDVI, SAR, GeoTIFF, temporal compositing, Prithvi, RemoteCLIP |
| Biodiversity ML | Species distribution modelling, MaxEnt, GBIF, long-tail recognition, BioCLIP, Grad-CAM |
| Advanced Preprocessing | Fourier analysis, DCT, Gabor filters, deconvolution, spectral bias in CNNs |
| Data Visualisation | Histogram, scatter, boxplot, heatmap, Tufte principles, chart selection |
| Cross-Topic Synthesis | Questions that connect concepts across multiple areas |
| Plain-English Definitions | Core concepts explained without jargon |

### Card structure

Each card has:
- **Front** — interview-style question (with difficulty level: Junior / Mid-level / Senior)
- **Back** — strong candidate answer in 2–4 sentences
- **Steps** — deeper follow-up: derivations, edge cases, connections to other concepts
- **Hint** — first move for non-obvious proofs or derivations

### Card tags

Cards are tagged to describe who they are for and what kind of question they are:

| Tag | Values |
|---|---|
| `role:` | `mle`, `ds`, `cv-engineer`, `nlp-engineer`, `research`, `eo-specialist` |
| `domain:` | `ml-core`, `deep-learning`, `computer-vision`, `transformers-nlp`, `self-supervised`, `generative-models`, `statistics`, `math`, `production-ml`, `vision-language`, `earth-observation`, `biodiversity-ml` |
| `style:` | `conceptual`, `derivation`, `problem-solving`, `practical`, `tradeoffs`, `debugging` |

You can use these tags in Anki's browser to filter and study a focused subset (e.g. all `role:mle` + `domain:transformers-nlp` cards).

---

## Obsidian Vault

Clone the repo and open the `vault/` folder as an Obsidian vault.

101 concept notes across 9 topic folders. Each note covers one concept with:
- A structured explanation split into Fundamental / Intermediate / Advanced sections
- `[[wikilinks]]` to related concepts (works with Obsidian graph view)
- Key equations, intuitions, and practical notes

| Folder | Concepts |
|---|---|
| `architectures/` | Attention mechanism, CNNs, ViT, GNN, Flash Attention, U-Net, VLM architectures, open-vocabulary detection, ... |
| `training/` | Backpropagation, Adam, batch normalisation, mixed precision, loss functions, data augmentation, ... |
| `theory/` | Bias-variance, spectral bias, kernel methods, calibration, generalisations bounds, biodiversity ML, explainability, ... |
| `math/` | Linear algebra, SVD, PCA, Fourier transform, matrix calculus, convexity, optimal transport, ... |
| `probability/` | Bayesian inference, MLE, entropy, distributions, hypothesis testing, MCMC/HMC, ... |
| `generative/` | VAEs, GANs, diffusion models (DDPM/DDIM/CFG), normalising flows, autoregressive models |
| `self-supervised/` | CLIP, BLIP, BERT, SimCLR, MoCo, BYOL, knowledge distillation, remote sensing foundation models, ... |
| `adaptation/` | LoRA, QLoRA, RLHF, DPO, instruction tuning, prompt engineering, transfer learning |
| `data/` | Evaluation metrics, feature preprocessing, Earth observation, satellite imagery, multi-source fusion, ... |

---

## License

[CC BY-NC-SA 4.0](LICENSE) — free to use and adapt for non-commercial purposes with attribution.
