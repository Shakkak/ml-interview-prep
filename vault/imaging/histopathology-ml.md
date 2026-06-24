---
title: "Histopathology and Computational Pathology"
tags: [histopathology, whole-slide-image, multiple-instance-learning, stain-normalization, patch-extraction, computational-pathology]
aliases: [WSI, whole slide image, computational pathology, digital pathology, ABMIL, MIL, H&E analysis]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview, sample-preparation-imaging, image-artifacts]
related: [classical-image-analysis-pipeline, clustering, feature-preprocessing, unet, vision-transformer, transfer-learning]
---

## Fundamental

### What Is Histopathology and Why Is It Hard for AI?

Histopathology is the microscopic examination of stained tissue sections to diagnose disease. A pathologist places a thin tissue slice on a glass slide, stains it (typically H&E), and examines it under a microscope. The outcome — cancer grade, tumour subtype, molecular marker prediction — drives treatment decisions.

The AI challenge is threefold:

1. **Scale**: a digitised slide (Whole Slide Image, WSI) is a gigapixel image — typically 100,000 × 100,000 pixels at 40× magnification. No GPU has enough VRAM to process it as a single input.
2. **Label scarcity**: pathologists label at the slide level ("this slide is Grade 3 adenocarcinoma"), not at the pixel level. Pixel-level annotation of a single WSI can take hours.
3. **Staining variability**: the same tissue prepared at different hospitals looks visually different due to variation in dye concentration, fixation, and scanner calibration. A model trained at Hospital A can fail at Hospital B.

### Whole Slide Image (WSI) Structure

A WSI is stored as a **pyramidal multi-resolution image** (formats: SVS, NDPI, MRXS, TIF). Each level of the pyramid is a downsampled version of the full-resolution scan:

| Level | Magnification | Approx. resolution | Pixel size (µm) |
|-------|--------------|-------------------|-----------------|
| 0 (base) | 40× | 100,000 × 100,000 | 0.25 µm/px |
| 1 | 20× | 50,000 × 50,000 | 0.5 µm/px |
| 2 | 10× | 25,000 × 25,000 | 1.0 µm/px |
| 3 | 5× | 12,500 × 12,500 | 2.0 µm/px |
| 4 (thumbnail) | 1.25× | ~3,000 × 3,000 | 8.0 µm/px |

Only the thumbnail level fits comfortably in RAM. All processing must work by extracting small patches from the desired resolution level.

### The Patch-Based Processing Paradigm

Because the full WSI cannot be loaded into GPU memory, the standard approach is:

1. **Detect tissue**: use the thumbnail (level 4) to create a binary mask separating tissue from glass background. Otsu thresholding in the saturation channel of HSV space works reliably — glass appears white/grey (low saturation), tissue appears pink/purple (higher saturation).
2. **Extract patches**: tile the tissue region at the desired magnification level into non-overlapping (or slightly overlapping) patches, typically 256 × 256 or 512 × 512 pixels. Only patches that are ≥ 50–80% tissue are kept.
3. **Encode patches**: pass each patch through a pretrained CNN or vision transformer to get a feature vector (e.g. 1024-dim). This converts the WSI into a **bag of feature vectors**.
4. **Aggregate**: combine the patch-level feature vectors into a single slide-level representation using an aggregation method (pooling, attention, transformer).
5. **Predict**: pass the slide-level representation through a classifier or regression head.

---

## Intermediate

### Stain Normalisation

H&E staining varies across hospitals, scanner brands, and even staining batches within the same lab. A model trained on one hospital's slides will see a distribution shift when deployed elsewhere.

**Macenko normalisation** (most common):
1. Convert the image from RGB to optical density (OD) space: $\text{OD} = -\log(I / I_0)$ where $I_0 = 255$.
2. Find the two stain directions (haematoxylin, eosin) by SVD on the OD vectors of pixels above an OD threshold.
3. Project each pixel's OD onto the two stain directions to get stain concentrations.
4. Rescale the concentrations to match those of a reference (target) image.
5. Reconstruct the normalised image.

**Reinhard normalisation** (simpler): transfer the mean and standard deviation of the LAB colour channels from a target image to the source image. Less accurate than Macenko but computationally trivial.

**Deep learning normalisation** (StainGAN, CycleGAN-based): learn an image-to-image translation from one staining distribution to another. More flexible; handles non-linear staining differences; requires paired or unpaired training data.

**Stain augmentation** (training-time alternative): instead of normalising to one target, randomly perturb H&E concentrations during training — shift haematoxylin/eosin colour and concentration stochastically. The model learns to be invariant to staining, not just normalised to one reference. More robust than normalisation for multi-site deployment.

**AI implication**: stain normalisation and augmentation are not mutually exclusive. A common recipe: normalise at inference time to a representative reference slide from the deployment site; augment during training with random stain perturbations.

### Magnification and Scale Selection

Different pathological features are visible at different magnifications:

| Magnification | What is visible | Typical tasks |
|--------------|-----------------|---------------|
| 40× (0.25 µm/px) | Subcellular detail: nuclei, mitoses, cell membranes | Mitosis detection, cell-level grading |
| 20× (0.5 µm/px) | Cell-level structure: glandular architecture, nuclear atypia | Gleason grading (prostate), tumour grade |
| 10× (1 µm/px) | Tissue architecture: gland arrangement, stromal patterns | Tumour subtype, invasion pattern |
| 5× (2 µm/px) | Overall tissue organisation: tumour vs. stroma proportion | Tumour budding, spatial layout |

**Multi-scale approaches**: important features exist at multiple scales simultaneously. A gland may need 10× to see its arrangement but 40× to detect nuclear atypia inside it. Multi-scale models process each patch at two or three magnification levels and fuse the representations.

### Multiple Instance Learning (MIL)

**Problem**: slide-level labels are available (tumour present/absent, Grade 1/2/3), but pixel-level or patch-level labels are not. This is the standard supervision setting in digital pathology.

**MIL formulation**: each WSI is a **bag** $\mathbf{B} = \{x_1, x_2, \ldots, x_N\}$ of $N$ patch feature vectors. A bag-level label $Y$ is observed; individual instance labels $y_i$ are unknown.

**Standard MIL assumption**: a bag is positive ($Y = 1$) if and only if it contains at least one positive instance. A bag is negative ($Y = 0$) only if all instances are negative. For cancer detection: a tumour slide contains at least one tumour patch; a normal slide contains no tumour patches.

**Max-pooling MIL** (simplest): $\hat{Y} = \max_i f(x_i)$ — the slide score is the maximum patch score. Extremes drive the prediction. Ignores context and the proportion of tumour.

**Mean-pooling MIL**: $\hat{Y} = \frac{1}{N} \sum_i f(x_i)$ — global average of patch scores. Accounts for tumour burden but is confused by a few outlier patches.

### Attention-Based MIL (ABMIL)

ABMIL (Ilse et al., 2018) learns a weight $a_i$ for each patch and computes a weighted average:

$$z = \sum_{i=1}^N a_i \mathbf{h}_i, \qquad a_i = \frac{\exp\left(\mathbf{w}^\top \tanh(\mathbf{V} \mathbf{h}_i)\right)}{\sum_j \exp\left(\mathbf{w}^\top \tanh(\mathbf{V} \mathbf{h}_j)\right)}$$

where $\mathbf{h}_i = f(x_i)$ is the feature embedding of patch $i$ (from a pretrained CNN), $\mathbf{V}$ is a projection matrix, and $\mathbf{w}$ is a learned attention vector. The slide representation $z$ is then fed to a classifier.

**What the attention weight represents**: $a_i$ is proportional to how much patch $i$ contributes to the slide-level prediction. High-attention patches are the most informative — in tumour classification, they tend to correspond to tumour regions. Attention maps are a form of **weakly supervised localisation**: without any pixel-level labels, the model highlights the diagnostically relevant regions.

**AI implications**:
- Attention weights are slide-relative — compare attention within one WSI, not across WSIs.
- Gated attention (ABMIL uses a tanh gate multiplied by a sigmoid gate) outperforms plain attention for multi-class problems.
- ABMIL assumes **instance independence**: patches are processed independently, ignoring spatial context. Spatial context (which patches are adjacent) is lost.

### TransMIL and Spatial Context

**TransMIL** (Shao et al., 2021) replaces the attention pooling of ABMIL with a transformer encoder over the patch sequence. Positional encodings encode the spatial coordinates of each patch. Self-attention between patches captures neighbourhood context — tumour patches adjacent to stroma look different from isolated tumour patches.

Trade-off: transformer over $N$ patches has $O(N^2)$ attention complexity. A typical WSI has $N = 10{,}000$–$100{,}000$ patches — quadratic attention is infeasible. Solutions: (a) sample a random subset of patches per slide during training ($N \leq 4{,}000$); (b) use linear attention; (c) use HIPT (hierarchical pretraining) which processes 256 × 256 patches, then 4096 × 4096 regions, then the full slide.

### Patch Sampling Strategy and Class Imbalance

In a tumour-positive slide, typically only 5–30% of patches contain tumour. The rest are stroma, fat, necrosis, or normal tissue. This creates extreme class imbalance at the patch level.

**Consequences for MIL**:
- Max-pooling MIL is less affected — it finds the highest-scoring patch regardless.
- Mean-pooling MIL is heavily diluted — 95% normal patches pull the mean toward 0.
- ABMIL mitigates this by up-weighting tumour patches, but only if the attention mechanism is strong enough to find them in thousands of patches.

**Sampling strategies**:
- **Random sampling**: sample $K$ patches uniformly per slide at each training epoch. Different subsets are seen each epoch (data augmentation effect). Simple but may miss rare tumour patches.
- **Tissue-stratified sampling**: ensure each sampled batch contains patches from different tissue regions (tumour, stroma, necrosis) using coarse region estimates from low magnification.
- **Hard negative mining**: identify normal patches the model incorrectly scores highly and over-sample them in the next epoch.
- **CLAM** (Chen et al., 2021): Clustering-constrained Attention MIL — uses instance-level clustering to encourage the attention mechanism to discover semantically distinct tissue types, providing additional supervision signal beyond the slide label.

---

## Advanced

### Pathology Foundation Models

Foundation models pretrained on large pathology corpora have largely replaced ImageNet-pretrained CNNs as patch encoders:

| Model | Pretraining data | Architecture | Pretraining method |
|-------|-----------------|--------------|-------------------|
| **UNI** (Chen et al., 2024) | 100k+ WSIs, 20+ tissue types | ViT-L | DINOv2 self-supervised |
| **CONCH** (Lu et al., 2024) | 1.17M pathology image-caption pairs | ViT + text encoder | CLIP-style contrastive |
| **Gigapath** (Xu et al., 2024) | 170k WSIs (Providence Health) | ViT-g | DINOv2, slide-level pretraining |
| **PLIP** (Huang et al., 2023) | Twitter pathology image-text pairs | ViT + text encoder | CLIP-style contrastive |
| **HIPT** (Chen et al., 2022) | TCGA 10k WSIs | Hierarchical ViT | DINO at patch + region level |

**Why pathology-specific pretraining matters**: ImageNet images (natural scenes, objects) have very different statistics from H&E images (periodic texture, colour range limited to pink/purple/blue, no natural backgrounds). Features from ImageNet-pretrained models capture texture statistics irrelevant to pathology. Pathology-pretrained models encode diagnostically meaningful features: nuclear shape, gland architecture, stromal density.

**Usage pattern**: freeze the foundation model encoder; extract 1024-dim patch embeddings; train only the MIL aggregation head. With < 1,000 labelled slides, this outperforms end-to-end fine-tuning of a CNN.

### Survival Analysis from WSIs

Predicting patient survival (time-to-event) from histology is a common task. The standard model is **Cox proportional hazards**:

$$h(t | \mathbf{z}) = h_0(t) \exp(\mathbf{w}^\top \mathbf{z})$$

where $h_0(t)$ is the baseline hazard, $\mathbf{z}$ is the slide representation (from ABMIL or a foundation model), and $\mathbf{w}$ are learned weights. The model is trained with the **partial likelihood** loss, which handles censored observations (patients still alive at follow-up).

**DeepSurv** replaces $\mathbf{w}^\top \mathbf{z}$ with a neural network; **SurvPath** uses attention-based MIL with a survival objective directly.

**Evaluation**: Harrell's concordance index (C-index) — probability that the model ranks two patients correctly by risk. C-index = 0.5 is random; C-index = 1.0 is perfect. Kaplan–Meier curves stratified by predicted risk quartile visualise group separation.

### Multi-Modal Pathology

Modern clinical models integrate WSI features with molecular data (genomics, transcriptomics, proteomics) and clinical variables (age, stage):

- **PORPOISE** (Chen et al., 2022): concatenates WSI-derived MIL features with genomic mutation vectors.
- **MCAT** (Chen et al., 2021): cross-attention between genomic tokens and WSI patch tokens — genomic features selectively attend to relevant tissue regions.

**AI implication**: the bottleneck is aligning different data modalities that have very different formats (gigapixel images vs. sparse mutation vectors vs. survival times). Missing modalities at test time (not all patients have RNA-seq) require imputation or modality-dropout training.

---

## Links

- [[imaging-modalities-overview]] — histopathology uses optical brightfield and fluorescence microscopy; WSI is a digitised brightfield slide
- [[sample-preparation-imaging]] — H&E staining, fixation artefacts, and batch effects are the main sources of domain shift in histopathology
- [[image-artifacts]] — tissue folds, air bubbles, and out-of-focus regions are common WSI artefacts that must be detected and excluded
- [[classical-image-analysis-pipeline]] — the classical cell segmentation pipeline applies at the patch level within a WSI
- [[image-quality-metrics]] — scanner focus, compression, and magnification calibration determine the effective resolution of a WSI
- [[clustering]] — unsupervised clustering of patch embeddings is used for tissue phenotyping and MIL initialisation
- [[feature-preprocessing]] — patch feature normalisation is required before ABMIL aggregation
- [[unet]] — U-Net variants are used for nuclei segmentation and gland segmentation within WSI patches
- [[vision-transformer]] — ViT-based encoders (DINOv2, UNI, CONCH) dominate patch feature extraction
- [[transfer-learning]] — pathology foundation models are used via feature extraction (frozen encoder), not full fine-tuning
