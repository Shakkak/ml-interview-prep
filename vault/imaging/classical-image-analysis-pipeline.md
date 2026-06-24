---
title: "Classical Image Analysis Pipeline"
tags: [image-analysis, segmentation, feature-extraction, blob-detection, morphological-operations, watershed]
aliases: [classical image pipeline, particle analysis pipeline, cell analysis pipeline, image processing pipeline]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview, image-quality-metrics, image-artifacts]
related: [sampling-aliasing, clustering, feature-preprocessing, unet, sample-preparation-imaging]
---

## Fundamental

### What Problem Does the Classical Pipeline Solve?

You have a grayscale image containing many objects of interest — cells, particles, bubbles, nuclei. Each object has a shape, a size, and an intensity. The goal is to: (1) find every object in the image, (2) separate touching objects from each other, (3) measure properties of each object, and (4) classify or cluster objects by those properties.

The classical pipeline solves this without any labelled training data. It is the correct first approach when:
- The number of training examples is small
- The object appearance is simple enough to model by hand (circular cells, bright spots on dark background)
- Interpretability of each step is required
- Results from the pipeline serve as ground truth for training a deep learning model

### The Six-Stage Pipeline

```
Raw image
   ↓
1. Preprocessing      — noise reduction, contrast enhancement
   ↓
2. Segmentation       — separate objects from background
   ↓
3. Separation         — split touching/overlapping objects
   ↓
4. Labelling          — assign a unique ID to each object
   ↓
5. Feature extraction — measure properties of each object
   ↓
6. Classification / Clustering — group or label objects
```

Each stage has a narrow, verifiable output. A failure at any stage propagates forward — a missed cell at stage 2 cannot be recovered at stage 5.

---

## Intermediate

### Stage 1 — Preprocessing

**Goal:** reduce noise and enhance contrast without moving object boundaries.

**Gaussian blur** ($\sigma$ = 1–2 px): reduces high-frequency quantum noise while preserving large-scale object structure. Use when noise is approximately white (Poisson/Gaussian).

**Median filter** (3×3 or 5×5 kernel): removes salt-and-pepper noise without blurring edges. Use when noise appears as isolated bright/dark pixels (dead pixels, cosmic rays in microscopy).

**CLAHE** (Contrast-Limited Adaptive Histogram Equalisation): compensates for non-uniform illumination by equalising local histograms in small tiles. Essential when the image is brighter in the centre than at the edges (Gaussian lamp profile). Prevents bright background regions from being mistakenly thresholded as objects.

**Decision rule:** apply CLAHE first if the image has illumination gradients. Then apply Gaussian or median blur depending on noise type.

### Stage 2 — Segmentation (Foreground / Background)

**Goal:** produce a binary mask — 1 where objects are, 0 where background is.

**Global Otsu thresholding:** finds the intensity threshold $t^*$ that minimises the weighted within-class variance of the two pixel classes. Optimal when the intensity histogram has a clear bimodal shape (well-separated object and background populations). Fails when illumination is non-uniform.

**Adaptive thresholding:** computes a local threshold for each pixel based on the intensity of its neighbourhood (mean or Gaussian-weighted). Handles non-uniform illumination at the cost of requiring a neighbourhood size parameter larger than the objects of interest.

**Blob detection** (alternative to thresholding for circular objects): detects bright circular regions directly without requiring a global threshold.

- **Laplacian of Gaussian (LoG):** convolve the image with $-\nabla^2 G_\sigma$, producing a strong negative response at the centre of bright blobs of radius $r \approx \sqrt{2}\sigma$. Search for local minima across a range of $\sigma$ values to detect blobs of different sizes. Computationally exact but slow for large $\sigma$.

- **Difference of Gaussians (DoG):** approximate LoG by $G_{\sigma_1} - G_{\sigma_2}$ with $\sigma_2 \approx 1.6\sigma_1$. Faster than LoG; used in SIFT feature detection and cell detection in fluorescence microscopy.

- **Determinant of Hessian (DoH):** uses the determinant of the image Hessian matrix to detect blob-like structures. Scale-invariant; efficient via integral images.

**When to use blob detection vs thresholding:** use thresholding when objects occupy large contiguous regions and the background is clear. Use blob detection (LoG/DoG) when objects are compact (roughly circular), vary in size, and may be close to each other — it directly provides centre coordinates and estimated radius for each blob.

### Stage 3 — Separation of Touching Objects

Thresholding merges touching objects into a single foreground region. Two objects that touch appear as one connected blob.

**Distance transform:** for each foreground pixel, compute its Euclidean distance to the nearest background pixel. Object centres — the pixels farthest from the boundary — become local maxima of the distance transform.

**Watershed on the distance transform:**
1. Compute the distance transform of the binary mask.
2. Invert it (so object centres are local minima).
3. Place one "seed" (water source) at each local minimum.
4. Let the watershed flood from each seed simultaneously.
5. Where two flooding fronts meet, draw a watershed line — this becomes the boundary separating the two objects.

The watershed algorithm is guaranteed to find one connected region per seed. The number of seeds equals the number of detected objects. If the distance transform has too many local minima (due to noise), over-segmentation occurs — one real object gets split into several regions. Smooth the distance transform with a Gaussian before watershed to merge spurious minima.

### Stage 4 — Labelling (Connected Components)

After separation, assign a unique integer label to each connected foreground region. This is **connected component labelling** (also called blob labelling). Standard algorithms (e.g. two-pass algorithm) run in $O(N)$ time where $N$ is the number of pixels.

Output: a label image where each pixel value is the ID of the object it belongs to (0 = background).

### Stage 5 — Feature Extraction per Object

For each labelled region, compute measurements:

**Morphological features:**
- **Area** $A$: number of pixels in the region (proxy for object size)
- **Perimeter** $P$: length of the boundary in pixels
- **Circularity** $C = 4\pi A / P^2 \in [0, 1]$: 1 for a perfect circle; decreases for elongated or irregular shapes
- **Aspect ratio** $r = \text{major axis} / \text{minor axis}$: 1 for a circle; large for elongated objects (bacteria, fibres)
- **Convexity** $= A / A_\text{convex hull}$: ratio of object area to its convex hull area; < 1 for concave objects
- **Solidity** $= A / A_\text{convex hull}$: same as convexity; < 1 indicates non-convex shape (e.g. cell with blebs)
- **Hu moments**: seven rotation-, translation-, and scale-invariant shape moments derived from the image moments of the region. Compact representation of shape for classification.
- **Equivalent diameter** $d = \sqrt{4A/\pi}$: diameter of a circle with the same area

**Intensity features** (measured over the pixels of each region):
- Mean intensity $\bar{I}$: average brightness; proxy for fluorophore concentration in microscopy
- Standard deviation of intensity $\sigma_I$: measures internal intensity variation (uniform vs. grainy)
- Min and max intensity: detect saturation; identify hot pixels
- Integrated intensity $= A \times \bar{I}$: total fluorescence signal; proportional to total protein amount if background-corrected

**Texture features:**
- **GLCM** (Grey-Level Co-occurrence Matrix): describes how often pairs of pixel intensities co-occur at a given offset. From GLCM: contrast, correlation, energy, homogeneity — describe fine-scale texture within the object
- **LBP** (Local Binary Pattern): encodes local texture as a histogram of binary patterns. Computationally cheap; rotation-invariant variants exist

### Stage 6 — Classification and Clustering

Once each object has a feature vector $\mathbf{x} \in \mathbb{R}^d$, standard ML methods apply.

**If labels are available (classification):**
- **SVM with RBF kernel:** good default for small datasets ($N < 10{,}000$ objects). Works well with normalised features.
- **Random Forest:** handles mixed feature types; provides feature importance rankings that help identify which measurements are discriminative.
- **Logistic regression:** interpretable baseline; useful for checking that features are linearly separable.

**If no labels are available (clustering):**
- **k-means:** partitions objects into $k$ clusters by minimising within-cluster variance. Fast, but requires specifying $k$ in advance and assumes spherical clusters in feature space. Use the elbow method or silhouette score to choose $k$.
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise): groups objects that are densely packed in feature space; labels isolated objects as noise. Does not require $k$; discovers clusters of arbitrary shape. Requires two parameters: $\varepsilon$ (neighbourhood radius) and `min_samples` (minimum cluster density). Use when the number of cell types is unknown or when outliers (debris, dead cells) should be excluded.
- **Hierarchical clustering (agglomerative):** builds a dendrogram by iteratively merging the two closest clusters. The dendrogram can be cut at any level. Useful for exploring cluster structure before committing to $k$.

**Feature normalisation before clustering:** all features must be on comparable scales. Standardise: $x_i' = (x_i - \mu_i) / \sigma_i$. Without this, area (thousands of pixels) will dominate circularity (0–1) and clustering will be driven entirely by size.

**Dimensionality reduction before clustering:** if the feature vector has many correlated components (e.g. GLCM generates 4 features, Hu moments 7), apply PCA first to retain 2–5 components that explain ≥ 95% of variance. This removes redundancy, speeds up clustering, and allows visualisation of the cluster structure as a scatter plot.

---

## Advanced

### Failure Modes and Diagnostics

| Stage | Common failure | Diagnostic | Fix |
|-------|---------------|------------|-----|
| Preprocessing | CLAHE creates artefacts at tile boundaries | Inspect tile boundaries at high zoom | Increase overlap between tiles |
| Thresholding | Threshold too low/high due to bimodal asymmetry | Plot intensity histogram; check where threshold falls | Use Otsu per-image; or Gaussian-fit each mode |
| Watershed | Over-segmentation (real cell split into 3) | Inspect distance transform for spurious minima | Smooth distance transform; increase minimum peak distance |
| Watershed | Under-segmentation (two cells merged) | Inspect where watershed lines fall | Verify seeds: one local maximum per cell needed |
| Feature extraction | Boundary effects (objects touching image edge) | Count pixels on boundary | Exclude edge-touching objects |
| Clustering | k-means finds size-based clusters instead of type-based | PCA scatter plot: check if size dominates PC1 | Normalise features; try clustering on circularity + texture only |

### When to Switch to Deep Learning

The classical pipeline degrades when:
- Objects overlap significantly (3D stacks, dense monolayers) — watershed fails
- Object boundaries are ambiguous (low CNR, heterogeneous staining)
- The feature set that discriminates classes is unknown — deep features outperform hand-crafted features
- Dataset is large enough to train a segmentation model (> ~200 annotated images)

The classical pipeline output — segmented objects with feature vectors — is a natural source of weak labels for training deep models: use it to bootstrap annotation or as a teacher for semi-supervised learning.

---

## Links

- [[image-quality-metrics]] — SNR and CNR determine whether thresholding can separate objects from background
- [[image-artifacts]] — noise type (Poisson, read noise, speckle) determines which preprocessing filter to use
- [[imaging-modalities-overview]] — the pipeline adapts to the modality: fluorescence microscopy uses blob detection + CLAHE; brightfield uses different contrast assumptions
- [[sample-preparation-imaging]] — flat-field correction must precede CLAHE; staining protocol determines which channel contains the target signal
- [[sampling-aliasing]] — objects smaller than 2 pixels cannot be reliably detected (Nyquist limit)
- [[feature-preprocessing]] — feature normalisation and PCA before clustering follow the same rules as for tabular ML
- [[clustering]] — k-means, DBSCAN, and hierarchical clustering applied to the extracted feature vectors
- [[unet]] — U-Net and Cellpose replace stages 2–4 when labelled data is available
