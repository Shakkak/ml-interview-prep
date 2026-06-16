---
title: Pooling Operations
tags: [cnn, pooling, global-pooling, spatial-pyramid, architecture]
aliases: [max pooling, average pooling, global average pooling, GAP, spatial pyramid pooling, adaptive pooling]
difficulty: 1
status: complete
depends_on: [convolution-math, linear-algebra-fundamentals]
related: [cnn-architectures-guide, convolution-math, arch-roi-align, feature-pyramid-networks, attention-mechanism, vision-transformer]
---

# Pooling Operations

---

## Fundamental

### What Pooling Does

Pooling reduces the spatial dimensions of a feature map by summarizing a local region with a single value. For a $2 \times 2$ window with stride 2 on a $4 \times 4$ map:

```
Input (4×4):        Max Pool 2×2:    Avg Pool 2×2:
1  3  2  4          3  4             2.25  3.5
5  6  1  2    →     6  4      →      3.5   2.75
7  1  4  3
2  8  3  6
```

**Three reasons to pool:**
1. **Reduce spatial size** → fewer parameters in subsequent layers, less memory
2. **Increase receptive field** → later layers see larger regions of the original image
3. **Translation invariance** → small shifts in input produce the same output (max pool is invariant to shifts within the window)

### Max Pooling

$$y_{i,j} = \max_{(p,q) \in \text{window}} x_{i \cdot s + p,\, j \cdot s + q}$$

For a $2 \times 2$ pool with stride 2: the output is half the spatial size in each dimension.

**Gradient:** zero for all non-maximum elements in the window; 1 for the maximum. This "winner-take-all" behavior means gradients only flow through the most activated position. Used almost exclusively in CNNs (ResNets, VGGs, classic architectures).

**Intuition:** selects the most activated response in each window. Good for detecting whether a feature is present anywhere in the window, regardless of exact position.

### Average Pooling

$$y_{i,j} = \frac{1}{|W|} \sum_{(p,q) \in W} x_{i \cdot s + p,\, j \cdot s + q}$$

**Gradient:** $1/|W|$ equally distributed to all positions in the window (uniform).

**Intuition:** averages the response over the window. Captures the overall presence of a feature, not just the peak. Produces smoother feature maps and more gradual gradients.

**Max vs Average:**

| Property | Max Pool | Average Pool |
|----------|----------|--------------|
| Selects | Peak activation | Mean activation |
| Translation invariance | Stronger (exact within window) | Weaker (smoother) |
| Gradient flow | Sparse (winner takes all) | Dense (even spread) |
| Use case | Feature presence detection | Background/texture smoothing |
| Standard in | Early CNNs, inception blocks | Global pooling head, diffusion |

---

## Intermediate

### Global Average Pooling (GAP)

Global average pooling reduces each channel's entire feature map to a single scalar by averaging all spatial positions:

$$\text{GAP}(X_c) = \frac{1}{H \cdot W} \sum_{h=1}^H \sum_{w=1}^W X_{c,h,w}$$

For an input of shape $C \times H \times W$, the output is a $C$-dimensional vector.

**Why GAP replaced FC layers for classification:** Lin et al. (2013) introduced GAP as a replacement for the massive fully-connected head at the end of CNNs. The original AlexNet used two FC layers with $4096$ units each — about 26M parameters (roughly half the model's total). GAP reduces the $C \times H \times W$ map directly to $C$ features, which are then classified with a single FC layer. ResNets, EfficientNets, and most modern CNNs use GAP.

**Benefits of GAP:**
1. No additional parameters (unlike FC) — no overfitting from the classification head
2. Works for any spatial resolution — the classification head doesn't depend on input image size
3. Natural interpretability: each channel can be thought of as a "concept detector," and its average activation is the concept strength

**Connection to class activation maps (CAM):** since GAP averages activation across all positions before classification, the classification weights can be projected back onto the spatial map to show "where" the network is looking for each class. $\text{CAM}_c(h, w) = \sum_k w_k^c X_{k,h,w}$ highlights the most discriminative regions.

### Adaptive Pooling

PyTorch's `AdaptiveAvgPool2d(output_size)` automatically computes the window size and stride to produce exactly `output_size` output, regardless of the input size:

```python
# Input: any spatial resolution → Output: always 7×7
pool = nn.AdaptiveAvgPool2d((7, 7))
```

**Why adaptive pooling is useful:** pretrained models need to process images at different resolutions. With adaptive pooling, the feature extraction backbone can handle variable input sizes, and only the pooling layer adjusts — no reshaping or resizing needed. This is how multi-scale inference works in detection and segmentation models.

### Spatial Pyramid Pooling (SPP)

SPP (He et al., 2015) pools the feature map at multiple scales simultaneously and concatenates the results:

```
Feature map (C×H×W)
   → GAP:        C×1
   → Pool 2×2:   C×4
   → Pool 4×4:   C×16
   → Concatenate: C×21 descriptor
```

**Benefit:** a fixed-size representation regardless of input resolution. Before SPP, CNNs required fixed-size input images. SPP allows a single model to handle images of any size at test time, enabling better multi-scale accuracy.

**SPP in detection:** Faster R-CNN uses RoI Pooling (a spatial-adaptive pooling variant) to extract fixed-size features from arbitrary RoI windows. SPPNet was the precursor that showed fixed-size extraction from variable-size regions is possible.

---

## Advanced

### Attention as Soft Pooling

Standard pooling uses fixed windows with uniform or max weights. Attention can be viewed as **learned, adaptive pooling** where the "weights" are computed dynamically based on the content:

$$\text{AttnPool}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

The attention weights $\text{softmax}(QK^\top / \sqrt{d})$ are non-uniform and input-dependent — effectively a learned spatial pooling that focuses on the most relevant positions. This is why attention-based models (ViT) have largely replaced conv+pooling hierarchies for high-level vision tasks: the aggregation is more flexible.

**ViT's CLS token as global pooling:** the `[CLS]` token in Vision Transformer (see [[vision-transformer]]) acts as a learned global pooling mechanism. After $L$ layers of self-attention, the CLS token's output embedding represents a global summary of the entire image — but with learned, dynamic, content-adaptive aggregation weights rather than uniform averaging.

### Stochastic Depth and Pooling as Regularization

Stochastic pooling (Zeiler & Fergus, 2013) selects pool winners stochastically, proportional to activation magnitude, rather than always taking the maximum:

$$P(\text{pick position }(p,q)) = \frac{x_{p,q}}{\sum_{(p',q')} x_{p',q'}}$$

This injects noise into the spatial summary, acting as a regularizer similar to dropout on feature maps. At inference, standard average pooling is used. Reported modest improvements over deterministic pooling in early CNNs; largely superseded by Dropout and BatchNorm as general regularizers.

### Pooling and the Spatial Hierarchy

The interaction between convolution stride/pooling and receptive field creates the spatial hierarchy that makes CNNs powerful for multi-scale tasks:

**ResNet-50 spatial downsampling schedule:**
- Input: $224 \times 224$
- After stem (stride 2 conv + MaxPool): $56 \times 56$ (4× downsampled)
- After stage 2 (stride 2): $28 \times 28$ (8× downsampled)
- After stage 3 (stride 2): $14 \times 14$ (16× downsampled)
- After stage 4 (stride 2): $7 \times 7$ (32× downsampled)
- After GAP: $1 \times 1$ (global summary)

Each $2\times$ downsampling doubles the effective receptive field and halves the computational cost of subsequent layers. The 4× downsampling in the stem via a $7 \times 7$ stride-2 conv followed by $3 \times 3$ max-pool stride-2 is what allows efficient processing of $224 \times 224$ images — without this, the first convolutional layers would be prohibitively expensive on full-resolution feature maps.

**Why detection heads need multi-scale features:** a $7 \times 7$ feature map (32× downsampled from $224 \times 224$) can detect objects that span large image regions but cannot localize small objects. Feature Pyramid Networks (see [[feature-pyramid-networks]]) reintroduce multi-scale features by laterally connecting downsampling-path features to an upsampling-path output — giving high-level semantics at all scales while maintaining spatial resolution for small object detection.

---

## Links

- [[convolution-math]] — pooling is a fixed (non-learned) spatial aggregation operation; it reduces spatial resolution without learnable parameters
- [[linear-algebra-fundamentals]] — global average pooling computes the mean over a spatial grid, equivalent to a dot product with the all-ones vector normalized by count
- [[cnn-architectures-guide]] — max pooling for spatial downsampling and global average pooling for the final feature vector are the two canonical uses in CNN architectures
- [[arch-roi-align]] — ROI Align is a differentiable, position-aware pooling that extracts fixed-size features from arbitrary-scale regions; it superseded ROI pooling
- [[feature-pyramid-networks]] — SPP (Spatial Pyramid Pooling) pools at multiple scales to produce multi-scale features without a full FPN top-down pathway
- [[vision-transformer]] — ViT has no pooling layers; the class token is the global summary instead; patch embeddings maintain fixed resolution throughout
- [[attention-mechanism]] — global attention can be seen as learned soft pooling; each output is a weighted average of all input values
