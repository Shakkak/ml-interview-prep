---
title: Convolution — Mathematical Foundations
tags: [convolution, signal-processing, cnn, fourier, linear-algebra]
aliases: [discrete convolution, cross-correlation, convolution theorem, convolutional layer math]
difficulty: 2
status: complete
related: [cnn-architectures-guide, fourier-transform, arch-depthwise-separable, arch-bottleneck-1x1, dilated-convolution, backpropagation-advanced, linear-algebra-fundamentals]
---

# Convolution — Mathematical Foundations

---

## Fundamental

### Continuous and Discrete Convolution

**Continuous convolution** of signals $f$ and $g$:

$$(f * g)(t) = \int_{-\infty}^\infty f(\tau) g(t - \tau)\, d\tau$$

Intuition: the output at time $t$ is a weighted sum of all past values of $f$, where the weights are given by $g$ run backwards. The $g(t - \tau)$ "flips and slides" the kernel over the input.

**Discrete convolution** (for sequences):
$$(f * g)[n] = \sum_{k=-\infty}^\infty f[k]\, g[n - k]$$

**2D discrete convolution** (for images):
$$(I * K)[i, j] = \sum_m \sum_n I[m, n]\, K[i-m, j-n]$$

### Cross-Correlation vs Convolution

In deep learning, what is called "convolution" is technically **cross-correlation** — the kernel is NOT flipped:

$$\text{(correlation: }I \star K\text{)} \quad [i, j] = \sum_m \sum_n I[i+m, j+n]\, K[m, n]$$

True convolution would use $K[-m, -n]$ (flipped kernel). For symmetric kernels (e.g., Gaussian blur), the two are identical. For learning: since the kernel is learned anyway, the flip doesn't matter — the network learns the flipped version if that's what's needed. PyTorch, TensorFlow, and all modern frameworks use cross-correlation but call it "convolution."

**Why this matters:** when deriving the backprop gradient of a conv layer, the gradient with respect to the input is a true convolution (with flipped kernel), while the gradient with respect to the kernel is cross-correlation with the input. This is why GPU convolution primitives (cuDNN) expose "forward" and "backward" variants with different flip modes.

### Convolutional Layer: The Full Operation

For input feature map $X \in \mathbb{R}^{C_{in} \times H \times W}$ and kernel $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_H \times K_W}$:

$$Y_{c_{out}, h, w} = \sum_{c_{in}} \sum_{k_h=0}^{K_H-1} \sum_{k_w=0}^{K_W-1} X_{c_{in},\, hs + k_h,\, ws + k_w} \cdot W_{c_{out}, c_{in}, k_h, k_w} + b_{c_{out}}$$

where $s$ is the stride. Each output channel is a 2D correlation of all input channels summed together — the kernel "mixes" spatial neighborhoods AND channels simultaneously.

**Output spatial dimensions:**

$$H_{out} = \left\lfloor\frac{H_{in} + 2p - K_H}{s}\right\rfloor + 1, \qquad W_{out} = \left\lfloor\frac{W_{in} + 2p - K_W}{s}\right\rfloor + 1$$

For $K=3, s=1, p=1$ (the most common config): $H_{out} = H_{in}$ — "same" padding preserves spatial size.

**Parameter count:** $C_{out} \times C_{in} \times K_H \times K_W + C_{out}$ (bias).

---

## Intermediate

### Convolution as Matrix Multiplication (im2col)

Every convolution can be rewritten as a matrix multiplication via **im2col** (image-to-column):

1. Extract all $K_H \times K_W \times C_{in}$-dimensional patches from the input: each patch position $(h, w)$ becomes a column in a matrix $P \in \mathbb{R}^{(K_H K_W C_{in}) \times (H_{out} W_{out})}$
2. Reshape kernels to $W' \in \mathbb{R}^{C_{out} \times (K_H K_W C_{in})}$
3. The output is $Y' = W' P \in \mathbb{R}^{C_{out} \times (H_{out} W_{out})}$, reshaped to $C_{out} \times H_{out} \times W_{out}$

This transforms the conv into a GEMM (general matrix-matrix multiply), allowing BLAS/cuBLAS optimized routines to execute it. The cost: im2col creates a copy of the input data that is $K_H K_W$ times larger — significant memory overhead. For small kernels on large feature maps, this overhead can be larger than the input itself.

**Why GPUs are fast at this:** the im2col GEMM is highly parallelizable and maps perfectly to tensor cores (which compute A×B in a single instruction). Strided convolutions and depthwise convolutions break this pattern and require specialized kernels.

### Receptive Field

The **receptive field** of a neuron is the region of the input that can influence its output. For a single conv layer with kernel $K$: receptive field = $K$.

For a stack of $L$ conv layers each with kernel size $K$ and stride 1:

$$\text{RF}_L = 1 + L(K - 1)$$

**With dilation $d$:** a dilated conv with dilation $d$ and kernel $K$ has effective kernel size $K + (K-1)(d-1) = d(K-1) + 1$. The receptive field grows as $1 + L \cdot d(K-1)$.

**With striding:** stride $s$ at layer $l$ multiplies the receptive field contribution of earlier layers by $s$. The receptive field grows multiplicatively with depth when using stride $> 1$ — this is why pooling (stride 2) expands the receptive field faster than padding+stacking.

**Why receptive field matters:** an object detection model needs a receptive field large enough to cover the entire object. A ResNet-50 at the 4th stage has a receptive field of ~196 pixels — enough for most objects in ImageNet (typical size ~200 px). For large objects in high-resolution images (satellite imagery, medical scans), the receptive field must be explicitly designed to be large enough. [[dilated-convolution|Dilated convolutions]] are the standard fix — exponentially growing dilation (1, 2, 4, 8, 16, ...) achieves exponential receptive field growth without striding or extra parameters.

### Depthwise Separable Convolution (Parameter Reduction)

A standard $K \times K$ convolution on $C$ channels has $K^2 C^2$ parameters (ignoring $C_{out} \neq C$ for simplicity). The **depthwise separable** decomposition splits this:

1. **Depthwise conv:** apply $C$ independent $K \times K$ filters, one per input channel → $K^2 C$ parameters
2. **Pointwise conv:** mix channels with $C \times C$ $1 \times 1$ convolutions → $C^2$ parameters

Total: $K^2 C + C^2 = C(K^2 + C)$. For $K=3, C=256$: standard has $589,824$; separable has $67,840$ — **8.7× fewer parameters**.

**The quality tradeoff:** the separable decomposition assumes spatial filtering (which features to keep where) and channel mixing (which feature combinations matter) can be done independently. In practice this is approximately true, with only a small accuracy drop (MobileNet achieves ~70% ImageNet top-1 vs ~76% for ResNet-50 with ~8× fewer ops). See [[arch-depthwise-separable]].

### Transposed Convolution (Deconvolution)

For upsampling in encoder-decoder architectures (UNet, generative models), transposed convolution inserts zeros between input elements and then convolves:

$$H_{out} = (H_{in} - 1) \cdot s - 2p + K$$

For $K=3, s=2, p=0$: $H_{out} = 2H_{in} - 1$ → nearly doubles spatial size. With $p=1$: $H_{out} = 2H_{in}$.

**Checkerboard artifacts:** the transposed conv is NOT the true inverse of the forward conv (which would require the kernel to perfectly tile). Overlapping kernel footprints create unevenness in the output. **Fix:** use bilinear upsampling followed by a regular $3 \times 3$ conv — smoother and artifact-free (Odena et al., 2016).

---

## Advanced

### The Convolution Theorem and Fast Computation

The convolution theorem: **convolution in the spatial domain = pointwise multiplication in the frequency domain**:

$$(f * g)(t) = \mathcal{F}^{-1}\!\left[\mathcal{F}[f] \cdot \mathcal{F}[g]\right]$$

where $\mathcal{F}$ is the Fourier transform. This enables **FFT-based convolution:**
1. FFT of input: $O(n \log n)$
2. FFT of kernel: $O(k \log k)$
3. Pointwise multiply: $O(n)$
4. Inverse FFT: $O(n \log n)$

Total: $O(n \log n)$ vs direct convolution's $O(nk)$. FFT convolution is faster when $k$ is large. For the small kernels typical in CNNs ($k = 3, 5$), direct convolution is faster due to lower constant factors.

**Practical use:** FFT convolution is used in audio processing (large 1D kernels) and some spectral methods. For computer vision, it's primarily a theoretical tool. [[fourier-transform]] covers the full Fourier theory.

### Winograd Convolution

A more practically useful fast algorithm for small kernels. Winograd's minimal filtering algorithm for computing $F(m, r)$ ($m$ outputs, $r$-tap filter) reduces multiplications from $m \times r$ to $m + r - 1$.

For $F(2, 3)$ (2 outputs, 3-tap filter):
- Direct: $2 \times 3 = 6$ multiplications
- Winograd: $2 + 3 - 1 = 4$ multiplications

**2D Winograd** for $3 \times 3$ convolution: $F(2 \times 2, 3 \times 3)$ needs $4 \times 4 = 16$ multiplications instead of $4 \times 9 = 36$ — a **2.25× speedup** in multiply-accumulate operations. cuDNN uses Winograd for $3 \times 3$ convolutions by default. The tradeoff: requires additional additions (which are cheap) and some restructuring of the computation.

### Backpropagation Through Convolution

For a conv layer, the gradient of the loss with respect to the input is a full convolution (with flipped kernel):

$$\frac{\partial L}{\partial X_{c_{in}, i, j}} = \sum_{c_{out}} \sum_{k_h}\sum_{k_w} \frac{\partial L}{\partial Y_{c_{out}, h', w'}} \cdot W_{c_{out}, c_{in}, k_h, k_w}$$

where $(h', w')$ are the output positions that receive input from $(i, j)$. This is a "dilated transposed convolution" if the forward pass used stride > 1 — the sparse gradient map is first expanded by inserting zeros (transposing the stride), then convolved with the flipped kernel.

**Gradient with respect to weights:**

$$\frac{\partial L}{\partial W_{c_{out}, c_{in}, k_h, k_w}} = \sum_{h, w} \frac{\partial L}{\partial Y_{c_{out}, h, w}} \cdot X_{c_{in},\, hs + k_h,\, ws + k_w}$$

This is a correlation of the input with the upstream gradient — itself a convolution. **Backprop through a conv layer is a conv operation** — the same hardware primitives handle both forward and backward passes, which is why conv backprop is efficient on GPUs.

### Convolution, Toeplitz Matrices, and Translation Equivariance

A 1D convolution with kernel $w$ is equivalent to multiplication by a **Toeplitz matrix** — a matrix where each diagonal has the same value:

$$\begin{pmatrix} y_0 \\ y_1 \\ y_2 \\ y_3 \end{pmatrix} = \begin{pmatrix} w_0 & 0 & 0 & 0 \\ w_1 & w_0 & 0 & 0 \\ w_2 & w_1 & w_0 & 0 \\ 0 & w_2 & w_1 & w_0 \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{pmatrix}$$

**Translation equivariance:** if the input is shifted by $k$ positions, the output shifts by $k$ positions. Formally: $\text{Conv}(T_k x) = T_k \text{Conv}(x)$ where $T_k$ is the translation operator. This is a direct consequence of the Toeplitz structure — shifting the input is equivalent to shifting the column of the Toeplitz matrix.

**Why this matters:** translation equivariance means CNNs share features across spatial locations. A face detector trained on one image location automatically applies to all locations — the learned filter is reused everywhere. This inductive bias is what makes CNNs so effective for vision tasks and so much more parameter-efficient than fully connected networks for spatial data.

**Limitation — global pooling breaks equivariance:** after global average pooling (used in classification heads), translation equivariance is traded for translation invariance — the output is the same regardless of where the feature is. This is why CNNs are good at "what is in the image" (classification) but need special handling for "where is it" (detection) — see [[feature-pyramid-networks]] and [[arch-roi-align]].

---

*See also: [[cnn-architectures-guide]] · [[fourier-transform]] · [[arch-depthwise-separable]] · [[dilated-convolution]] · [[backpropagation-advanced]] · [[arch-bottleneck-1x1]]*
