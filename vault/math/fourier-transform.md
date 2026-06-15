---
title: Fourier Transform for Computer Vision
tags: [signal-processing, frequency-domain, image-processing, mathematics, fft, convolution-theorem]
aliases: [DFT, FFT, frequency analysis, 2D Fourier, convolution theorem, power spectrum, Gabor filter]
difficulty: 1
status: complete
related: [attention-mechanism, backpropagation-advanced, spectral-bias, dilated-convolution, feature-preprocessing]
---

# Fourier Transform for Computer Vision

---

## Fundamental

The Fourier transform decomposes any signal — sound, an image, a time series — into its constituent frequencies, like a prism splitting white light into colors. A low frequency means slow, gradual variation; a high frequency means rapid oscillation. The transform tells you *how much* of each frequency is present. This is useful because many operations (blurring, sharpening, audio filtering) are simple multiplications in frequency space but expensive convolutions in the original space.

**The key mathematical building block — Euler's formula:**

$$e^{i\theta} = \cos\theta + i\sin\theta$$

where $i = \sqrt{-1}$ is the imaginary unit. This means a complex exponential $e^{i\theta}$ is simply a sinusoid — it oscillates as $\theta$ increases. The DFT formula uses this to represent each frequency component as a rotating sinusoid.

---

### 1D DFT

Given a discrete signal $f[n]$ with $N$ samples (e.g., $N$ pixels of one row of an image, or $N$ audio samples):

$$F[k] = \sum_{n=0}^{N-1} f[n]\, e^{-2\pi ikn/N}$$

**What each symbol means:**
- $n$ — index of the input sample (runs from $0$ to $N-1$)
- $N$ — total number of samples
- $k$ — index of the frequency bin (runs from $0$ to $N-1$); frequency $k$ represents $k/N$ cycles per sample
- $i$ — imaginary unit ($\sqrt{-1}$)
- $e^{-2\pi ikn/N}$ — a sinusoid at frequency $k$, evaluated at position $n$ (using Euler's formula above)
- $F[k]$ — a **complex number** encoding two things: how much of frequency $k$ is present ($|F[k]|$ = amplitude/energy) and where in the cycle the wave starts ($\angle F[k]$ = phase)

Intuitively: $F[k]$ is computed by multiplying the signal by a sinusoid at frequency $k$ and summing the result. If the signal oscillates at that frequency, the products reinforce and $|F[k]|$ is large. If it doesn't, they cancel and $|F[k]| \approx 0$.

**Inverse DFT** — reconstructs the original signal from its frequencies:

$$f[n] = \frac{1}{N}\sum_{k=0}^{N-1} F[k]\, e^{2\pi ikn/N}$$

This expresses $f[n]$ as a weighted sum of sinusoids at all frequencies $k$. The weights are $F[k]/N$.

**Parseval's theorem:** $\sum_n |f[n]|^2 = \frac{1}{N}\sum_k |F[k]|^2$

Total energy in the signal equals total energy in its frequency representation. The DFT is an energy-preserving change of basis — no information is lost.

---

### Worked Example

$f = [1, 2, 3, 4]$, so $N = 4$. We compute $F[k]$ for each $k$.

**$F[0]$** (zero frequency = DC component = average level of the signal):
$$F[0] = 1 + 2 + 3 + 4 = 10$$
$k=0$ means $e^{0} = 1$, so we just sum all samples. The DC component is always the sum of the signal.

**$F[1]$** (one full cycle across the signal):

Using Euler's formula with $\theta = -2\pi n / 4$:
- $n=0$: $e^0 = 1$
- $n=1$: $e^{-i\pi/2} = \cos(-90°) + i\sin(-90°) = 0 - i = -i$
- $n=2$: $e^{-i\pi} = \cos(-180°) = -1$
- $n=3$: $e^{-3i\pi/2} = \cos(-270°) + i\sin(-270°) = 0 + i = +i$

$$F[1] = 1\cdot1 + 2\cdot(-i) + 3\cdot(-1) + 4\cdot(i) = (1 - 3) + (-2 + 4)i = -2 + 2i$$

$|F[1]| = \sqrt{4 + 4} = 2\sqrt{2}$ — moderate energy at this frequency.

**$F[2]$** (two full cycles = Nyquist frequency, the fastest oscillation representable):
$$F[2] = 1\cdot1 + 2\cdot(-1) + 3\cdot1 + 4\cdot(-1) = 1 - 2 + 3 - 4 = -2$$

The alternating $+1, -1, +1, -1$ pattern means $k=2$ detects the fastest possible oscillation (sign alternates every sample).

**$F[3]$** = $\overline{F[1]} = -2 - 2i$

For real-valued signals, $F[N-k] = \overline{F[k]}$ (complex conjugate symmetry). This means the second half of the DFT output carries no new information — it mirrors the first half.

**Reading the result:** $F[0] = 10$ (large DC), $|F[1]| \approx 2.8$, $|F[2]| = 2$, $|F[3]| = 2.8$ — the signal [1,2,3,4] has a strong DC offset (its values are all positive and rising) and moderate low-frequency content.

---

### The Convolution Theorem

$$h = f * g \quad \Longleftrightarrow \quad H[k] = F[k] \cdot G[k]$$

**Convolution in the signal domain = pointwise multiplication in the frequency domain.**

Why this matters: a convolution of $f$ and $g$ naively requires $O(N^2)$ operations (each output sample uses all $N$ input samples weighted by the kernel). In the frequency domain, the same result requires only $O(N)$ multiplications (multiply each $F[k]$ by $G[k]$), plus two FFTs at $O(N \log N)$ each. For large signals or kernels, this is dramatically faster.

Why it's true intuitively: each frequency component in $h$ comes purely from the same frequency in $f$ scaled by the same frequency in $g$ — frequencies don't mix under convolution. A blur kernel suppresses high frequencies; its frequency representation has small $G[k]$ for large $k$, so $H[k] = F[k] \cdot G[k]$ automatically attenuates those frequencies.

Large blurs (wide Gaussian kernel, motion blur) are always computed via FFT in practice because the kernel can be hundreds of pixels wide.

---

## Intermediate

### 2D DFT and Spatial Frequencies in Images

In an image, **spatial frequency** means how rapidly pixel values change across the image:
- **Low spatial frequency** = slow variation = large blobs, gradual gradients, background color
- **High spatial frequency** = rapid variation = edges, fine texture, sharp details

The 2D DFT of an image $f[m, n]$ of size $M \times N$ pixels:

$$F[k, l] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} f[m,n]\, e^{-2\pi i(km/M + ln/N)}$$

**What each symbol means:**
- $m, n$ — row and column indices of the input pixel
- $k, l$ — row and column frequency indices of the output; $(k, l)$ represents a spatial frequency with $k$ cycles vertically and $l$ cycles horizontally
- $F[k, l]$ — complex amplitude at that frequency; $|F[k,l]|^2$ is the power (energy) at that frequency

**Reading the 2D frequency spectrum:** after `fftshift` (which rearranges the output so the zero-frequency is at the center instead of the corner):
- **Center of the image** = DC = average brightness
- **Distance from center** = spatial frequency (farther = higher frequency = finer detail)
- **Direction from center** = orientation; vertical edges create horizontal streaks in the spectrum, horizontal edges create vertical streaks (perpendicular to the edge direction)

**The $1/f^2$ power law of natural images:**

Power spectral density $P(f) \propto 1/f^2$. Low frequencies carry far more energy than high frequencies in photographs of natural scenes. Consequences:
- A random white-noise image has a flat spectrum ($P \propto f^0$) — every frequency has equal power, which looks nothing like a photo
- JPEG compression works by discarding high-frequency DCT coefficients, which are near-zero for natural images
- CNNs with local filters efficiently capture most image information — their receptive fields are matched to where image energy concentrates

---

### Frequency-Domain Filtering

A filter in frequency space multiplies $F[k,l]$ by a mask $H[k,l]$, then transforms back. The spatial-domain equivalent is convolution with the filter's inverse DFT.

| Filter | Frequency action | Spatial effect |
|--------|-----------------|----------------|
| Gaussian (low-pass) | Attenuates high frequencies | Blur / smooth |
| High-pass | Removes DC and low frequencies | Sharpen, emphasize edges |
| Band-pass | Keeps a ring of frequencies | Gabor filter response |
| Notch | Zeros out one specific frequency | Removes periodic noise |

**Spatial-frequency duality (uncertainty principle):**

A Gaussian blur $g(x) = e^{-x^2/2\sigma^2}$ has a Gaussian DFT: $G(f) \propto e^{-2\pi^2\sigma^2 f^2}$.

- Wider spatial Gaussian (larger $\sigma$) → narrower frequency Gaussian → stronger low-pass (removes more high frequencies)
- Narrower spatial Gaussian → wider frequency response → weaker filtering

This is a fundamental tradeoff: you cannot simultaneously localize a filter narrowly in both space and frequency. A filter that is sharp in space (few pixels wide) must be broad in frequency (affects many frequency components), and vice versa.

---

### The Fast Fourier Transform (FFT)

Naive DFT: $O(N^2)$ — for $N = 4096$, that is $\sim 16$ million multiplications.

**Cooley-Tukey FFT (1965):** achieves $O(N \log N)$ by divide and conquer. The key insight: the DFT of an $N$-point signal can be computed from the DFTs of its even-indexed and odd-indexed halves:

$$F[k] = \underbrace{\sum_{n=0}^{N/2-1} f[2n]\, e^{-2\pi ikn/(N/2)}}_{E[k] \text{ (even half)}} + e^{-2\pi ik/N} \underbrace{\sum_{n=0}^{N/2-1} f[2n+1]\, e^{-2\pi ikn/(N/2)}}_{O[k] \text{ (odd half)}}$$

**What each symbol means:**
- $E[k]$ — DFT of the even-indexed samples $f[0], f[2], f[4], \ldots$
- $O[k]$ — DFT of the odd-indexed samples $f[1], f[3], f[5], \ldots$
- $e^{-2\pi ik/N}$ — a "twiddle factor": a phase rotation that adjusts the odd half for its offset position; often written $W^k$ where $W = e^{-2\pi i/N}$

**Butterfly operation:** using the symmetry $E[k + N/2] = E[k]$ and $O[k + N/2] = O[k]$:
$$F[k] = E[k] + W^k O[k], \qquad F[k + N/2] = E[k] - W^k O[k]$$

Each pair $(F[k], F[k+N/2])$ costs just one complex multiplication and two additions. Applying this recursively ($N \to N/2 \to N/4 \to \ldots$) gives $\log_2 N$ stages, each with $N/2$ butterfly operations.

**Cost:** $T(N) = 2T(N/2) + O(N) \Rightarrow T(N) = O(N \log N)$.

For $N = 4096$: from $\sim 16M$ multiplications down to $\sim 49K$ — a $\sim 330\times$ speedup. This is why `np.fft.fft` is fast even for large signals.

---

### Circular Convolution and Zero-Padding

The DFT implicitly assumes the signal is **periodic** — after the last sample, it wraps back to the first. This means FFT-based convolution gives **circular convolution** (the tail of the kernel wraps around to the beginning of the signal), not the linear convolution we usually want.

**Fix:** zero-pad both $f$ and $g$ to length $\geq M + K - 1$ before taking DFTs, where $M$ is the signal length and $K$ is the kernel length.

Why this works: zero-padding creates enough room that the wrap-around contributions land in the padding region, not in the valid output.

Without zero-padding: Gaussian blur via FFT produces faint edge artifacts — opposite image edges bleed into each other. This is especially visible when there is strong intensity contrast at image boundaries.

---

## Advanced

### Gabor Filters and the Heisenberg-Gabor Limit

A **Gabor filter** combines a sinusoid with a Gaussian envelope:

$$g(x, y;\; f_0, \theta, \sigma) = \exp\!\left(-\frac{x_\theta^2 + \gamma^2 y_\theta^2}{2\sigma^2}\right)\cos(2\pi f_0 x_\theta + \psi)$$

**What each symbol means:**
- $(x_\theta, y_\theta)$ — coordinates of the image point rotated by angle $\theta$ (so the filter is oriented at angle $\theta$)
- $f_0$ — center spatial frequency of the filter (cycles per pixel)
- $\sigma$ — width of the Gaussian envelope (controls spatial extent)
- $\gamma$ — aspect ratio (spatial extent in perpendicular direction)
- $\psi$ — phase offset (0 = cosine, $\pi/2$ = sine)

The Gaussian envelope localizes the filter spatially; the sinusoid selects a specific frequency. The product is a filter that is both spatially localized *and* frequency-selective.

**The Heisenberg-Gabor uncertainty principle:**

$$\Delta x \cdot \Delta f \geq \frac{1}{4\pi}$$

where $\Delta x$ is the spatial extent of the filter and $\Delta f$ is its frequency bandwidth. No filter can be simultaneously narrow in both space and frequency — there is a fundamental tradeoff. The Gabor filter achieves exact equality: it is the unique filter with minimum joint space-frequency uncertainty.

**Biological and deep learning convergence:** V1 simple cells in the visual cortex have Gabor-like receptive fields (Marcelja, 1980). AlexNet's first conv layer spontaneously learns Gabor-like filters from data — no explicit bias for this structure. The visual system and CNNs independently converge on the same representation because the $1/f^2$ statistics of natural images reward exactly the joint space-frequency tuning that Gabors provide.

---

### DCT and JPEG Compression

The **Discrete Cosine Transform (DCT-II)** is a close relative of the DFT that uses only cosines (no complex numbers):

$$C[k] = \sum_{n=0}^{N-1} f[n]\cos\!\left(\frac{\pi k(2n+1)}{2N}\right)$$

**What each symbol means:**
- $f[n]$ — input pixel value at position $n$
- $k$ — frequency index (0 = DC, higher $k$ = higher frequency)
- $C[k]$ — amplitude of the $k$-th cosine component; purely real (no imaginary part)

JPEG applies the 2D DCT on non-overlapping $8 \times 8$ pixel blocks:
1. Subtract 128 from each pixel (center values around zero)
2. Apply 2D DCT-II → 64 frequency coefficients per block
3. Quantize by dividing each coefficient by a quantization matrix entry (larger divisors for high frequencies = more lossy)
4. Zig-zag scan (visits coefficients from low to high frequency) + run-length encode the many near-zero high-frequency values + Huffman code

**Why DCT over DFT for compression:**
- DFT assumes periodicity, so tiling $8\times 8$ blocks creates discontinuities at block edges → Gibbs ringing artifacts (the "blocking" effect at high JPEG compression). DCT implicitly applies even-symmetric extension, reducing spectral leakage.
- DCT produces purely real coefficients — no need to store imaginary parts, halving storage.
- The $1/f^2$ power law means high-frequency DCT coefficients are near zero for natural images and can be quantized coarsely with minimal perceptual loss. JPEG at 10:1 compression discards ~90% of information while remaining visually acceptable.

---

### Spectrogram and Mel Frequency Representation

The DFT of a full audio signal gives global frequency content — it tells you *what* frequencies are present but not *when*. For speech and music, timing matters: the word "cat" has the same phonemes as "tac" but in a different order.

**Short-Time Fourier Transform (STFT):** compute the DFT on short overlapping windows of the signal:

$$S[m, k] = \sum_n f[n]\, w[n - mH]\, e^{-2\pi ikn/N}$$

**What each symbol means:**
- $f[n]$ — audio sample at time $n$
- $w[\cdot]$ — window function (typically Hann window) that smoothly tapers to zero at the edges, reducing spectral leakage from the sharp cut-off of each frame
- $m$ — frame index (which window we are computing)
- $H$ — hop size in samples (how many samples to advance between frames; $H < N$ means frames overlap)
- $k$ — frequency bin index within the frame
- $S[m, k]$ — complex amplitude at time frame $m$, frequency $k$; taking $|S[m,k]|^2$ gives the **spectrogram** (power at each time-frequency cell)

**Mel spectrogram:** human hearing is logarithmic — doubling frequency raises pitch by one octave, so high frequencies are compressed in perception. The **mel scale** maps frequency $f$ (Hz) to perceived pitch:

$$m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

**What each symbol means:**
- $f$ — frequency in Hz
- $m$ — frequency in mels (perceptual units)
- 700 Hz — the "corner frequency" where the scale transitions from linear (below) to logarithmic (above)
- 2595 — a scaling constant so that 1000 Hz = 1000 mels (by convention)

At low frequencies (< 500 Hz), mel spacing is nearly linear — the ear can distinguish small frequency differences. At high frequencies (> 4 kHz), it is strongly logarithmic — the ear treats a much wider range of frequencies as equivalent.

Modern audio deep learning (Whisper, wav2vec 2.0, AudioMAE) uses 80-channel log-mel spectrograms as inputs. The mel compression reduces raw audio at 16 kHz (16,000 samples per second) to an $80 \times 100$ matrix (80 mel bins × 100 time frames per second) — a 10:1 compression that discards frequencies the ear cannot distinguish anyway.

**Connection to [[spectral-bias]]:** neural networks trained with gradient descent learn low-frequency components of the target function first. This is the Fourier-domain interpretation of why networks overfit slowly and why early stopping works — training is halted before high-frequency noise components are fit.

---

*See also: [[spectral-bias]] · [[dilated-convolution]] · [[feature-preprocessing]] · [[attention-mechanism]] · [[distributions-gaussian]]*
