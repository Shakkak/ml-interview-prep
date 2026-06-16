---
title: Grouped Convolution
tags: [grouped-convolution, resnext, depthwise-convolution, cardinality, efficient-cnn]
aliases: [grouped convolution, group convolution, ResNeXt, cardinality, channel groups]
difficulty: 1
status: complete
depends_on: [convolution-math, arch-bottleneck-1x1]
related: [arch-depthwise-separable, convolution-math, arch-residual-block, arch-bottleneck-1x1, cnn-architectures-guide]
---

# Grouped Convolution

---

## Fundamental

### Standard vs Grouped Convolution

**Standard convolution:** a filter of shape $(k \times k \times C_\text{in})$ convolves with all $C_\text{in}$ input channels to produce one output channel. For $C_\text{out}$ output channels: parameters = $k^2 \times C_\text{in} \times C_\text{out}$.

**Intuition:** standard convolution lets every output channel interact with every input channel — a fully connected mixing across channels. Grouped convolution restricts this: each output channel group only sees the corresponding input channel group. This is like having $g$ smaller independent convolutions running in parallel, which cuts cost by $g\times$ and sometimes improves accuracy by forcing the model to learn more diverse feature subsets.

**Grouped convolution** partitions input and output channels into $g$ groups, applying convolution independently within each group:

- Group $i$ sees input channels $[iC_\text{in}/g, (i+1)C_\text{in}/g]$ and produces output channels $[iC_\text{out}/g, (i+1)C_\text{out}/g]$
- Parameters: $k^2 \times (C_\text{in}/g) \times (C_\text{out}/g) \times g = k^2 \times C_\text{in} \times C_\text{out} / g$

**Savings:** $g\times$ fewer parameters and $g\times$ fewer FLOPs vs standard convolution, at the cost of no cross-group channel interactions.

---

## Intermediate

### Depthwise Convolution as Extreme Case

**Depthwise convolution** ($g = C_\text{in} = C_\text{out}$): each input channel has its own $k \times k$ filter. One filter per channel, no cross-channel mixing.

Parameters: $k^2 \times C$ (vs $k^2 \times C^2$ for standard). For $C = 256$: 256× parameter reduction.

This is the extreme case of grouped convolution and the spatial filtering step in MobileNet / depthwise-separable convolutions (see [[arch-depthwise-separable]]).

### ResNeXt: Cardinality as a New Dimension

**ResNeXt (Xie et al., 2017)** applies grouped convolution inside residual bottleneck blocks, introducing **cardinality** (number of groups) as a new dimension beyond width and depth:

Standard bottleneck: $1\times1$ → $3\times3$ ($C$ filters) → $1\times1$

ResNeXt bottleneck: $1\times1$ → $3\times3$ ($g$ groups, $C/g$ filters per group) → $1\times1$

**Key finding:** increasing cardinality $g$ while keeping total parameters constant improves accuracy more than increasing width or depth. ResNeXt-101 (32 groups, 8d wide) matches ResNet-200 accuracy at half the complexity.

**Intuition:** grouped convolution forces the network to learn $g$ independent feature transformations in parallel (like an ensemble within a layer), encouraging diversity of learned representations.

### Implementation and Hardware

Grouped convolutions are natively supported in PyTorch (`nn.Conv2d(groups=g)`) and are efficiently implemented on modern GPUs via parallel cuDNN kernels.

**Efficient ordering:** batch the $g$ group convolutions into one operation by reshaping tensors — modern DL frameworks do this automatically. FLOPs reduction directly translates to wall-clock speedup when $g$ is a power of 2.

---

## Advanced

### ShuffleNet: Cross-Group Information Flow

Pure grouped convolution blocks no cross-group communication — information stays within each group throughout the network. **ShuffleNet (Zhang et al., 2018)** adds a **channel shuffle** operation after grouped convolution:

Reshape channels to $(g, C/g)$, transpose to $(C/g, g)$, flatten. This redistributes channels across groups, enabling cross-group information flow while keeping the efficiency of grouped convolution.

### Connection to Multi-Head Attention

Grouped query attention (see [[grouped-query-attention]]) in transformers is structurally analogous to grouped convolution:
- Different attention heads independently compute attention over a partition of the embedding space
- Groups correspond to heads
- Cross-head interaction happens at the output projection (concatenation + $W_O$)

Both grouped convolution and multi-head attention leverage the same insight: parallel independent transformations on feature partitions, combined at the output.

## Links

- [[convolution-math]] — grouped convolution partitions input channels into $G$ groups and convolves each group independently; standard convolution is the special case $G=1$
- [[arch-bottleneck-1x1]] — ResNeXt uses grouped $3\times 3$ convolutions inside bottleneck blocks; increasing cardinality $G$ improves accuracy at the same FLOPs
- [[arch-depthwise-separable]] — depthwise convolution is the extreme case of grouped convolution where $G = C_\text{in}$; each channel forms its own group
- [[arch-residual-block]] — ResNeXt replaces standard convolutions in residual blocks with grouped convolutions, demonstrating "cardinality" as a dimension for scaling
- [[cnn-architectures-guide]] — ShuffleNet combines grouped convolutions with channel shuffling to enable cross-group information flow at low FLOPs
