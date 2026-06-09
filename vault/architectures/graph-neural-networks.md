---
title: Graph Neural Networks (GNNs)
tags: [gnn, graph-neural-networks, message-passing, gcn, graphsage, gat, point-cloud, scene-graphs]
aliases: [GNN, graph neural network, message passing, GCN, GraphSAGE, GAT, graph attention network]
difficulty: 3
status: complete
related: [attention-mechanism, arch-positional-encoding, cnn-architectures-guide, feature-pyramid-networks, contrastive-learning]
---

# Graph Neural Networks (GNNs)

---

## Fundamental

CNNs exploit the regular grid structure of images; transformers exploit the sequence structure of text. Many real-world problems have **irregular, relational structure** that neither framework handles natively:

- **3D point clouds:** LiDAR scans have no grid — points are unordered sets in $\mathbb{R}^3$.
- **Molecular graphs:** atoms = nodes, covalent bonds = edges. The same molecule has many valid orderings.
- **Scene graphs:** objects = nodes, spatial/semantic relations = edges ("dog is *on* mat").
- **Skeleton pose graphs:** joints = nodes, limb segments = edges.

A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is defined by node features $h_v \in \mathbb{R}^d$, optional edge features $e_{uv}$, and adjacency $A \in \{0,1\}^{N \times N}$. GNNs generalize deep learning to this arbitrary relational structure.

### The Message Passing Framework (MPNN)

All modern GNNs instantiate the **message passing neural network** framework (Gilmer et al., 2017). Each layer performs three steps:

**1. Message:** each neighbor $u$ of node $v$ computes a message:
$$m_{u \to v}^{(k)} = \phi\!\left(h_v^{(k-1)},\ h_u^{(k-1)},\ e_{uv}\right)$$

**2. Aggregate** (must be permutation-invariant):
$$a_v^{(k)} = \bigoplus_{u \in \mathcal{N}(v)} m_{u \to v}^{(k)}, \quad \bigoplus \in \{\text{sum, mean, max}\}$$

**3. Update:**
$$h_v^{(k)} = \psi\!\left(h_v^{(k-1)},\ a_v^{(k)}\right)$$

After $K$ layers, each node representation aggregates information from its $K$-hop neighborhood. GCN, GraphSAGE, and GAT are all special cases of this framework — they differ only in how $\phi$, $\bigoplus$, and $\psi$ are defined.

---

## Intermediate

### GCN: Spectral Convolution as Spatial Averaging

GCN (Kipf & Welling, 2017) derives its update rule from [[eigenvalues-pca|spectral graph theory]], but the result simplifies to a spatially interpretable operation:

$$H^{(k+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2} H^{(k)} W^{(k)}\right)$$

where $\tilde{A} = A + I$ (self-loops), $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (degree). The symmetric normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ gives each node a normalized-average of its neighbors' features. This is a mean aggregation with degree-based normalization — a high-degree node contributes less per-neighbor, preventing the scale of aggregated features from depending on graph structure.

**Key limitations:**
- **Transductive:** the normalized adjacency is precomputed for the training graph — cannot generalize to new nodes not seen during training.
- **Oversmoothing:** after $K$ layers, the $K$-hop neighborhoods of adjacent nodes overlap substantially. With $K \to \infty$, all node representations converge to the same value — the global mean. In practice, GCNs rarely benefit from more than 2-3 layers.

### GraphSAGE: Inductive Aggregation

GraphSAGE (Hamilton et al., 2017) makes GNNs **inductive** — the learned aggregator applies to unseen nodes at test time:

$$h_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \text{CONCAT}\!\left[h_v^{(k-1)},\ \text{AGG}\!\left(\{h_u^{(k-1)} : u \in \mathcal{N}_s(v)\}\right)\right]\right)$$

Two changes from GCN: (1) sample a fixed-size neighborhood $\mathcal{N}_s(v)$ instead of using all neighbors (enables mini-batch training on giant graphs like Pinterest's 3B-node graph), and (2) concatenate the node's own representation with the aggregated neighbor representation instead of averaging — preserving the node's identity.

AGG options: mean (equivalent to GCN), [[rnn-lstm|LSTM]] over randomly ordered neighbors (breaks permutation-invariance but often works empirically), or max pooling of MLP outputs.

### GAT: Attention-Weighted Neighbors

GAT (Veličković et al., 2018) replaces fixed degree-normalized aggregation with **learned attention weights** over neighbors:

$$\alpha_{uv} = \frac{\exp\left(\text{LeakyReLU}\!\left(a^T [Wh_v \| Wh_u]\right)\right)}{\sum_{j \in \mathcal{N}(v)} \exp\left(\text{LeakyReLU}\!\left(a^T [Wh_v \| Wh_j]\right)\right)}$$

$$h_v' = \sigma\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u\right)$$

The attention coefficient $\alpha_{uv}$ depends on the pair $(h_v, h_u)$ — allowing the network to selectively weight informative neighbors and suppress noisy ones. Multi-head GAT runs $K$ independent attention mechanisms; intermediate layers concatenate outputs, the final layer averages them.

**Advantage over GCN:** no normalization by degree required; naturally handles heterogeneous graphs where edge informativeness varies. **Disadvantage:** $O(|\mathcal{E}|)$ attention computations — expensive on dense graphs.

### Graph Pooling for Graph-Level Tasks

For graph classification (e.g., molecule property prediction), we need one vector per graph:

- **Global pooling:** mean/sum/max over all $N$ node representations. Permutation-invariant, $O(N)$, but loses all structural information.
- **DiffPool (Ying et al., 2018):** learn a soft cluster assignment matrix $S \in \mathbb{R}^{N \times C}$ via a GNN, producing $C$ "supernodes." Apply GNN on the coarsened graph, repeat. Preserves hierarchy but costs $O(N^2)$ to compute $S$.
- **Top-k pooling:** score each node with a learned scalar, select top-$k$, drop the rest. Retain the induced subgraph. $O(N \log N)$ — scalable and differentiable via gating.

---

## Advanced

### Expressivity: The 1-WL Limitation

GNNs with sum aggregation are at most as powerful as the **1-dimensional Weisfeiler-Lehman (1-WL) graph isomorphism test**. Two graphs that 1-WL cannot distinguish will produce identical GNN representations — regardless of depth.

A concrete failure case: 1-WL (and hence any standard GNN) cannot distinguish a cycle $C_6$ from two triangles $K_3 \cup K_3$ if they share the same multiset of 1-hop neighborhoods. Neither graph has any node with a distinguishing local signature.

**Fix:** sum aggregation (not mean or max) is necessary for maximal expressivity within the 1-WL class — mean loses information about neighborhood size (a node with 2 identical neighbors vs 4 identical neighbors look the same). Graph Isomorphism Network (GIN, Xu et al., 2019) shows sum with an MLP aggregator reaches the 1-WL upper bound:

$$h_v^{(k)} = \text{MLP}^{(k)}\!\left((1+\epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$$

The learnable $\epsilon$ allows the model to interpolate between ignoring ($\epsilon=0$) and heavily weighting ($\epsilon \gg 1$) the node's own features.

### Higher-Order GNNs and Positional Encodings

Going beyond 1-WL requires either: (1) $k$-WL methods operating on $k$-tuples of nodes (exponential cost), or (2) injecting structural position information.

**Laplacian eigenvector positional encodings (LSPE):** compute the eigenvectors of the graph Laplacian $L = D - A$ (sorted by eigenvalue). Assign node $v$ its eigenvector entry as a positional feature. These eigenfeatures distinguish nodes that 1-WL cannot, because they encode global graph structure — similar to how [[arch-positional-encoding|sinusoidal encodings]] place tokens in a global coordinate system.

**Random walk positional encodings:** $\text{RW}_{v,k} = [A^k]_{vv}$ — the probability of returning to $v$ after $k$ steps. Cheap to compute, captures local topology.

### GNNs in Computer Vision

**3D point cloud processing:**
- PointNet (Qi et al., 2017): applies shared MLP to each point independently + global max pooling. No message passing — permutation-invariant but ignores local structure.
- PointNet++: hierarchical application of PointNet on local neighborhoods (farthest-point sampling + radius query). A GNN with radius-graph edges.
- DGCNN: recomputes the $k$-NN graph at each layer based on **current feature space** (not input xyz). Edges in layer 2 reflect semantic similarity learned in layer 1 — the graph becomes dynamic.

**Scene graph generation:**
1. Detect objects → bounding boxes + visual features → nodes.
2. Predict spatial/semantic relation labels ("on top of", "holding") → edge labels.
3. GNN propagates relational context: knowing an entity is "on top of" a flat surface should upweight the probability that it is a plate/cup.

**Skeleton-based action recognition (ST-GCN):**
- Nodes: body joints ($J=25$ joints for NTU RGB+D).
- Spatial edges: anatomical skeleton connections.
- Temporal edges: same joint at adjacent frames.
- Partition strategy: label each edge with a partition (root, centripetal neighbor, centrifugal neighbor) and learn separate weights per partition.
- ST-GCN achieves state-of-the-art on action recognition at 1/10 the parameters of CNN-based methods.

---

*See also: [[attention-mechanism]] · [[arch-positional-encoding]] · [[contrastive-learning]] · [[feature-pyramid-networks]] · [[eigenvalues-pca]] · [[linear-algebra-fundamentals]] · [[rnn-lstm]]*
