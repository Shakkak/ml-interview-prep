---
title: "VLM Explainability and Interpretability"
tags: [explainability, grad-cam, attention-rollout, concept-bottleneck-model, tcav, vlm-attribution, interpretability, xai]
aliases: [VLM Interpretability, Explainable VLMs, Grad-CAM, Attention Rollout, CBM]
status: complete
---

Explainability methods let us understand why a model makes a particular prediction. For scientific applications like ecology and biodiversity monitoring, explainability is not optional — it is a core requirement for trust, debugging, and regulatory compliance. This file covers methods from first principles.

## 1. Why Explainability Matters for Ecology ML

In ecology, predictions from ML models are used to guide conservation decisions, report to funding agencies, and publish in peer-reviewed journals. A model that correctly predicts "this location is suitable habitat for species X" must be interpretable to scientists who want to know *which environmental features* drove the prediction. Otherwise:

- Errors cannot be diagnosed
- Spurious correlations (e.g., model uses road proximity as a proxy for habitat) go undetected
- Regulatory frameworks for environmental impact assessment require justification

**Specific use cases for explainability in VLMs:**
- "Which part of the leaf image made the model predict 'high chlorophyll content'?"
- "Why did the VLM classify this habitat as 'degraded' rather than 'pristine'?"
- "Which spectral bands drove the species distribution prediction?"

---

## 2. Gradient-Based Attribution

### 2.1 Vanilla Gradient (Saliency Maps)

The simplest attribution: compute the gradient of the output class score with respect to the input pixels.

$$\text{Saliency}(x_i) = \left|\frac{\partial f(x)}{\partial x_i}\right|$$

High gradient at pixel $i$ means changing that pixel slightly would strongly change the output. Intuition: the model is "sensitive" to that pixel.

**Problem**: gradients are noisy and locally inconsistent. A pixel important for the classification may have low gradient if the loss surface is flat around the correct class.

### 2.2 Grad-CAM: Gradient-weighted Class Activation Mapping

**Paper**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (Selvaraju et al., 2017)

**The core insight**: the final convolutional feature map retains spatial information. The gradient of the class score with respect to this feature map tells us which spatial locations in the feature map are important.

**Algorithm** for a CNN classifying image $I$ as class $c$:

1. Forward pass: compute feature maps $A^k$ at the last convolutional layer (shape: $H_{fm} \times W_{fm}$ for each of $K$ channels)
2. Compute class score: $y^c$ (before softmax)
3. Backpropagate to get gradients: $\frac{\partial y^c}{\partial A^k_{ij}}$ for each feature map pixel $(i,j)$ in channel $k$
4. Global average pool the gradients to get per-channel importance weights:
$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$
5. Combine: weighted sum of feature maps, ReLU to keep only positive contributions:
$$\text{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
6. Upsample to input image size via bilinear interpolation

**Result**: a heatmap showing which image regions activated for class $c$.

**Intuition**: if channel $k$ of the feature map has high activations in a specific spatial region AND has high gradient for class $c$, that region is both activated and important for the classification → highlight it.

```python
# Grad-CAM implementation sketch (for a CNN)
from torch.autograd import grad

model.eval()
output = model(image)
class_score = output[0, target_class]
class_score.backward()

gradients = model.get_gradients()  # ∂y/∂A (hook registered earlier)
activations = model.get_activations()  # A (hook registered earlier)

weights = gradients.mean(dim=[2, 3], keepdim=True)  # global avg pool
cam = (weights * activations).sum(dim=1, keepdim=True)
cam = F.relu(cam)
cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear')
```

### 2.3 Grad-CAM++

Extension of Grad-CAM using a weighted combination of higher-order gradients. Better at handling:
- Multiple instances of the same class in one image
- Objects partially off-frame

Weight formula:
$$\alpha_{ijk}^c = \frac{\frac{\partial^2 y^c}{\partial (A^k_{ij})^2}}{2\frac{\partial^2 y^c}{\partial (A^k_{ij})^2} + \sum_{a,b}A^k_{ab}\frac{\partial^3 y^c}{\partial (A^k_{ij})^3}}$$

### 2.4 Integrated Gradients

**Paper**: Sundararajan et al., 2017

Computes the integral of gradients along a straight path from a baseline input $x'$ (e.g., black image) to the actual input $x$:

$$\text{IntGrad}_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha$$

This satisfies the *completeness axiom*: attributions sum to the model output difference $F(x) - F(x')$.

More principled than Grad-CAM but requires many forward passes (approximated with ~50 steps).

---

## 3. Attention-Based Explainability

### 3.1 Raw Attention Weights

[[vision-transformer|Vision Transformers (ViT)]] compute attention weights: for each query token, how much does it attend to each key token?

It's tempting to visualize the [[attention-mechanism|attention]] from the [CLS] token to the patch tokens as a saliency map. **This is unreliable** because:
- Attention is not attribution: a token attending to another doesn't mean it uses that information for the final prediction
- Attention in early layers is often diffuse ("attending everywhere")
- Multi-head attention has many heads with different roles; averaging is misleading

### 3.2 Attention Rollout

**Paper**: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

Attention rollout propagates attention through all transformer layers by multiplying attention matrices:

$$\hat{A}^{(l)} = 0.5 \cdot A^{(l)} + 0.5 \cdot I$$  (add residual identity)
$$\text{Rollout}^{(L)} = \hat{A}^{(L)} \cdot \hat{A}^{(L-1)} \cdots \hat{A}^{(1)}$$

This gives the *effective* attention from the output to each input token, accounting for how information flows through skip connections.

**Intuition**: in layer 1, token A attends to B; in layer 2, B attends to C; so effectively, A also attends to C through B. Rollout propagates these indirect paths.

Better than raw attention, but still not a causal attribution (not the same as gradient-based methods).

### 3.3 DINO Self-Attention

[[contrastive-learning|DINO]] (self-supervised ViT training) produces attention heads that naturally segment objects without any supervision. The last-layer attention maps from the [CLS] token to patch tokens form remarkably clean object boundaries.

**Why**: DINO's self-distillation objective encourages the model to produce sharp, consistent representations. The attention focuses on object structure rather than low-level texture.

For ecology: DINO attention maps on plant images cleanly segment leaves, flowers, and stems even without any leaf segmentation training. Useful for weak supervision.

### 3.4 GradCAM for ViT

Apply Grad-CAM to a ViT's attention output (not convolutional feature maps):
1. Use the [CLS] token gradient with respect to the attention output at layer L
2. Weight each attention head by its gradient magnitude
3. Average across heads and upsample to image resolution

---

## 4. Perturbation-Based Methods

### 4.1 LIME: Local Interpretable Model-Agnostic Explanations

**Paper**: Ribeiro et al., 2016

LIME explains any black-box model by locally approximating it with an interpretable surrogate:

1. **Perturb**: create variations of the input image by randomly masking superpixels (connected pixel regions with similar color)
2. **Query**: pass each perturbed image through the model → get predictions
3. **Fit**: fit a linear model to (perturbed_image, prediction) pairs, weighted by proximity to original
4. **Explain**: the linear coefficients identify which superpixels contributed positively/negatively

**For ecology**: LIME can explain which leaf region (vein, margin, surface texture) was used for species classification. Superpixels correspond to semantically meaningful leaf parts.

### 4.2 SHAP: SHapley Additive exPlanations

SHAP computes each feature's contribution by its Shapley value — the average marginal contribution across all possible feature subsets:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

For images: features are superpixels or individual pixels. Very expensive to compute exactly; KernelSHAP approximates it efficiently.

**Advantage over LIME**: SHAP has theoretical guarantees (completeness, symmetry, dummy) that LIME lacks.

---

## 5. VLM-Specific Explainability

### 5.1 Cross-Attention Visualization in VLMs

In [[vlm-architectures|VLMs]] like Flamingo (which uses cross-attention between text and image), cross-attention weights directly show which image regions the model attends to when generating each output token.

For a generated token "leaf":
- Extract the cross-attention weight matrix from the cross-attention layer when "leaf" was generated
- The row corresponding to the "leaf" token gives attention weights over image patch tokens
- Reshape to spatial grid and overlay on image → shows what image region the word "leaf" referred to

This is the most natural explainability tool for Flamingo-style VLMs.

### 5.2 Token Attribution for Autoregressive VLMs (LLaVA)

LLaVA uses self-attention, not cross-attention. To explain which visual tokens influenced a generated answer token $t$:

1. Compute gradient of log-probability of $t$ with respect to each visual token embedding: $\frac{\partial \log P(t)}{\partial h_i}$ for visual token $i$
2. Multiply by token value (integrated gradients idea): $\alpha_i = h_i \cdot \frac{\partial \log P(t)}{\partial h_i}$
3. Take L2 norm: $\text{importance}_i = \|\alpha_i\|_2$

This gives per-visual-token importance scores for generating a specific answer word.

### 5.3 Language Grounding Verification

For VLMs used in ecology, verify that the language grounding is correct:
- Ask "What part of the plant is shown?" → the VLM should highlight the correct region
- Check if the VLM's description of the image matches what a botanist would say
- Use Grounding DINO to verify that "I see a leaf with dark veins" actually corresponds to regions with dark veins

---

## 6. Concept Bottleneck Models (CBMs)

### 6.1 What are CBMs?

**Paper**: "Concept Bottleneck Models" (Koh et al., 2020)

Standard models: image → black-box features → prediction.
CBMs: image → **concept predictions** → prediction.

The concepts are human-defined semantic attributes (for a bird: "has red breast", "has pointed beak", "has long tail"). The model must predict each concept first, then predicts the class from those concepts only.

```
Image
  ↓ Concept predictor (one sigmoid output per concept)
[P(red breast)=0.9, P(pointed beak)=0.3, P(long tail)=0.8]
  ↓ Concept-to-class predictor (linear or small MLP)
Class prediction: "Robin"
```

**Advantages**:
- Fully transparent: you can see exactly which concepts drove the prediction
- Editable: if the model says "has red breast" but you know it doesn't, you can override the concept
- Debuggable: if prediction is wrong, check which concept was wrong

**Disadvantages**:
- Requires concept annotations at training time (expensive)
- Performance can be lower than black-box models if concepts are incomplete

### 6.2 Post-Hoc CBMs

**Paper**: "Post-hoc Concept Bottleneck Models" (Yuksekgonul et al., 2022)

Trains a concept predictor on top of a frozen pretrained model. No need to retrain from scratch. Concepts are defined as text descriptions, and concept predictions use [[clip|CLIP]] similarity:

$$P(\text{concept}_k \mid \text{image}) = \text{CLIP-similarity}(\text{image}, \text{concept}_k\_\text{text})$$

More flexible: concepts can be defined at test time.

### 6.3 Label-Free CBMs with CLIP

Use CLIP to define concepts without any concept-level annotation:
1. Define concepts as text strings: ["has elongated leaves", "has yellow flowers", "growing in dry soil"]
2. Concept scores: CLIP-similarity of each image with each concept text
3. Train a linear classifier on these concept scores

The full pipeline is trained only on class labels; concept interpretations are free.

### 6.4 TCAV: Testing with Concept Activation Vectors

**Paper**: Kim et al., 2018

TCAV asks: "How sensitively does the model's prediction depend on concept C?"

1. Collect positive examples of concept C (images with the concept) and negative examples
2. Train a linear classifier in the model's activation space to separate concept vs no-concept → the classifier's weight vector is the "Concept Activation Vector" (CAV)
3. Compute TCAV score: fraction of inputs in class $k$ for which moving in the CAV direction increases the class probability

TCAV score > 0.5 means the concept positively influences the class prediction.

---

## 7. Application: Explainable VLM for Ecology

### 7.1 Workflow for Explainable Plant Trait Prediction

1. **Model**: LLaVA fine-tuned on plant images with trait labels
2. **Question**: "What leaf characteristics indicate high nitrogen content?"
3. **Generate answer**: "The bright green color and large leaf area suggest high nitrogen."
4. **Verify with Grad-CAM**: apply Grad-CAM to the visual encoder → should highlight green, large leaf regions
5. **Verify with attention rollout**: cross-attention in the decoder should attend to leaf body when generating "nitrogen"
6. **Concept check**: use CLIP to verify "bright green" and "large leaf area" correlate with the image

If the Grad-CAM heatmap focuses on the flower rather than the leaf when predicting leaf nitrogen, the model has learned a spurious correlation (flowers and leaf nitrogen may co-vary in a biased dataset).

### 7.2 Explainability for Species Distribution Models

For a CNN predicting habitat suitability from Sentinel-2:
- **Band importance**: use SHAP on the 13 input bands → which bands are most important?
- **Spatial saliency**: Grad-CAM shows which image regions drove the prediction
- **Concept probing**: does the model activate for "forest texture" or "water body nearby"?
- **Counterfactual**: "What would need to change in this image for the model to predict unsuitable habitat?"

---

## See Also

[[vlm-architectures]], [[biodiversity-ml]], [[vision-transformer]], [[clip]], [[attention-mechanism]], [[contrastive-learning]], [[self-supervised-overview]]
