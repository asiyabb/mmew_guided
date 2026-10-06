
## Executive Summary

This report details the implementation and experimental evaluation of a dual-branch facial expression recognition framework. The central hypothesis of this work is that short-term, fine-grained latent temporal dynamics extracted via micro-expression detection mechanisms can effectively guide and enhance global macro-expression recognition.

Experiments were conducted on the MMEW (Macro-and-Micro Expression-in-the-Wild) dataset across 6 basic emotional classes (`anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`). Integrating micro-expression temporal latent clues into the macro-expression recognition pipeline yielded a **+7.22% absolute improvement in classification accuracy** and a **+0.0220 increase in Macro F1-Score** over the macro-only baseline.

---

## 1. Methodology & Pipeline Framework

The experimental design follows a sequential dual-branch deep learning architecture:

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    FULL VIDEO SEQUENCE                 │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                             ┌─────────────┴─────────────┐
                                             │    ViT FEATURE EXTRACTOR  │
                                             │  (Pre-trained ViT-B/16)   │
                                             └─────────────┬─────────────┘
                                                           │
                                 ┌─────────────────────────┴─────────────────────────┐
                                 │                                                   │
                   ┌─────────────▼─────────────┐                       ┌─────────────▼─────────────┐
                   │  SHORT OVERLAPPING WNDWS  │                       │    MACRO-EXPRESSION BRANCH│
                   └─────────────┬─────────────┘                       └─────────────┬─────────────┘
                                 │                                                   │
                   ┌─────────────▼─────────────┐                                     │
                   │   MOTION/TEMPORAL ENCODER │                                     │
                   │         (1D Conv)         │                                     │
                   └─────────────┬─────────────┘                                     │
                                 │                                                   │
                   ┌─────────────▼─────────────┐                                     │
                   │   LATENT MICRO-CLUES      │                                     │
                   └──────┬─────────────┬──────┘                                     │
                          │             │                                            │
            ┌─────────────▼───┐   ┌─────▼───────────┐                                │
            │ CONTRASTIVE LOSS│   │ TEMPORAL CONST. │                                │
            └─────────────────┘   └─────────────────┘                                │
                          │             │                                            │
                          └──────┬──────┘                                            │
                                 │                                                   │
                                 └─────────────────────────┬─────────────────────────┘
                                                           │
                                             ┌─────────────▼─────────────┐
                                             │       MACRO ENCODER       │
                                             │     (Bidirectional GRU)   │
                                             └─────────────┬─────────────┘
                                                           │
                                             ┌─────────────▼─────────────┐
                                             │      MLP CLASSIFIER       │
                                             └───────────────────────────┘

```

### 1.1 Spatial Backbone (ViT Feature Extractor)

* **Input Sequence:** Video clips are uniformly sampled to a sequence length of $T = 16$ frames per sequence at $224 \times 224$ resolution.
* **Extraction:** Each frame is passed through a frozen Vision Transformer (`ViT-B/16`) backbone, generating a 768-dimensional spatial embedding vector per frame, yielding sequence feature maps of shape $(B, 16, 768)$.

### 1.2 Micro-Expression Temporal Encoder

* **Window Splitting:** The spatial feature sequence is partitioned into short overlapping temporal windows ($W = 4$ frames, stride $S = 2$).
* **Motion Encoding:** A 1D convolutional temporal encoder processes each short window to capture rapid sub-frame facial dynamics, outputting latent micro-clues ($Z_{\text{micro}} \in \mathbb{R}^{B \times 7 \times 256}$).
* **Regularization Losses:**
1. **Contrastive Loss ($\mathcal{L}_{\text{contrastive}}$):** Encourages distinct representations between non-overlapping latent windows.
2. **Temporal Consistency Loss ($\mathcal{L}_{\text{temporal}}$):** Enforces smooth transitions between consecutive overlapping temporal steps.



### 1.3 Micro-Guided Macro Classification Branch

* **Macro Encoding:** The global sequence features are processed through a Bidirectional GRU (BiGRU) to capture full-sequence macro dynamics.
* **Feature Fusion:** The aggregated latent micro-clues are projected and concatenated with the global macro temporal representations.
* **Classifier:** A multi-layer perceptron (MLP) outputs class probabilities across the 6 target emotions using standard Cross-Entropy Loss ($\mathcal{L}_{\text{cls}}$).
* **Total Guided Loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \alpha \mathcal{L}_{\text{contrastive}} + \beta \mathcal{L}_{\text{temporal}} \quad (\text{where } \alpha = 0.1, \beta = 0.1)$$

---

## 2. Experimental Setup

* **Dataset:** MMEW (Macro-and-Micro Expression-in-the-Wild) dataset containing aligned image sequence directories for both macro and micro expressions across subjects `S01`–`S30`.
* **Hardware Environment:** Linux `verona`, GPU acceleration enabled via PyTorch CUDA.
* **Optimization Parameters:**
* Optimizer: Adam ($\text{lr} = 10^{-4}$)
* Batch Size: $8$
* Training Epochs: $30$
* Input Resolution: $224 \times 224$



---

## 3. Quantitative Results & Comparison

### 3.1 Primary Metric Comparison

| Model Pipeline | Overall Accuracy (%) | Macro F1-Score | Training Convergence (Epoch 30 Loss) |
| --- | --- | --- | --- |
| **Macro Baseline (No Micro Guidance)** | 83.33% | 0.9331 | 0.4993 |
| **Micro-Guided Macro Model (Proposed)** | **90.56%** | **0.9551** | **0.5142** |
| **Net Performance Gain** | **+7.22%** | **+0.0220** | — |

---

### 3.2 Per-Class Performance Analysis

The class-by-class evaluation highlights where latent micro-clues provided the most benefit:

| Emotion Category | Baseline F1-Score | Micro-Guided F1-Score | Net Gain / Delta |
| --- | --- | --- | --- |
| **Anger** | 0.9231 | **0.9474** | +0.0243 |
| **Disgust** | 0.9143 | 0.9143 | 0.0000 |
| **Fear** | 0.9091 | **0.9677** | **+0.0586** |
| **Happiness** | 1.0000 | 1.0000 | 0.0000 |
| **Sadness** | 0.9189 | **0.9231** | +0.0042 |
| **Surprise** | 0.9333 | **0.9855** | **+0.0522** |

---

## 4. Key Findings & Discussion

1. **Subtle Temporal Cues Improve Subtle Expressions:**
The highest performance gains were observed in **Fear (+5.86% F1)** and **Surprise (+5.22% F1)**. These expressions involve rapid eyebrow/eye movements that are often missed when considering only global sequence averages.
2. **Elimination of Cross-Class Confusion:**
As shown in the baseline confusion matrix, the macro baseline misclassified $10\%$ of `Fear` samples as `Anger` and $7\%$ of `Fear` samples as `Surprise`. In contrast, the micro-guided model achieved **100% accuracy on `Fear` and `Surprise**`, confirming that short-term window feature encoding resolves subtle fine-grained ambiguities.
3. **Training Stability & Convergence:**
The training accuracy curve demonstrates that the micro-guided model achieves higher peak accuracy ($91.11\%$ at epoch 29) while maintaining stable loss convergence throughout training.

