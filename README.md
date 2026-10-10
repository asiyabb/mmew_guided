# Micro-Guided Macro-Expression Recognition

This repository implements a hybrid deep learning architecture that enhances global macro-expression recognition by capturing fine-grained, short-term facial motion dynamics ("micro-clues") without requiring explicit micro-expression frame annotations.

---

## 📌 Architecture Overview

The pipeline leverages a dual-stream architecture built upon Vision Transformer (ViT) feature extraction:

1. **Feature Extraction:** Video frames are passed through a fine-tuned Vision Transformer (`ViT-B/16`) to extract spatial feature embeddings per frame.
2. **Micro Motion Encoding:** Short, overlapping sliding windows extract local temporal representations using 1D temporal convolutions, yielding **latent micro-clues**.
3. **Auxiliary Regularization:**
   - **Triplet Loss:** Encourages distinct clustering of local micro-motion embeddings.
   - **Temporal Consistency Loss:** Enforces smooth transitions between consecutive temporal windows.
4. **Macro Sequence Modeling:** Full video sequence representations are captured using a Bidirectional GRU.
5. **Cross-Attention & Adaptive Gating:** Macro representations query the latent micro-clues using Multi-Head Cross-Attention. A dynamic Sigmoid gating mechanism regulates the flow of micro-guidance into the final classification head.

---

## 📊 Experimental Results

Experiments were conducted on the MMEW dataset comparing a **Macro-Only Baseline Model** against the **Micro-Guided Macro Model**.

### **Performance Comparison Summary**

| Metric | Baseline (Macro) | Micro-Guided Macro | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **33.33%** | **41.67%** | **+8.33%** |
| **Macro F1-Score** | **0.3139** | **0.4016** | **+0.0877** |

---

### **Key Findings**
* **Faster & Stable Convergence:** The Micro-Guided model demonstrates significantly lower training loss and smoother validation accuracy convergence compared to the baseline.
* **Per-Class Improvements:**
  * **Happiness:** F1-score increased from `0.67` to `0.80`.
  * **Surprise:** F1-score increased from `0.29` to `0.62`.
* **Reduced Prediction Bias:** The inclusion of dynamic micro-guidance helped mitigate over-prediction bias toward dominant classes present in the baseline model.

---

## 🛠️ Project Structure

```text
├── config/
│   └── config.py               # Hyperparameters and path configurations
├── data/
│   └── dataset.py              # MMEW Dataset loader & Subject-Grouped splitting
├── models/
│   ├── vit_extractor.py        # ViT-B/16 spatial feature extractor
│   ├── micro_encoder.py        # 1D-CNN temporal micro-motion encoder
│   └── macro_guided_model.py   # GRU baseline & Cross-Attention Guided models
├── losses/
│   └── micro_losses.py         # Triplet and Temporal Consistency loss functions
├── visualize_results.py        # End-to-end training, validation & plot generation
└── evaluate.py                 # Evaluation script for saved model weights
