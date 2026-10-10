# Micro-Guided Macro-Expression Recognition Framework

A generalized deep learning framework for macro-expression recognition guided by micro-expression temporal clues on the MMEW dataset.

---

## 📌 Project Architecture

* **Spatial Feature Extractor:** ViT-B/16 (ImageNet pre-trained) with top-block Transformer fine-tuning.
* **Temporal Micro-Encoder:** Sliding window temporal encoder guided by Triplet & Temporal Consistency losses.
* **Micro-Guided Macro Model:** Bidirectional GRU macro sequence processor augmented with dynamic Cross-Attention and Gated Fusion.
* **Subject-Independent Splitting:** 3-way Grouped Split (Train / Val / Test) preventing subject data leakage.

---

## 📁 Repository Structure

```text
mmew_guided_recognition/
├── config/
│   └── config.py               # Global hyperparameters and dataset configurations
├── data/
│   └── dataset.py              # Sequence dataset loader & subject-grouping splitters
├── losses/
│   └── micro_losses.py         # Micro-Triplet & Temporal Consistency loss functions
├── models/
│   ├── vit_extractor.py        # ViT Feature Extractor
│   ├── micro_encoder.py        # Temporal Motion Micro-Encoder
│   └── macro_guided_model.py   # Baseline GRU and Cross-Attention Micro-Guided models
├── baseline_best.pth           # Saved best checkpoint for Macro Baseline
├── guided_model_best.pth       # Saved best checkpoint for Micro-Guided Model
├── micro_encoder_best.pth     # Saved best checkpoint for Micro-Encoder
├── visualize_results.py        # Training, checkpointing, and visualization script
├── evaluate.py                 # Standalone Test-Set evaluation script
└── experiment_comparison.png   # Generated training & validation comparison charts
