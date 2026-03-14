# Deepfake Video Detection

## Project Goal
The objective of this project is to develop a deepfake detection pipeline across two modalities — **video** and **image** — that extracts features using convolutional neural networks, applies dimensionality reduction techniques such as PCA or LDA, and classifies content as real or fake using machine learning classifiers.

## Dataset
[FaceForensics++ Extracted Frames](https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames) — pre-extracted frames from real and manipulated videos covering five manipulation methods: Deepfakes, Face2Face, FaceSwap, FaceShifter, and NeuralTextures.

| Class  | Frames  | Videos | % of total frames |
|--------|---------|--------|-------------------|
| Real   | 31,949  | 1,089  | 16.7%             |
| Fake   | 159,601 | 1,089  | 83.3%             |
| **Total** | **191,550** | **2,178** | |

## Approach
1. We compared two CNN backbones — **MobileNetV2** and **EfficientNetV2-S** (both frozen, pretrained on ImageNet) — and selected **EfficientNetV2-S** for the full pipeline based on its deeper architecture and stronger performance on image recognition benchmarks
2. Apply **Global Max Pooling** to produce a 1,280-D vector per frame, then aggregate across all sampled frames via **mean pooling** into a single video-level vector
3. Reduce dimensionality with **PCA** (99% variance threshold: 1,280 → 30 components)
4. Visualize class separation with **multi-class LDA** (6 classes: real + 5 manipulation methods)
5. Train and compare **Logistic Regression**, **SVM (RBF kernel)**, and **MLP**
6. Select the best model by F1 score and evaluate on a stratified 20% held-out test set

## Team Members

| Name | Contributions |
|------|--------------|
| Muaataz Ismaeel | Video detection pipeline — CNN feature extraction (EfficientNetV2-S / MobileNetV2), PCA/LDA dimensionality reduction, classifier training and evaluation, inference pipeline, notebook and repository organization |
| Bilal Hasanov | Image detection pipeline — image-level deepfake classification using CNN features, preprocessing, model training and evaluation |

## Results

### Model Comparison

| Model               | F1 Score |
|---------------------|----------|
| Logistic Regression | 0.710    |
| MLP                 | 0.761    |
| **SVM (RBF)**       | **0.761** ✓ best |

### Best Model Evaluation — SVM (test set: 436 samples)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.76      | 0.78   | 0.77     | 218     |
| Fake  | 0.77      | 0.75   | 0.76     | 218     |
| **Overall Accuracy** | | | **76%** | 436 |

- **Misclassified samples:** 103 / 436 (23.6%)
- **Dimensionality reduction:** PCA 1,280 → 30 components, retaining **99.03%** of variance
- **LDA explained variance (2 components):** 72.60%
- **LDA analysis videos:** 5,694 across 6 classes (real + 5 manipulation methods)

### Inference Example
```
Prediction : Real
Confidence : 0.3213
```

## What Worked
- **EfficientNetV2-S** frozen features provided strong discriminative signal, achieving **76% accuracy without any fine-tuning**
- PCA aggressively compressed 1,280 dimensions down to just **30** while retaining 99% of the variance, making classifiers fast to train
- SVM with RBF kernel was the best performer, tying with MLP (F1 = 0.761) while being faster to train
- LDA visualization revealed meaningful separation between manipulation methods in 2D space

## What Did Not Work / Limitations
- The CNN backbone is **not fine-tuned** — features are generic ImageNet representations, not optimized for detecting deepfake artifacts
- **MobileNetV2 was not run through the full pipeline** — it was defined and compared at the architecture level, but EfficientNetV2-S was used for all experiments due to its stronger feature quality
- Frame features are **mean-averaged** into a single vector, discarding all temporal information across the video
- KernelPCA was explored but excluded due to high memory cost on the Kaggle environment
- Confidence scores from the SVM are derived via sigmoid approximation (not true probabilities), resulting in low-confidence outputs like 0.32 even on correct predictions
- Class imbalance at the frame level (16.7% real vs. 83.3% fake frames) could affect the quality of sampled representations

## Install Dependencies Locally
```bash
pip install -r requirements.txt
```

## Repository Structure (Video Branch)
```
├── deepfake_detection.ipynb   # main graded notebook 
├── dataset.py                 # frame loading, grouping, sampling, preprocessing
├── model.py                   # EfficientNetV2 / MobileNetV2 extractor builders
├── features.py                # batch CNN extraction, aggregation, caching
├── train.py                   # PCA/LDA fitting, classifier training & selection
├── evaluate.py                # confusion matrix, scatter plots, report
├── predict.py                 # inference on a new video directory
├── requirements.txt
└── README.md
```
