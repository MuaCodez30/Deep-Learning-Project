"""
Modular pipeline for video-level deepfake detection on FaceForensics++.

The package exposes the building blocks used by the graded notebook:

  - config         : centralized paths and hyperparameters
  - data_split     : preparing the FF++ train/val/test split
  - dataset        : transforms, ImageFolder + FFPPFrameDataset
  - model          : EfficientNet-B0 / ResNet50 / EfficientNet-V2-S +
                     fine-tuned classifier and feature-extractor variants
  - training       : train/eval loops and full fine-tuning schedule
  - inference      : frame-level inference and video-level aggregation
                     (majority voting, average softmax)
  - features       : video-level feature extraction (mean of frame embeddings)
  - classifiers    : SVM / MLP / LogReg trained on top of those features
  - evaluation     : metrics printing and confusion-matrix plots
"""
