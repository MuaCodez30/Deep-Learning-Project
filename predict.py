"""
predict.py
----------
Inference pipeline: given a directory of frames for a single video,
run the full feature extraction → PCA → classifier pipeline and return
a prediction label ("Real" or "Fake") with a confidence score.
"""

import random

import numpy as np

from dataset import find_frame_files, preprocess_frame, sample_frame_paths
from features import extract_features_batch, aggregate_frame_features


def predict_video_from_frames(
    frames_dir,
    feature_extractor,
    pca_model,
    scaler_model,
    classifier,
    max_frames=60,
    batch_size=32,
    frame_size=(224, 224),
    aggregation="mean",
    rng=None,
):
    """
    Predict whether a video is Real or Fake given a directory of its frames.

    Args:
        frames_dir:        Path to a folder containing frame images for one video.
        feature_extractor: Loaded Keras CNN extractor (from model.py).
        pca_model:         Fitted PCA loaded from disk (pca.pkl).
        scaler_model:      Fitted StandardScaler loaded from disk (scaler.pkl).
        classifier:        Trained classifier loaded from disk (final_model.pkl).
        max_frames:        Max frames to sample (matches training config).
        batch_size:        CNN forward-pass batch size.
        frame_size:        (H, W) used during preprocessing — must match training.
        aggregation:       'mean' or 'max' frame feature aggregation.
        rng:               random.Random instance for reproducible frame sampling.

    Returns:
        label:      "Real" or "Fake"
        confidence: float in [0, 1]
    """
    if rng is None:
        rng = random.Random(42)

    frame_paths = find_frame_files(frames_dir)
    frame_paths = sample_frame_paths(frame_paths, max_frames, rng)

    frames = []
    for fp in frame_paths:
        f = preprocess_frame(fp, frame_size)
        if f is not None:
            frames.append(f)

    if not frames:
        raise ValueError(f"No valid frames found in: {frames_dir}")

    # Extract CNN features and aggregate into a single video-level vector
    frame_features = extract_features_batch(frames, feature_extractor, batch_size)
    video_feature  = aggregate_frame_features(frame_features, method=aggregation)

    # Apply the same PCA → scale pipeline used during training
    X_in = pca_model.transform(video_feature.reshape(1, -1))
    X_in = scaler_model.transform(X_in)

    if hasattr(classifier, "predict_proba"):
        probs      = classifier.predict_proba(X_in)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    else:
        # SVM does not expose predict_proba by default;
        # convert the decision score to a probability via the sigmoid function
        score      = float(classifier.decision_function(X_in)[0])
        confidence = float(1 / (1 + np.exp(-score)))
        pred_idx   = 1 if score >= 0 else 0

    return ("Fake" if pred_idx == 1 else "Real"), confidence
