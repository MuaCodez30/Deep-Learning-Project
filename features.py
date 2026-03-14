"""
features.py
-----------
CNN feature extraction and aggregation utilities.
Handles batched forward passes, per-frame → per-video aggregation,
and full dataset feature extraction with disk caching.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import (
    find_frame_files,
    group_frames_by_video,
    preprocess_frame,
    sample_frame_paths,
)


def extract_features_batch(frames, feature_extractor, batch_size):
    """
    Stack all frames into one array and run a single batched CNN forward pass.

    Args:
        frames:            List of preprocessed numpy arrays, each shape (H, W, 3).
        feature_extractor: Keras Model that maps (batch, H, W, 3) → (batch, D).
        batch_size:        Number of frames per forward pass.

    Returns:
        np.ndarray of shape (N_frames, D).
    """
    frames_array = np.array(frames, dtype="float32")
    return feature_extractor.predict(frames_array, verbose=0, batch_size=batch_size)


def aggregate_frame_features(frame_features, method="mean"):
    """
    Collapse a per-frame feature matrix (N_frames × D) into a single video-level vector.

    method='max'  — captures the most strongly activated feature across all sampled frames.
    method='mean' — smooth average over all sampled frames (default).
    """
    if method == "max":
        return np.max(frame_features, axis=0)
    return np.mean(frame_features, axis=0)


def build_video_feature(frame_paths, feature_extractor, max_frames, batch_size,
                         frame_size, aggregation, rng):
    """
    Extract and aggregate CNN features for a single video.

    Returns a 1-D feature vector, or None if no valid frames are found.
    """
    frame_paths = sample_frame_paths(frame_paths, max_frames, rng)
    frames = []
    for fp in frame_paths:
        f = preprocess_frame(fp, frame_size)
        if f is not None:
            frames.append(f)
    if not frames:
        return None
    feats = extract_features_batch(frames, feature_extractor, batch_size)
    return aggregate_frame_features(feats, method=aggregation)


def extract_all_video_features(frames_dir, feature_extractor, max_frames, batch_size,
                                frame_size, aggregation, rng,
                                cache_dir=None):
    """
    Extract video-level features for every real and fake video under `frames_dir`.
    If `cache_dir` is provided and cached .npy files exist there, they are loaded
    instead of re-running the expensive CNN forward passes.

    Returns:
        X          np.ndarray (N_videos, D)
        y          np.ndarray (N_videos,)   — 0=real, 1=fake
        video_ids  np.ndarray (N_videos,)   — string identifiers
    """
    frames_dir = Path(frames_dir)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        X_path   = cache_dir / "X_video_features.npy"
        y_path   = cache_dir / "y_video_label.npy"
        ids_path = cache_dir / "video_ids.npy"
        if X_path.exists() and y_path.exists():
            X        = np.load(X_path)
            y        = np.load(y_path)
            video_ids = np.load(ids_path) if ids_path.exists() else None
            print("Loaded cached features from:", cache_dir)
            return X, y, video_ids

    video_features, video_labels, video_ids = [], [], []

    for class_name, label in [("real", 0), ("fake", 1)]:
        class_path = frames_dir / class_name
        if not class_path.exists():
            continue
        frame_files = find_frame_files(class_path)
        if not frame_files:
            continue
        video_groups = group_frames_by_video(frame_files)
        print(f"{class_name}: {len(video_groups)} videos")
        for video_id, paths in tqdm(video_groups.items(), desc=f"{class_name} videos"):
            feat = build_video_feature(paths, feature_extractor, max_frames,
                                       batch_size, frame_size, aggregation, rng)
            if feat is None:
                continue
            video_features.append(feat)
            video_labels.append(label)
            video_ids.append(f"{class_name}_{video_id}")

    X        = np.array(video_features)
    y        = np.array(video_labels)
    video_ids = np.array(video_ids)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(X_path, X)
        np.save(y_path, y)
        np.save(ids_path, video_ids)
        print(f"Saved features to: {cache_dir}")

    return X, y, video_ids
