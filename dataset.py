"""
dataset.py
----------
Data loading, frame discovery, grouping, sampling, and preprocessing utilities.
All functions are framework-agnostic and used by the notebook and other modules.
"""

from pathlib import Path
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path):
    """Create directory and all parents if they don't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def preprocess_frame(frame_path, size=(224, 224)):
    """
    Load an image, resize to `size`, convert BGR→RGB, and normalize to [0, 1].
    Returns None if the file cannot be read.
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None
    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV loads BGR; models expect RGB
    frame = frame.astype("float32") / 255.0          # Normalize to [0, 1]
    return frame


def find_frame_files(directory, extensions=None):
    """
    Recursively find all image files under `directory`.
    Handles both lower- and upper-case extensions for case-sensitive filesystems.
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg"]
    directory = Path(directory)
    frame_files = []
    for ext in extensions:
        frame_files.extend(list(directory.rglob(f"*{ext}")))
        frame_files.extend(list(directory.rglob(f"*{ext.upper()}")))
    return frame_files


def group_frames_by_video(frame_paths):
    """
    Group frame paths by their source video ID.
    Expects filenames in the format  <video_id>_<frame_number>.<ext>.
    Falls back to using the full stem as the video ID if the pattern does not match.
    """
    video_groups = defaultdict(list)
    for frame_path in frame_paths:
        filename = Path(frame_path).stem
        parts = filename.split("_")
        # Strip the trailing frame index to recover the video ID
        if len(parts) >= 2 and parts[-1].isdigit():
            video_id = "_".join(parts[:-1])
        else:
            video_id = filename
        video_groups[video_id].append(frame_path)
    return video_groups


def sample_frame_paths(frame_paths, max_frames, rng):
    """
    Return all frames when within the cap; otherwise draw a random subset
    without replacement to keep memory and compute tractable.
    """
    if max_frames is None or len(frame_paths) <= max_frames:
        return frame_paths
    return rng.sample(frame_paths, max_frames)


def find_data_directory(path):
    """Return `path` if it contains both real/ and fake/ subdirectories, else None."""
    path = Path(path)
    if (path / "real").exists() and (path / "fake").exists():
        return path
    return None


def load_dataset_summary(frames_dir):
    """
    Build a DataFrame summarising frame and video counts per class.
    Returns (summary_df, real_frames, fake_frames, real_video_groups, fake_video_groups).
    """
    frames_dir = Path(frames_dir)
    real_dir = frames_dir / "real"
    fake_dir = frames_dir / "fake"

    real_frames = find_frame_files(real_dir) if real_dir.exists() else []
    fake_frames = find_frame_files(fake_dir) if fake_dir.exists() else []
    real_video_groups = group_frames_by_video(real_frames) if real_frames else {}
    fake_video_groups = group_frames_by_video(fake_frames) if fake_frames else {}

    summary = pd.DataFrame([
        {"class": "real", "frames": len(real_frames), "videos": len(real_video_groups)},
        {"class": "fake", "frames": len(fake_frames), "videos": len(fake_video_groups)},
    ])
    total_f = summary["frames"].sum()
    total_v = summary["videos"].sum()
    summary["frame_pct"] = (summary["frames"] / total_f * 100) if total_f else 0
    summary["video_pct"] = (summary["videos"] / total_v * 100) if total_v else 0

    return summary, real_frames, fake_frames, real_video_groups, fake_video_groups
