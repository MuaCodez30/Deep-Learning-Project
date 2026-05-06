from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_video_features(
    dataset,
    feature_extractor,
    device="cpu",
    batch_size=32,
    num_workers=2,
):
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor.eval()

    # collect per-video lists of frame embeddings and a label per video
    video_to_features = defaultdict(list)
    video_to_label = {}

    with torch.no_grad():
        for images, labels, _img_paths, video_ids, _filenames in loader:
            images = images.to(device)

            # forward pass that returns a 1280-d embedding per frame
            feats = feature_extractor(images).cpu().numpy()
            labels = labels.cpu().numpy()

            # group frame embeddings by video id
            for i in range(len(video_ids)):
                vid = video_ids[i]
                video_to_features[vid].append(feats[i])
                video_to_label[vid] = int(labels[i])

    # average frame embeddings within each video
    X, y, vid_list = [], [], []
    for vid in sorted(video_to_features.keys()):
        # stack into [num_frames, D] and take the mean along the frame axis
        frame_feats = np.stack(video_to_features[vid], axis=0)
        X.append(frame_feats.mean(axis=0))
        y.append(video_to_label[vid])
        vid_list.append(vid)

    # final shapes: X is [num_videos, D], y is [num_videos]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, vid_list
