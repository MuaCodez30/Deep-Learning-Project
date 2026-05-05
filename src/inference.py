from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import IDX_TO_CLASS


def evaluate_frame_classifier(model, dataset, device, batch_size=32, num_workers=2):
    # standard pytorch dataloader with no shuffle
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # eval mode
    model.eval()
    all_preds, all_labels = [], []

    # no grad since no training
    with torch.no_grad():
        for batch in loader:
            # only first two elements matter (image, label)
            images, labels = batch[0], batch[1]
            images = images.to(device)

            # forward pass
            outputs = model(images)
            # take argmax to get predicted class
            _, preds = torch.max(outputs, dim=1)

            # collect predictions and ground truth
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def run_frame_inference(model, dataset, device, batch_size=32, num_workers=2):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    rows = []

    with torch.no_grad():
        for images, labels, img_paths, video_ids, filenames in loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass + softmax to get class probabilities
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # argmax for predicted class, max for confidence
            pred_labels = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1).values
            # per-class probabilities (we'll need both for soft aggregation)
            fake_probs = probs[:, 0]
            real_probs = probs[:, 1]

            # build one row per frame in this batch
            for i in range(len(img_paths)):
                rows.append({
                    "video_id": video_ids[i],
                    "filename": filenames[i],
                    "path": img_paths[i],
                    "true_label_idx": int(labels[i].item()),
                    "true_label": IDX_TO_CLASS[int(labels[i].item())],
                    "pred_label_idx": int(pred_labels[i].item()),
                    "pred_label": IDX_TO_CLASS[int(pred_labels[i].item())],
                    "confidence": float(confidence[i].item()),
                    "fake_prob": float(fake_probs[i].item()),
                    "real_prob": float(real_probs[i].item()),
                })

    return pd.DataFrame(rows)


def aggregate_to_videos(frame_df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    # process one video at a time
    for video_id, group in frame_df.groupby("video_id"):
        # the true label is the same for all frames of a video, just take the mode
        true_label_idx = int(group["true_label_idx"].mode()[0])
        true_label = IDX_TO_CLASS[true_label_idx]

        # hard majority voting on per-frame predictions
        vote_counts = Counter(group["pred_label_idx"].tolist())
        majority_pred_idx = vote_counts.most_common(1)[0][0]
        majority_pred = IDX_TO_CLASS[majority_pred_idx]

        # soft averaging of softmax probabilities, then argmax
        avg_fake_prob = float(group["fake_prob"].mean())
        avg_real_prob = float(group["real_prob"].mean())
        avg_pred_idx = int(avg_real_prob > avg_fake_prob)
        avg_pred = IDX_TO_CLASS[avg_pred_idx]

        # store both predictions for this video
        rows.append({
            "video_id": video_id,
            "num_frames": len(group),
            "true_label_idx": true_label_idx,
            "true_label": true_label,
            "majority_pred_idx": majority_pred_idx,
            "majority_pred": majority_pred,
            "avg_fake_prob": avg_fake_prob,
            "avg_real_prob": avg_real_prob,
            "avg_pred_idx": avg_pred_idx,
            "avg_pred": avg_pred,
        })

    # sort by video_id for reproducibility
    return (
        pd.DataFrame(rows)
        .sort_values("video_id")
        .reset_index(drop=True)
    )
