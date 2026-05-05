import os
import random
import shutil
from collections import defaultdict


def get_real_ids(real_dir):
    # one subfolder per source video, name = id
    return sorted([
        d for d in os.listdir(real_dir)
        if os.path.isdir(os.path.join(real_dir, d))
    ])


def get_fake_source_ids(fake_dir):
    method_to_sources = defaultdict(set)
    # walk each forgery method folder
    for method in os.listdir(fake_dir):
        method_path = os.path.join(fake_dir, method)
        if not os.path.isdir(method_path):
            continue
        # each subfolder is "src_dst", we only care about the source id
        for video in os.listdir(method_path):
            source_id = video.split("_")[0]
            method_to_sources[method].add(source_id)
    return method_to_sources


def split_ids(ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    # seeded shuffle for reproducibility
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    # 70/15/15 split
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return ids[:n_train], ids[n_train:n_val], ids[n_val:]


def copy_images(src_dir, dst_dir, max_frames=10, seed=42):

    # copy at most max_frames images from src_dir to dst_dir. If more than max_frames images, sample without repalcement
    os.makedirs(dst_dir, exist_ok=True)
    images = os.listdir(src_dir)

    # subsample if there are too many frames
    if len(images) > max_frames:
        rng = random.Random(seed)
        images = rng.sample(images, max_frames)

    # copy each fram and prefix with the source folder name to avoid collisions
    for img in images:
        src_path = os.path.join(src_dir, img)
        new_name = f"{os.path.basename(src_dir)}_{img}"
        dst_path = os.path.join(dst_dir, new_name)
        shutil.copy(src_path, dst_path)


def prepare_split(real_dir, fake_dir, output_dir, max_frames=10, seed=42):
    # start from a clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # create the empty target folders
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # split real source ids into train/val/test
    real_ids = get_real_ids(real_dir)
    train_ids, val_ids, test_ids = split_ids(real_ids, seed=seed)
    train_set, val_set = set(train_ids), set(val_ids)

    # copy real frames into their corresponding split
    for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        for vid in ids:
            src = os.path.join(real_dir, vid)
            dst = os.path.join(output_dir, split, "real")
            copy_images(src, dst, max_frames=max_frames, seed=seed)

    # copy fake frames and route to same split as the real source video
    for method in os.listdir(fake_dir):
        method_path = os.path.join(fake_dir, method)
        if not os.path.isdir(method_path):
            continue
        # each src_dst folder routes by its source id
        for video in os.listdir(method_path):
            source_id = video.split("_")[0]
            if source_id in train_set:
                split = "train"
            elif source_id in val_set:
                split = "val"
            else:
                split = "test"
            src = os.path.join(method_path, video)
            dst = os.path.join(output_dir, split, "fake")
            copy_images(src, dst, max_frames=max_frames, seed=seed)
