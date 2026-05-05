
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

# class mapping and image preprocessing constants live in config
from .config import CLASS_TO_IDX, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform():
    return transforms.Compose([
        # resize to 224x224 (architecture input size)
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        # data augmentation using flip, color jitter, crop 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),

        # to tensor and normalize with imagenet stats
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        # resize without augmentation for testing
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FFPPFrameDataset(Dataset):
    """
    fake: 006_002_000.png then the video_id is 006_002
    real: 006_002.png then the video_id is 006

    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # list of (path, label, class_name, video_id, filename) tuples
        self.samples = []

        # walk both class folders
        for class_name in ["fake", "real"]:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            # sorted so the order is deterministic across runs
            for img_path in sorted(class_dir.glob("*.png")):
                filename = img_path.name

                # split filename on '_' to extract the video id
                parts = img_path.stem.split("_")

                if class_name == "fake":
                    video_id = "_".join(parts[:2])
                else:
                    video_id = parts[0]

                # convert class name to integer label for fake=0 and real=1
                label = CLASS_TO_IDX[class_name]
                self.samples.append(
                    (str(img_path), label, class_name, video_id, filename)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # unpack the sample tuple
        img_path, label, _class_name, video_id, filename = self.samples[idx]

        # load the image and convert to RGB since some pngs can be grayscale
        image = Image.open(img_path).convert("RGB")

        # apply transforms 
        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_path, video_id, filename


def count_split(base_dir):
    base = Path(base_dir)
    counts = {}
    # walk all three splits
    for split in ["train", "val", "test"]:
        split_dir = base / split
        if not split_dir.exists():
            continue
        counts[split] = {}
        # count pngs in each class folder
        for cls in ["real", "fake"]:
            folder = split_dir / cls
            counts[split][cls] = (
                len(list(folder.rglob("*.png"))) if folder.exists() else 0
            )
    return counts
