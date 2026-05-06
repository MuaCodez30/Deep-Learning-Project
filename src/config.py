from pathlib import Path
import torch

# on Colab, mount Google Drive and set this to BASE_DIR = Path("/content/drive/MyDrive/deepfake_detection")
BASE_DIR = Path("/content/drive/MyDrive/deepfake_detection")


# the prepared train/val/test split 
DATA_DIR = BASE_DIR / "ffpp_split"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# saved checkpoints from kaggle training
MODELS_DIR = BASE_DIR / "models"
CKPT_PATH = MODELS_DIR / "best_model_final.pth"  # default for video-level experiments

# where this notebook writes its output (csvs, pickles, plots)
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class mapping
# torchvision.datasets.ImageFolder assigns labels alphabetically, so fake = 0, real = 1
CLASS_TO_IDX = {"fake": 0, "real": 1}
IDX_TO_CLASS = {0: "fake", 1: "real"}


# Runtime hyperparameters
SEED = 42                  
BATCH_SIZE = 32
NUM_WORKERS = 2
IMG_SIZE = 224             
MAX_FRAMES_PER_VIDEO = 10  

# imagenet normalization stats (the backbones are imagenet-pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# gpu if available, otherwise cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
