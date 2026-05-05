# Deepfake Detection on FaceForensics++

CSCI 4701, Deep Learning, Spring 2026. Final project (Milestone 2).

Team: Bilal Hasanov, Muaataz Abdulhakeem Ismaeel.

## 1. Project Goal

We try to figure out if deep CNNs can reliably tell apart real face videos from AI-manipulated ones, using the FaceForensics++ extracted-frames dataset. FF++ is video-based but most pretrained image classifiers work on individual frames, so the project is split into three phases:

1. Architecture comparison. We fine-tune three ImageNet-pretrained backbones (ResNet-50, EfficientNet-B0, EfficientNet-V2-S) on the FF++ frame-level binary task and pick the best one.
2. Frame-level inference and video-level aggregation. We take the winning model, classify every frame of a test video, and combine the per-frame predictions into one video-level decision using either majority voting or average softmax.
3. Backbone as a feature extractor with classical ML. We drop the trained classification head, embed every frame to a 1280-D vector with the frozen backbone, mean-pool the embeddings within each video, and train SVM, MLP, and Logistic Regression on those video-level features.

## 2. Repository Structure

The repo has three top-level folders and a couple of files:

- `notebooks/milestone2.ipynb` is the graded notebook. It contains no algorithmic logic, it just imports from `src/` and orchestrates the experiments.
- `notebooks/architecture_training/` contains the three Kaggle notebooks where the actual fine-tuning happened. Their outputs (epoch logs, classification reports, confusion matrices) are preserved inside the notebooks.
- `src/` is the reusable Python package. It contains `config.py` (paths and hyperparameters), `data_split.py` (the FF++ train/val/test splitting code), `dataset.py` (transforms and the custom dataset class), `model.py` (all three architectures plus a feature-extractor variant of EfficientNet-V2-S), `training.py` (the training loop), `inference.py` (frame inference and video aggregation), `features.py` (video-level feature extraction), `classifiers.py` (SVM, MLP, LogReg), and `evaluation.py` (metrics and confusion matrix plots).
- `README.md` is this file. `requirements.txt` lists dependencies.

## 3. Team Responsibilities

Bilal Hasanov: built the FF++ data preparation pipeline (source-level train/val/test split in `src/data_split.py`), fine-tuned all three architectures on Kaggle (the three notebooks under `notebooks/architecture_training/`), implemented Phase 2 (frame-level inference plus video-level aggregation with majority voting and average softmax), implemented Phase 3 (feature extraction and classical ML classifiers), refactored the Kaggle notebooks into the modular `src/` package, and authored the merged graded notebook and this README.

Muaataz Abdulhakeem Ismaeel: worked on the parallel AI-video detection branch of the project, reviewed the modular refactor and verified the notebook runs end-to-end on Colab, contributed to the limitations and analysis sections.

Both team members are expected to be able to defend any part of the submission.

## 4. How to Run

The notebook is designed to run end-to-end on Google Colab with a GPU runtime (T4 is enough). The submission ZIP and the GitHub repo only contain the code. The data and the trained checkpoints live elsewhere because they are too large to ship with the repo:

- The prepared FF++ split is published as a public Kaggle dataset at https://www.kaggle.com/datasets/delno002/split-ffpp. The notebook pulls it via `kagglehub` on first run.
- The three trained `.pth` checkpoints are in a public Google Drive folder. The link is below.

### 4.1 One-time setup

First, upload the repo to your Google Drive. The notebook expects it at `MyDrive/csci-4701-deepfake-detection/`. If you put it somewhere else, edit the `%cd` line in Section 1 of the notebook to point at your actual path.

Second, get the trained checkpoints. Open this Drive folder:

PASTE-YOUR-DRIVE-LINK-HERE

Click "Add shortcut to Drive" and place the shortcut at the root of your Drive, so it shows up at `MyDrive/Downloadables/`. Using a shortcut means it does not consume your own Drive quota. The folder contains three files: `resnet50_weights.pth`, `efficientnet_b0_weights.pth`, and `efficientnet-v2-s_weights.pth`.

Third (optional), the notebook contains Kaggle credentials for a throwaway account, so you do not need to do anything with Kaggle yourself. If for some reason that account stops working, generate your own token at kaggle.com/settings (API section, "Create New Token") and replace the username/key in Section 1 cell 4 of the notebook.

### 4.2 Running the notebook

In your Drive, navigate to `csci-4701-deepfake-detection/notebooks/`, right-click `milestone2.ipynb`, and pick "Open with -> Google Colaboratory". Do not open it directly from the GitHub link, the notebook reads `src/` from your Drive folder.

Set runtime to GPU. Run the cells in order. Section 1 mounts Drive, installs `kagglehub`, and configures Kaggle credentials. Section 2 downloads the FF++ dataset from Kaggle (cached after the first run) and points the config at the checkpoints in your Drive shortcut.

End-to-end runtime is roughly 10 to 20 minutes on a Colab T4 on a fresh runtime. Most of that is the initial Kaggle download and the feature-extraction pass over the training split (Section 7.2). Subsequent runs in the same session are much faster because everything is cached.

### 4.3 Running locally

Clone the repo, install requirements with `pip install -r requirements.txt`, download the three checkpoints from the Drive link above, and put your `kaggle.json` at `~/.kaggle/kaggle.json` (with `chmod 600`). In the notebook, replace the Drive-mount cell with a `%cd` to your local repo path, and in Section 2 change `DOWNLOADABLES` to wherever you put the `.pth` files. Then `jupyter lab notebooks/milestone2.ipynb`.

## 5. Reproducibility

All randomness in the classical classifiers (Section 7) is controlled by `random_state=42` in `src/classifiers.py`. Fine-tuning uses `SEED=42` in `src/config.py`. Each fine-tuning run takes several hours on a T4, so the graded notebook does not retrain from scratch. It loads the saved checkpoints. The training notebooks under `notebooks/architecture_training/` retain their full epoch-by-epoch logs as preserved outputs, so the training history is auditable.

Frame counts and split sizes are printed at the top of the notebook so you can immediately verify the data is mounted correctly.

## 6. Main Results

The exact numbers are produced by the notebook itself when re-run on Colab. The numbers below summarize what is recorded in the training notebooks:

For frame-level test accuracy in the architecture comparison, EfficientNet-V2-S reaches around 92% accuracy with macro F1 of about 0.89 and real-class recall around 0.89, and is the architecture we selected for the video-level experiments. ResNet-50 is a close second at around 90% accuracy and macro F1 of 0.87. EfficientNet-B0 is meaningfully behind at around 85% accuracy.

Once we move to the video level using EfficientNet-V2-S, the test accuracy increases. A single confidently-misclassified frame no longer counts as a mistake at the video level if the rest of the video votes the other way. Average-softmax aggregation is consistently at least as good as majority voting on this split. The two strategies agree on the vast majority of videos.

For Phase 3, SVM, MLP, and Logistic Regression trained on the mean-pooled video-level embeddings perform competitively with the fine-tuned classification head. The 1280-D post-pooling embedding space is approximately linearly separable for the FF++ binary task: Logistic Regression is competitive with SVM-RBF and MLP, and we did not see a large jump going from a linear classifier to a non-linear one.

## 7. Analysis

### 7.1 What worked

Architecture choice matters more than expected. The gap between EfficientNet-V2-S (around 92%) and EfficientNet-B0 (around 85%) on FF++ is bigger than what you would naively predict from "just" a larger model. Capacity matters here.

Soft averaging is consistently at least as good as hard voting because voting throws away the model's confidence. A prediction at 0.51 and one at 0.99 count the same under voting. Soft averaging keeps that signal, which helps on the small subset of videos where per-frame predictions are genuinely mixed.

Most test videos are easy at the video level. The two aggregation strategies agree on most videos and only diverge on borderline cases, where soft averaging is mildly more reliable.

Classical classifiers on top of the frozen backbone are competitive with the fine-tuned head. SVM, MLP, and LogReg all reach broadly similar accuracy when fed mean-pooled embeddings. The heavy lifting is being done by the representation, not the final classifier.

### 7.2 What did not work or what we would do next

Train and test both come from FF++. We did not run a cross-dataset evaluation (for example training on FF++ and testing on Celeb-DF or DFDC) within the project timeline. A high test number on FF++ is therefore not a strong claim about generalization to other forgery pipelines.

Mean pooling discards temporal structure. Averaging frame embeddings ignores their order. That is fine for FF++, where every frame of a manipulated video is independently manipulated, but it would be a poor choice for forgeries whose tell-tale signature is temporal inconsistency (flicker between adjacent frames, identity drift). A recurrent or attention-based aggregator would be needed there.

Class imbalance was only partially addressed. During training we used a class-weighted CrossEntropyLoss, but we did not apply oversampling or threshold tuning at inference time. The minority (real) class still sees lower recall in some methods.

Hyperparameter search was minimal. We used reasonable defaults for SVM-RBF, MLP, and LogReg rather than running a grid search. The relative ordering of these classifiers might shift slightly with proper tuning, though the high-level conclusion would not.

Detection and alignment errors propagate silently. We process precomputed face crops, not raw videos. A few of the misclassified videos we spot-checked had crops where the face was poorly centered.

### 7.3 Bottom line

EfficientNet-V2-S is the strongest of the three backbones we tested. Aggregating per-frame predictions to per-video decisions improves accuracy, and soft averaging is the most cost-effective method we tried. A frozen backbone plus a classical classifier is competitive with the fine-tuned head. The biggest practical gains, if we were continuing this project, would come from cross-dataset evaluation, a temporally-aware aggregator, and explicit handling of class imbalance at inference time.

## 8. Acknowledgements

FaceForensics++: Rossler et al., FaceForensics++: Learning to Detect Manipulated Facial Images, ICCV 2019.
EfficientNet-V2: Tan and Le, EfficientNetV2: Smaller Models and Faster Training, ICML 2021.
EfficientNet: Tan and Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019.
ResNet: He et al., Deep Residual Learning for Image Recognition, CVPR 2016.

We used `torchvision`'s pretrained `efficientnet_v2_s`, `efficientnet_b0`, and `resnet50` weights as starting points for fine-tuning.
