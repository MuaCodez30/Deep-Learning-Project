## Background

The original goal of our project was to develop an AI content detector capable of identifying whether a video was AI-generated or real. As the project evolved, we refined the scope by dividing the work into two related directions: AI-generated video detection and AI-generated image detection. For Milestone 1, this notebook focuses on the image-detection branch of the project and presents a complete PyTorch training pipeline for classifying images as either real or AI-generated.


As artificial intelligence technologies improve, distinguishing AI-generated content from authentic media has become increasingly challenging. This project investigates whether deep learning models can reliably detect AI-generated images and evaluates their effectiveness compared to human perception. The goal is to determine these  models can serve as a baseline for future research in automated synthetic media detection.

## Technicalities

The implementation uses the EfficientNet-B0 architecture with pretrained weights. The original classification layer is replaced with a new fully connected layer for binary classification (real vs. AI-generated images). At first, only the new classifier layer is trained, while keeping the pretrained backbone frozen. After three epochs, the last two convolutional blocks are unfrozen and fine-tuned to allow the model to adapt its learned features to the AI-image detection task. Various techniques like data augmentation, dropout, and transfer learning are applied to improve generalization and reduce overfitting.

## Running Requirements



NOTE: Unfortunately due to time constraints, I have updated the model but could not finish training it on time. However, the training accuracy should be around 89%. Code for running other metrics are in the notebook.
https://colab.research.google.com/drive/1bpjAwD-xrOxqQlNdesfn7ZIB312CDDdS?usp=sharing
