import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import unfreeze_deeper_blocks


def train_one_epoch(model, loader, optimizer, criterion, device):
    # train mode: enable dropout and batchnorm updates
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        # move batch to gpu
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # standard backprop step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate for the epoch average
        running_loss += loss.item()

    # return mean loss over batches
    return running_loss / len(loader)


def evaluate_loader(model, loader, device):
    # eval mode: no dropout, batchnorm uses running stats
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            # argmax over the 2 classes
            _, preds = torch.max(outputs, 1)
            # collect preds and labels
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return all_labels, all_preds


def make_class_weighted_criterion(train_targets, device):

    # count frames per class
    class_counts = np.bincount(train_targets)
    # weight = total / count per class (so minority class gets higher weight)
    class_weights = torch.tensor(
        [len(train_targets) / c for c in class_counts],
        dtype=torch.float,
    ).to(device)
    # weighted cross-entropy
    return nn.CrossEntropyLoss(weight=class_weights)


def train_with_finetuning(
    model,
    arch,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs=30,
    initial_lr=1e-4,
    finetune_lr=1e-5,
    unfreeze_at_epoch=3,
    patience=5,
    save_path=None,
):
  
    # adam over only the trainable params (initially its just the head)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
    )

    train_losses = []
    best_val_acc = 0.0
    counter = 0 # epochs since last improvement (for early stopping)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        # at epoch 3, unfreeze deeper layers and lower the lr
        if epoch == unfreeze_at_epoch:
            print(f"Unfreezing deeper blocks at epoch {epoch + 1}")
            unfreeze_deeper_blocks(model, arch)
            # rebuild the optimizer to include the newly unfrozen params
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=finetune_lr,
            )

        # train one epoch
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(loss)

        # evaluate on validation set
        val_labels, val_preds = evaluate_loader(model, val_loader, device)
        val_acc = float(np.mean(np.array(val_labels) == np.array(val_preds)))

        print(f"Epoch {epoch + 1}/{epochs}")
        print("Train loss:", loss)
        print("Val accuracy:", val_acc)
        print()

        # save the best model so far + early stopping bookkeeping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            counter += 1

        # stop if no improvement for `patience` epochs
        if counter >= patience:
            print("Early stopping triggered")
            break

    return train_losses
