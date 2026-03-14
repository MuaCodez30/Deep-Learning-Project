from imports import *

def train(model, loader, optimizer, criterion):
    
    model.train()
    running_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # forward pass, feed images through nn
        outputs = model(images)

        loss = criterion(outputs, labels)

        # reset gradient so they dont accumulate across batches
        optimizer.zero_grad()
        # gradient of the loss with respect to each weight
        loss.backward()
        # Adam optimizer updates the weight
        optimizer.step()

        # loss.item() converts the loss tensor into a Python number to calc avg loss later
        running_loss += loss.item()
    # return average loss per batch/epoch
    return running_loss / len(loader)