from imports import *

def evaluate(model, loader):
    # switch the nn into evaluation mode i.e to disable dropout
    model.eval()

    # lists to store predicted labels and true labels
    all_preds = []
    all_labels = []

    # we are not training, so do not compute gradients
    with torch.no_grad():
        for images, labels in loader:

            # move images to the same hardware as model
            images = images.to(device)
            # forward pass, feed images through nn
            outputs = model(images)

            # find the highest value along a dimension
            _, preds = torch.max(outputs, 1)
            # move predictions back to CPU and save them in lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # return true labels and predicted labels
    return all_labels, all_preds