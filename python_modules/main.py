from imports import *
from dataset import *
from model_setup import *
from train import train
from evaluate import evaluate

epochs = 20
for epoch in range(epochs):

    # unfreeze the last two conv. blocks starting from epoch 3 (finetuning the deeper layers)
    if epoch == 3:
        for param in model.features[-1].parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
        )
    
    loss = train(model, train_loader, optimizer, criterion)

    # training metrics
    train_labels, train_preds = evaluate(model, train_loader)
    train_accuracy = np.mean(np.array(train_labels) == np.array(train_preds))

    # test metrics
    test_labels, test_preds = evaluate(model, test_loader)
    test_accuracy = np.mean(np.array(test_labels) == np.array(test_preds))

    print(f"Epoch {epoch+1}/{epochs}")
    print("Train loss:", loss)
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
    print()

labels, preds = evaluate(model, test_loader)
print(classification_report(labels, preds))

cm = confusion_matrix(labels, preds)
print(cm)

torch.save(model.state_dict(), "ai_image_detector.pth")

model = models.resnet50()

model.fc = nn.Linear(model.fc.in_features,2)

model.load_state_dict(torch.load("ai_fake_detector.pth"))

model.eval()