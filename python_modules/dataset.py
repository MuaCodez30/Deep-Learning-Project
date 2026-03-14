from imports import *

# get the path to and download the CIFAKE dataset from kaggle, containing 60,000 real and 60,000 AI generated images using Stable Diffusion v4
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

print("Path to dataset files:", path)

# apply random transforms to the training data to increase the diversity of the dataset and help prevent overfitting, while keeping the test data unchanged for accurate evaluation of model performance
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# apply transforms to test data, 224x244 resizing and specific normalization due to ImageNet standards
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_dataset = datasets.ImageFolder(
    root=f"{path}/train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{path}/test",
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)