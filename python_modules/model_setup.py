from imports import *

# load EfficientNet architecture with pretrained weights on imagenet
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# check how many features replace only the final layer 
num_features = model.classifier[1].in_features
# add dropout and replace the final layer from num_features to 2 features (0 - real, 1 - fake)
model.classifier = nn.Sequential(
    nn.BatchNorm1d(num_features),
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)
)

# train only the classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# move the model to correct hardware
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.0001
)