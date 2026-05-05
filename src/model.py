import torch
import torch.nn as nn
from torchvision import models

def _build_efficientnet_b0(num_classes: int = 2) -> nn.Module:
    # build efficientnet-b0 with no weights (we load weights from checkpoint later)
    net = models.efficientnet_b0(weights=None)
    # number of features going into the classifier head
    in_features = net.classifier[1].in_features
    # replace the head with our 2-class one (batchnorm + dropout + linear)
    net.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return net


def _build_efficientnet_v2_s(num_classes: int = 2) -> nn.Module:
    # same idea for efficientnet-v2-s
    net = models.efficientnet_v2_s(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return net


def _build_resnet50(num_classes: int = 2) -> nn.Module:
    # resnet stores its head under .fc instead of .classifier
    net = models.resnet50(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return net

# lookup table for the three architectures
_BUILDERS = {
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnet_v2_s": _build_efficientnet_v2_s,
    "resnet50": _build_resnet50,
}


def build_model(arch: str, num_classes: int = 2, pretrained: bool = False) -> nn.Module:

    if arch not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture '{arch}'. Expected one of {list(_BUILDERS)}."
        )

    # no pretrained weights - just return the empty arch with our head
    if not pretrained:
        return _BUILDERS[arch](num_classes=num_classes)

    # otherwise load imagenet pretrained weights
    weights_map = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
    }
    ctor_map = {
        "efficientnet_b0":models.efficientnet_b0,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "resnet50": models.resnet50,
    }
    net = ctor_map[arch](weights=weights_map[arch])

    # replace the original 1000-class imagenet head with 2-class head
    if arch == "resnet50":
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    return net


def unfreeze_deeper_blocks(model, arch: str) -> None:

    if arch == "resnet50":
        # unfreeze the last three resnet stages
        for layer in [model.layer2, model.layer3, model.layer4]:
            for param in layer.parameters():
                param.requires_grad = True
    elif arch in ("efficientnet_b0", "efficientnet_v2_s"):
        # unfreeze the last seven efficientnet blocks
        for param in model.features[-7:].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown architecture '{arch}'")




def _load_checkpoint(net: nn.Module, ckpt_path, device) -> nn.Module:
    # load the file (state dict or training-checkpoint dict)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    # extract the state dict (handles both formats)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # copy weights into the model
    net.load_state_dict(state_dict, strict=True)
    return net


def load_model(arch: str, ckpt_path, device, num_classes: int = 2) -> nn.Module:

    # build empty arch, load weights, move to device, set eval mode
    net = build_model(arch=arch, num_classes=num_classes, pretrained=False)
    net = _load_checkpoint(net, ckpt_path, device)
    net.to(device)
    net.eval()
    return net


def build_classifier(ckpt_path, device, num_classes: int = 2) -> nn.Module:
    return load_model("efficientnet_v2_s", ckpt_path, device, num_classes)


class EfficientNetV2SFeatureExtractor(nn.Module):

    def __init__(self, ckpt_path=None, num_classes: int = 2, device="cpu"):
        super().__init__()
        # build the full network so we can load the checkpoint
        net = _build_efficientnet_v2_s(num_classes=num_classes)
        if ckpt_path is not None:
            net = _load_checkpoint(net, ckpt_path, device)

        # keep only the convolutional backbone and the avgpool layer
        self.features = net.features
        self.avgpool = net.avgpool

    def forward(self, x):
        x = self.features(x) # [B, C, H, W]
        x = self.avgpool(x) # [B, C, 1, 1]
        x = torch.flatten(x, 1) # [B, C] (the 1280-d embedding)
        return x
