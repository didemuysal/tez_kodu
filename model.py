from torch import nn
from torchvision import models

def create_brain_tumour_model(model_name: str = 'resnet50', pretrained: bool = True) -> nn.Module:
    model_name_lower = model_name.lower()
    if model_name_lower == 'resnet18':
        bt_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name_lower == 'resnet50':
        bt_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError("Choose 'resnet18' or 'resnet50'")
    
    num_features = bt_model.fc.in_features

    for param in bt_model.parameters():
        param.requires_grad = False

    bt_model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 3))

    return bt_model