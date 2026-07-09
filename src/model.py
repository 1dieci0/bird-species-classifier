from torchvision.models import resnet18
import torch.nn as nn

def create_model():

    model = resnet18(weights="DEFAULT")

    model.fc = nn.Linear(
        model.fc.in_features,
        200
    )

    return model
