import torch
import torch.nn as nn
import torchvision.models as models

class ClothingTypeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # remove fc
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.fc(x)
        return x  # logits