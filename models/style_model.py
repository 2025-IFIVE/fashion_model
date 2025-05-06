import torch
import torch.nn as nn
import torchvision.models as models

class StyleClassifier(nn.Module):
    def __init__(self, num_styles, num_substyles):
        super().__init__()
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # remove fc
        feature_dim = base_model.fc.in_features

        self.style_head = nn.Linear(feature_dim, num_styles)
        self.substyle_head = nn.Linear(feature_dim, num_substyles)

    def forward(self, x):
        x = self.backbone(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        return {
            "style": self.style_head(x),
            "substyle": self.substyle_head(x)
        }