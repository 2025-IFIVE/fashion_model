import torch
import torch.nn as nn
import torchvision.models as models

class FashionAttributeNet(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        # ResNet18 백본 사용 (가중치는 나중에 로드)
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 fc 제거
        self.feature_dim = resnet.fc.in_features  # 512

        # 속성별 분기 헤드
        self.attribute_heads = nn.ModuleDict()
        for attr_name, num_classes in num_classes_dict.items():
            self.attribute_heads[attr_name] = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)

        output = {}
        for attr_name, head in self.attribute_heads.items():
            output[attr_name] = head(features)  # logits
        return output
