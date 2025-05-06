import torch
from models.classifier_model import ClothingTypeClassifier
from models.fashion_model_all import FashionAttributeNet
from models.style_model import StyleClassifier
from dataset.kfashion_dataset_all import ATTRIBUTE_LOOKUP, ATTRIBUTE_SCHEMA
from dataset.style_dataset import STYLE_CATEGORIES, SUBSTYLE_CATEGORIES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    type_model = ClothingTypeClassifier(num_classes=4).to(DEVICE)
    type_model.load_state_dict(torch.load("weights/clothing_classifier.pth", map_location=DEVICE))
    type_model.eval()

    num_classes_dict = {}
    for part in ATTRIBUTE_SCHEMA:
        for attr in ATTRIBUTE_SCHEMA[part]["single"]:
            num_classes_dict[f"{part}_{attr}"] = len(ATTRIBUTE_LOOKUP[part][attr]) + 1
        for attr in ATTRIBUTE_SCHEMA[part]["multi"]:
            num_classes_dict[f"{part}_{attr}"] = len(ATTRIBUTE_LOOKUP[part][attr]) + 1
    attr_model = FashionAttributeNet(num_classes_dict).to(DEVICE)
    attr_model.load_state_dict(torch.load("weights/saved_model_all.pth", map_location=DEVICE))
    attr_model.eval()

    style_model = StyleClassifier(
        num_styles=len(STYLE_CATEGORIES),
        num_substyles=len(SUBSTYLE_CATEGORIES)
    ).to(DEVICE)
    style_model.load_state_dict(torch.load("weights/style_classifier.pth", map_location=DEVICE))
    style_model.eval()

    return type_model, attr_model, style_model