import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torch import nn
from torchvision.models import efficientnet_b0

# Deterministic
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]
num_classes = len(class_names)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(num_classes):
    model_path = os.path.join(os.path.dirname(__file__), "best_effnet_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_image(image_input, model):
    image = transform(image_input).unsqueeze(0)

    disease_metadata = {
        "Bacterial Spot": {
            "description": "Dark spots with yellow halos, affecting leaves, stems, and fruit.",
            "treatment": "Use copper-based bactericides and avoid overhead watering."
        },
        "Early Blight": {
            "description": "A common fungal disease causing dark spots with concentric rings on leaves.",
            "treatment": "Apply copper-based fungicide every 7-10 days and remove affected leaves."
        },
        "Late Blight": {
            "description": "Serious fungal disease that can destroy entire crops rapidly.",
            "treatment": "Immediate fungicide treatment and remove affected plants."
        },
        "Leaf Mold": {
            "description": "Yellow patches on upper surfaces with fuzzy growth underneath.",
            "treatment": "Reduce humidity, improve greenhouse ventilation, and remove lower leaves."
        },
        "Septoria Leaf Spot": {
            "description": "Small circular spots with dark borders and light gray centers.",
            "treatment": "Mulch around plants, prune lower branches, and apply preventive fungicide."
        },
        "Spider Mites": {
            "description": "Tiny spider mites causing yellowing and stippling on leaves.",
            "treatment": "Use miticides and maintain proper humidity."
        },
        "Target Spot": {
            "description": "Dark, target-like spots on leaves and stems.",
            "treatment": "Remove infected foliage and apply fungicides."
        },
        "Tomato Yellow Leaf Curl Virus": {
            "description": "Virus causing yellowing and curling of leaves, stunting growth.",
            "treatment": "Control whitefly vectors and remove infected plants."
        },
        "Tomato Mosaic Virus": {
            "description": "Mosaic pattern on leaves causing mottled colors and deformation.",
            "treatment": "Use resistant varieties and disinfect tools."
        },
        "Healthy": {
            "description": "Your tomato plant is healthy! No signs of disease detected.",
            "treatment": "Continue regular care and monitoring."
        }
    }

    # Map your exact class names to clean names
    class_name_map = {
        "Tomato_Bacterial_spot": "Bacterial Spot",
        "Tomato_Early_blight": "Early Blight",
        "Tomato_Late_blight": "Late Blight",
        "Tomato_Leaf_Mold": "Leaf Mold",
        "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider Mites",
        "Tomato__Target_Spot": "Target Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
        "Tomato_healthy": "Healthy",
    }

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        predicted_idx = predicted.item()
        confidence = probs[0][predicted_idx].item()

        raw_class_name = class_names[predicted_idx]

        # Map to clean disease name
        disease_name = class_name_map.get(raw_class_name, "Unknown")

        metadata = disease_metadata.get(disease_name, {"description": "N/A", "treatment": "N/A"})

        return {
            "class": disease_name,
            "confidence": round(confidence * 100, 2),
            "description": metadata["description"],
            "treatment": metadata["treatment"]
        }
