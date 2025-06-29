import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.models import resnet18


# Load class labels
class_names = [
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# Preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load Trained Model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_resnet_model.pth")
    print("Final model path:", model_path)
    print("Exists:", os.path.exists(model_path))

    num_classes = len(class_names)
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Make Prediction
def predict_image(image_file, model):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0)  # Shape: [1, 3, 256, 256]
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
