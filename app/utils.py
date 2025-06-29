import torch
import torchvision.transforms as transforms
from PIL import Image
from Model.model_building import TomatoCNN  # or import your ResNet if you're using that
from data.data_loading import dataset

# Load class labels
class_names = dataset.classes

# Define image preprocessing (MUST match training transforms)
transform = transforms.Compose([
    transforms.Resize((256, 256)),     # Resize to model input
    transforms.ToTensor(),             # Convert PIL image to Tensor
])

# =====================
# Load Trained Model
# =====================
def load_model(model_path="best_renet_model.pth"):
    num_classes = len(class_names)
    model = TomatoCNN(num_classes=num_classes)  # or ResNet if you used it
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Turn off dropout/batchnorm in inference
    return model

# =====================
# Make Prediction
# =====================
def predict_image(image_file, model):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Turn off gradient computation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
