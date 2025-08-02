# === Import core libraries ===
import torch                                    # PyTorch for model and tensor operations
import torchvision.transforms as transforms    # For preprocessing images
from PIL import Image                          # To open and manipulate images
import os                                      # To handle file paths
from torch import nn                           # For defining neural network layers
from torchvision.models import efficientnet_b0 # Load EfficientNet-B0 architecture

# === Set deterministic behavior for reproducibility ===
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Load class names from text file ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # Directory where this script lives
with open(os.path.join(BASE_DIR, "class_names.txt")) as f:   # Read class names line-by-line
    class_names = [line.strip() for line in f]               # Strip newline characters

num_classes = len(class_names)  # Total number of output classes for the model

# === Define image preprocessing pipeline ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),                            # Resize to 256x256
    transforms.ToTensor(),                                    # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],          # Normalize using ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# === Import PyTorch quantization module ===
import torch.quantization  # Enables dynamic quantization to reduce model size for inference

# === Load model architecture and weights ===
def load_model(num_classes):
    model_path = os.path.join(os.path.dirname(__file__), "best_effnet_model.pth")  # Model file path
    if not os.path.exists(model_path):                                             # Safety check
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = efficientnet_b0(pretrained=False)                    # Load architecture without pretrained weights
    in_features = model.classifier[1].in_features                # Input size of final FC layer
    model.classifier[1] = nn.Linear(in_features, num_classes)   # Replace FC layer with one for your num_classes

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load trained weights

    # ✅ Dynamically quantize only the linear layers for memory efficiency
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    model.eval()     # Set model to evaluation mode
    return model     # Return the ready-to-use model

# === Predict disease from image input ===
def predict_image(image_input, model):
    image = transform(image_input).unsqueeze(0)  # Apply preprocessing and add batch dimension

    # === Dictionary with disease metadata ===
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

    # === Mapping internal class labels to clean disease names ===
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

    with torch.no_grad():                      # Disable gradient computation for inference
        outputs = model(image)                 # Get model predictions (logits)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
        _, predicted = torch.max(probs, 1)     # Get class index with highest probability
        predicted_idx = predicted.item()       # Convert tensor to int
        confidence = probs[0][predicted_idx].item()  # Get confidence score of predicted class

        raw_class_name = class_names[predicted_idx]   # Look up class name from index

        # Get clean name from internal class name
        disease_name = class_name_map.get(raw_class_name, "Unknown")

        # Get metadata (description & treatment) for the disease
        metadata = disease_metadata.get(disease_name, {"description": "N/A", "treatment": "N/A"})

        # Return a dictionary of prediction results
        return {
            "class": disease_name,
            "confidence": round(confidence * 100, 2),
            "description": metadata["description"],
            "treatment": metadata["treatment"]
        }
