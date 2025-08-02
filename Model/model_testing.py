# === Import necessary libraries ===
import torch
from sklearn.metrics import accuracy_score, confusion_matrix  # For evaluation metrics
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For prettier plots (confusion matrix)
from torchvision.models import efficientnet_b0  # Predefined EfficientNet model from torchvision

# === Import data loaders and dataset ===
from data.data_splitting_and_transforms import effnet_test_loader, cnn_test_loader  # Test loaders
from data.data_loading import dataset  # Full dataset (for class info)

# === Import your custom CNN model ===
from Model.model_training import TomatoCNN  # Your CNN architecture
from torchvision import models  # (Redundant here but might be used elsewhere)
import torch.nn as nn  # For modifying final classifier layer
import numpy as np  # For array operations

# === Device configuration (GPU if available, else CPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load class names from external file ===
with open("../app/class_names.txt") as f:
    class_names = [line.strip() for line in f]  # Read and clean each line (class label)

# === Determine number of classes from class names list ===
num_classes = len(class_names)

# === EfficientNet-B0 Loader ===
def get_effnet_model(num_classes):
    # Load EfficientNet-B0 without pretrained weights
    model = efficientnet_b0(pretrained=False)

    # Get number of input features of the original classifier
    in_features = model.classifier[1].in_features

    # Replace original classifier with a custom one matching our num_classes
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model  # Return the modified model

# === Generic evaluation function used for both models ===
def evaluate(model, model_name, model_path, test_loader):
    # Load model weights from saved file
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move model to the computation device
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    y_true, y_pred = [], []  # Lists to store ground truth and predicted labels

    with torch.no_grad():  # Disable gradient calculation (saves memory, faster)
        for images, labels in test_loader:
            # Move input and labels to the correct device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class by taking argmax over output logits
            preds = torch.argmax(outputs, dim=1)

            # Convert to CPU numpy arrays and store for evaluation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ {model_name} Accuracy: {acc * 100:.2f}%")

    # === Plot confusion matrix ===
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# === Entry point of the script ===
if __name__ == "__main__":
    # === Evaluate custom CNN model ===
    cnn_model = TomatoCNN(num_classes=num_classes)
    evaluate(cnn_model, model_name="Custom CNN", model_path="best_model.pth", test_loader=cnn_test_loader)

    # === Evaluate EfficientNet-B0 model ===
    effnet = get_effnet_model(num_classes)
    evaluate(effnet, model_name="Effnet", model_path="../app/best_effnet_model.pth", test_loader=effnet_test_loader)
