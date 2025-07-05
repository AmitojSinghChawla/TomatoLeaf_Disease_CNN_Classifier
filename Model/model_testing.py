import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import efficientnet_b0

from data.data_splitting_and_transforms import effnet_test_loader,cnn_test_loader
from data.data_loading import dataset
from Model.model_training import TomatoCNN  # CNN model
from torchvision import models
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("../app/class_names.txt") as f:
    class_names = [line.strip() for line in f]

num_classes = len(class_names)

# === EfficientNet-B0 Loader ===
def get_effnet_model(num_classes):
    model = efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# === Evaluation Function (shared) ===
def evaluate(model, model_name, model_path,test_loader):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ {model_name} Accuracy: {acc * 100:.2f}%")

    # Confusion Matrix
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


if __name__ == "__main__":
    # === Evaluate CNN Model ===
    cnn_model = TomatoCNN(num_classes=num_classes)
    evaluate(cnn_model, model_name="Custom CNN", model_path="best_model.pth",test_loader=cnn_test_loader)

    # === Evaluate Effnet Model ===
    effnet = get_effnet_model(num_classes)
    evaluate(effnet, model_name="Effnet", model_path="../app/best_effnet_model.pth", test_loader=effnet_test_loader)
