import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from data.data_loading import dataset  # or your get_datasets()
from data.data_splitting_and_transforms import effnet_train_loader, effnet_val_loader

# === Load Pretrained ResNet ===
def get_effnet_model(num_classes, pretrained=True, freeze=False):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classifier layer for tomato classification
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

# === Training Loop ===
def train_effnet_model(save_path="best_effnet_model.pth", num_epochs=10, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)
    model = get_effnet_model(num_classes=num_classes, pretrained=True, freeze=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in effnet_train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(effnet_train_loader)
        train_acc = 100 * correct / total

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in effnet_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(effnet_val_loader)
        avg_val_acc = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stopping count: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best model saved.")

# Save class names after training
with open("../app/class_names.txt", "w") as f:
    for c in dataset.classes:
        f.write(c + "\n")


if __name__ == "__main__":
    train_effnet_model()
