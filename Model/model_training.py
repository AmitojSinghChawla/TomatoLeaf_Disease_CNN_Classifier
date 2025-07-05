import torch
from torch.utils.data import DataLoader
from data.data_loading import dataset
from Model.model_building import TomatoCNN  # Make sure this points to your model file
import torch.nn as nn
import torch.optim as optim
from data.data_splitting_and_transforms import cnn_train_loader,cnn_val_loader
def train_model(save_path="best_model.pth", num_epochs=10, batch_size=32, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_classes = len(dataset.classes)

    # ================
    # Model Definition
    # ================
    model = TomatoCNN(num_classes=num_classes).to(device)

    # Reset model weights to avoid reusing previously trained ones
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(reset_weights)

    # =========================
    # Loss, Optimizer, Scheduler
    # =========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler:
    # Lowers LR if validation loss doesn't improve for `patience` epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,)

    #rbose= ==================
    # Tracking Best Model
    # ==================
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Store losses for plotting later
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in cnn_train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(cnn_train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)

        # ==================
        # Validation
        # ==================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in cnn_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(cnn_val_loader)
        avg_val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)

        # Scheduler step — checks if validation loss improved
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2f}%")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_accuracy:.2f}%")

        # ================
        # Early Stopping
        # ================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)  # Save best model
            early_stop_counter = 0  # Reset counter if improved
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stopping count: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training complete. Best model saved.")
    return train_losses, val_losses  # For optional plotting later

# If run directly
if __name__ == '__main__':
    train_model()
