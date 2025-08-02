# === Import core libraries ===
import torch
from torch.utils.data import DataLoader  # Unused here but often used for loading datasets manually
from data.data_loading import dataset  # Needed to get number of classes
from Model.model_building import TomatoCNN  # Your custom CNN architecture
import torch.nn as nn  # Neural network layers and loss functions
import torch.optim as optim  # Optimizers like Adam

# === Import train/val data loaders ===
from data.data_splitting_and_transforms import cnn_train_loader, cnn_val_loader

# === Training Function ===
def train_model(save_path="best_model.pth", num_epochs=10, batch_size=32, patience=3):
    # Set device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of classes from dataset object
    num_classes = len(dataset.classes)

    # === Initialize CNN Model ===
    model = TomatoCNN(num_classes=num_classes).to(device)

    # === Reset all model weights before training ===
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):  # Some layers (like Linear, Conv) have this
            m.reset_parameters()
    model.apply(reset_weights)

    # === Define loss function and optimizer ===
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # === Learning rate scheduler ===
    # If validation loss plateaus, reduce learning rate by half
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # === Tracking Best Validation Loss for Early Stopping ===
    best_val_loss = float('inf')  # Initialize to large value
    early_stop_counter = 0  # Counter for early stopping condition

    # === Store loss history for visualization ===
    train_losses = []
    val_losses = []

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # === Train on batches ===
        for images, labels in cnn_train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute batch accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # === Epoch-level Train Metrics ===
        epoch_loss = running_loss / len(cnn_train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)

        # === Validation Phase ===
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No need to calculate gradients during validation
            for images, labels in cnn_val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        # === Epoch-level Validation Metrics ===
        avg_val_loss = val_loss / len(cnn_val_loader)
        avg_val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # === Print Epoch Summary ===
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2f}%")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_accuracy:.2f}%")

        # === Early Stopping Check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)  # Save best model so far
            early_stop_counter = 0  # Reset counter
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stopping count: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break  # Exit training loop early

    print("Training complete. Best model saved.")
    return train_losses, val_losses  # Useful for plotting learning curves later

# === Entry point ===
if __name__ == '__main__':
    train_model()  # Start training when script is run directly
