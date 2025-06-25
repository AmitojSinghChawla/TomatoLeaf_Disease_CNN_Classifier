
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from data.data_loading import dataset
#We are defining a neural network by creating a class Net that inherits from nn.Module.
# It includes two convolutional layers with ReLU and max pooling, followed by three fully connected layers.
# In the forward method, we pass the input through these layers, flattening it before the dense layers.
# Finally we create an instance of this model as net.

import torch.nn as nn
import torch.nn.functional as F

from data.data_loading import dataset

num_classes=len(dataset.classes)
# Define a Convolutional Neural Network class
class TomatoCNN(nn.Module):
    def __init__(self, num_classes):
        # Call the constructor of the parent class nn.Module
        super(TomatoCNN, self).__init__()

        # =======================
        # BLOCK 1: Conv + Pool
        # =======================
        # Input: RGB image (3 channels), size (256 x 256)
        # Applies 32 filters of size 3x3 to extract basic features
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)  # Output: (32, 254, 254)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                # Downsample to (32, 127, 127)

        # =======================
        # BLOCK 2: Conv + Pool
        # =======================
        # Further extracts more complex patterns using 64 filters
        self.conv2 = nn.Conv2d(32, 64, 3)                                       # Output: (64, 125, 125)
        self.pool2 = nn.MaxPool2d(2)                                            # Downsample to (64, 62, 62)

        # =======================
        # BLOCK 3: Conv + Pool
        # =======================
        # Still using 64 filters, extracts even higher-level features
        self.conv3 = nn.Conv2d(64, 64, 3)                                       # Output: (64, 60, 60)
        self.pool3 = nn.MaxPool2d(2)                                            # Downsample to (64, 30, 30)

        # =======================
        # BLOCK 4: Conv + Pool
        # =======================
        # Final conv block before flattening
        self.conv4 = nn.Conv2d(64, 64, 3)                                       # Output: (64, 28, 28)
        self.pool4 = nn.MaxPool2d(2)                                            # Downsample to (64, 14, 14)

        # ===========================
        # Fully Connected Classifier
        # ===========================
        self.flatten = nn.Flatten()  # Converts 3D output to 1D vector: (64 × 14 × 14) = 12544

        # Dense layer: from 12544 → 64 neurons
        self.fc1 = nn.Linear(64 * 14 * 14, 64)

        # Output layer: from 64 → num_classes (e.g., 5 tomato disease classes)
        self.fc2 = nn.Linear(64, num_classes)

    # ===================
    # Forward Pass Logic
    # ===================
    def forward(self, x):
        # Apply Conv → ReLU → MaxPool for each block
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # Flatten the output to feed into dense layers
        x = self.flatten(x)

        # Hidden dense layer with ReLU
        x = F.relu(self.fc1(x))

        # Output layer (raw logits for classification)
        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TomatoCNN(num_classes=num_classes)  # Change if your dataset has different #classes
model = model.to(device)

# === LOSS FUNCTION ===
# For multi-class classification. Accepts raw logits + integer class labels.
criterion = nn.CrossEntropyLoss()

# === OPTIMIZER ===
# Adam is fast and handles learning rate adaptively
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Optional: print model summary
# print(model)
