import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# === 1. Path to your dataset ===
# This is the folder that contains subfolders like Tomato___Early_blight, etc.
data_dir = 'PlantVillage'  # ← CHANGE THIS to your actual folder path

# === 2. Image Preprocessing Transformations ===
# - Resize all images to 256x256
# - Convert image from PIL to tensor format (C x H x W), and normalize pixel values to [0, 1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# === 3. Load Images with Labels ===
# ImageFolder:
# - Automatically assigns labels based on folder names
# - Loads images using the transform defined above
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# === 4. Create DataLoader to Feed Images in Batches ===
# - batch_size = 32 images per batch
# - shuffle = True for randomizing data order (important for training)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 5. Test One Batch ===
# Get a single batch (images + labels) to verify everything works
images, labels = next(iter(dataloader))

# === 6. Print Info ===
print(f"Total images in dataset: {len(dataset)}")
print(f"Classes found: {dataset.classes}")  # Folder names are treated as class labels
print(f"One batch shape: {images.shape}")   # Should be [32, 3, 256, 256]
print(f"Labels in batch: {labels[:5]}")     # Print first few labels

for image,label in dataset:
    for i in range(10):
        plt.imshow(image[i])
        plt.title(dataset[labels[i]])
        plt.axis('off')
        plt.show()