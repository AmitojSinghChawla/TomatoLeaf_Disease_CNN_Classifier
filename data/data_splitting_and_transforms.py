#__________Spliiting Data into Train,Validation and Test Sets
import torch
from data.data_loading import dataset
from torch.utils.data import random_split

def get_pytorch_dataset_partitions(dataset, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    assert train_split + val_split + test_split == 1, "Splits must sum to 1"

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size  # ensures no rounding issues

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_ds, val_ds, test_ds

# In PyTorch, many functions that involve randomness (like splitting datasets)
# use an internal random number generator. By default, this generator produces
# different results each time you run the script.

# torch.Generator() allows us to create our own random number generator.
# By calling .manual_seed(42), we set a fixed "starting point" for randomness.
# This ensures that the train/validation/test split is the same every time the script is run.

# This is important for reproducibility — it helps us compare models,
# debug consistently, and avoid confusion caused by random variations in data.

train_ds , val_ds , test_ds = get_pytorch_dataset_partitions(dataset)

from torchvision import transforms

# Define a sequence of preprocessing steps using transforms.Compose
# This is similar to tf.keras.Sequential for image preprocessing in TensorFlow

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(72),
    transforms.ToTensor(),  # Always last
])

# Only resize + convert for validation and test sets
basic_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


# The underlying dataset object is shared, so we set its .transform for each subset
train_ds.dataset.transform = train_transform   # Augment only training data
val_ds.dataset.transform = basic_transform     # Clean for validation
test_ds.dataset.transform = basic_transform    # Clean for testing


from torch.utils.data import DataLoader

# === Train DataLoader ===
train_loader = DataLoader(
    dataset=train_ds,       # The training dataset (subset)
    batch_size=32,          # Load 32 samples at a time (like .batch(32) in TensorFlow)
    shuffle=True,           # Randomize data order each epoch (equivalent to .shuffle())
    num_workers=2,          # Use 2 subprocesses to load data in parallel (like prefetching)
    pin_memory=True         # Speeds up GPU data transfer (no direct TF equivalent but helps performance)
)

# === Validation DataLoader ===
val_loader = DataLoader(
    dataset=val_ds,         # The validation dataset
    batch_size=32,          # Same batch size
    shuffle=False,          # No shuffling — validation should be consistent every run
    num_workers=2,          # Preload batches in background threads
    pin_memory=True         # Same performance boost if using GPU
)

# === Test DataLoader ===
test_loader = DataLoader(
    dataset=test_ds,        # The test dataset
    batch_size=32,          # Batch size for inference
    shuffle=False,          # Test data must not be shuffled — ensures consistent accuracy evaluation
    num_workers=2,          # Background loading
    pin_memory=True         # Boosts data transfer speed to GPU
)

# print(f"Total samples: {len(dataset)}")
# print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
# print(f"Classes: {dataset.classes}")