import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from data.data_loading import dataset  # Your original dataset

# ======== 1. Dataset Splitting Function ========
def get_pytorch_dataset_partitions(dataset, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    assert train_split + val_split + test_split == 1, "Splits must sum to 1"

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size  # ensures no rounding issues

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_pytorch_dataset_partitions(dataset)

# ======== 2. Transform Definitions ========

# === A. For custom CNN (no ImageNet normalization) ===
cnn_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(72),
    transforms.ToTensor(),
])

cnn_basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# === B. For pretrained ResNet (with ImageNet normalization) ===
effnet_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(72),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

effnet_basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======== 3. Apply transforms to duplicated datasets ========

# Clone datasets so both models can use their own version
from copy import deepcopy
cnn_train_ds = deepcopy(train_ds)
cnn_val_ds   = deepcopy(val_ds)
cnn_test_ds  = deepcopy(test_ds)

effnet_train_ds = deepcopy(train_ds)
effnet_val_ds   = deepcopy(val_ds)
effnet_test_ds  = deepcopy(test_ds)

# Apply transforms
cnn_train_ds.dataset.transform    = cnn_train_transform
cnn_val_ds.dataset.transform      = cnn_basic_transform
cnn_test_ds.dataset.transform     = cnn_basic_transform

effnet_train_ds.dataset.transform = effnet_train_transform
effnet_val_ds.dataset.transform   = effnet_basic_transform
effnet_test_ds.dataset.transform  = effnet_basic_transform

# ======== 4. Dataloaders ========

def create_loader(ds):
    return DataLoader(
        dataset=ds,
        batch_size=32,
        shuffle=isinstance(ds, torch.utils.data.Subset) and ds == cnn_train_ds or ds == effnet_train_ds,
        num_workers=3,
        pin_memory=True
    )

# Custom CNN loaders
cnn_train_loader = create_loader(cnn_train_ds)
cnn_val_loader   = create_loader(cnn_val_ds)
cnn_test_loader  = create_loader(cnn_test_ds)

# ResNet loaders
effnet_train_loader = create_loader(effnet_train_ds)
effnet_val_loader   = create_loader(effnet_val_ds)
effnet_test_loader  = create_loader(effnet_test_ds)
