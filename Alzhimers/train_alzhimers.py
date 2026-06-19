from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------------
# PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

train_dir = BASE_DIR / "alzhimers_dataset" / "train"

model_path = BASE_DIR / "models" / "alzhimers_model.pth"

# -----------------------------------
# IMAGE TRANSFORMS
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# -----------------------------------
# LOAD DATASET
# -----------------------------------

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

print("Dataset Loaded!")
print("Classes:", train_dataset.classes)

# -----------------------------------
# MODEL
# -----------------------------------

model = models.resnet18(
    weights="DEFAULT"
)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

device = torch.device("cpu")

model = model.to(device)

# -----------------------------------
# LOSS FUNCTION
# -----------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

print("Training Started...")

# -----------------------------------
# TRAINING LOOP
# -----------------------------------

for epoch in range(epochs):

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        if batch_idx % 20 == 0:

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

# -----------------------------------
# SAVE MODEL
# -----------------------------------

torch.save(
    model.state_dict(),
    model_path
)

print("Model Saved Successfully!")
