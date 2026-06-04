import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Dataset paths
train_dir = r"C:\Users\masre\OneDrive\Desktop\Mini Project Code\Alzhimers\dataset\train"

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
train_dataset = datasets.ImageFolder(
    train_dir,
    transform=transform
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

print("Dataset Loaded!")
print("Classes:", train_dataset.classes)

# ResNet18
model = models.resnet18(
    weights="DEFAULT"
)

# 4 classes
model.fc = nn.Linear(
    model.fc.in_features,
    4
)

device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 1

print("Training Started...")

for epoch in range(epochs):

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        if batch_idx % 20 == 0:

            print(
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

# Save model
torch.save(
    model.state_dict(),
    r"C:\Users\masre\OneDrive\Desktop\Mini Project Code\Alzhimers\models\alzhimers_model.pth"
)

print("Model Saved Successfully!")