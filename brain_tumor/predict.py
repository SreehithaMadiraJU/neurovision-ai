import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

# Classes
classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Load model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        r"C:\Users\masre\OneDrive\Desktop\Mini Project Code\brain_tumor\models\tumor_model.pth",
        map_location="cpu"
    )
)

model.eval()

# MRI image path
image_path = input(
    "Enter MRI image path: "
).strip('"')

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image
image = Image.open(
    image_path
).convert("RGB")

image = transform(
    image
).unsqueeze(0)

# Prediction
with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    )

    confidence = (
        probabilities[0][prediction.item()]
        * 100
    )

print("\nPrediction:", classes[prediction.item()])
print(f"Confidence: {confidence:.2f}%")