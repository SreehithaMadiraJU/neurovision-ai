from pathlib import Path

import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

# --------------------------------
# CLASSES
# --------------------------------

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# --------------------------------
# PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "models" / "tumor_model.pth"

# --------------------------------
# LOAD MODEL
# --------------------------------

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()

# --------------------------------
# INPUT IMAGE
# --------------------------------

image_path = input(
    "Enter MRI image path: "
).strip('"')

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor()
])

image = Image.open(
    image_path
).convert(
    "RGB"
)

image = transform(
    image
).unsqueeze(
    0
)

# --------------------------------
# PREDICTION
# --------------------------------

with torch.no_grad():

    output = model(
        image
    )

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

# --------------------------------
# OUTPUT
# --------------------------------

print()

print(
    "Prediction:",
    classes[prediction.item()]
)

print(
    f"Confidence: {confidence:.2f}%"
)
