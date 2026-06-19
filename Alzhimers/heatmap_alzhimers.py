from pathlib import Path

import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --------------------------------
# CLASSES
# --------------------------------

classes = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# --------------------------------
# PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "models" / "alzhimers_model.pth"

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

save_path = output_dir / "outputs.jpg"

# --------------------------------
# LOAD MODEL
# --------------------------------

model = models.resnet18(weights=None)

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

image = Image.open(
    image_path
).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

input_tensor = transform(
    image
).unsqueeze(0)

# --------------------------------
# PREDICTION
# --------------------------------

with torch.no_grad():

    output = model(input_tensor)

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

print()

print(
    "Prediction:",
    classes[prediction.item()]
)

print(
    f"Confidence: {confidence:.2f}%"
)

# --------------------------------
# HEATMAP
# --------------------------------

rgb_img = image.resize(
    (224,224)
)

img_np = np.array(
    rgb_img
) / 255.0

target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor
)[0]

visualization = show_cam_on_image(
    img_np,
    grayscale_cam,
    use_rgb=True
)

# --------------------------------
# SAVE HEATMAP
# --------------------------------

heatmap_image = Image.fromarray(
    visualization
)

heatmap_image.save(
    save_path
)

print()

print(
    "Heatmap saved successfully!"
)

print(
    "Location:",
    str(save_path)
)

print()

print("Heatmap Interpretation:")

print("🔴 Red    -> Very High Attention")

print("🟡 Yellow -> High Attention")

print("🟢 Green  -> Moderate Attention")

print("🔵 Blue   -> Low Attention")
