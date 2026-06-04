import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Classes
classes = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

# Load model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        r"C:\Users\masre\OneDrive\Desktop\Mini Project Code\Alzhimers\models\alzhimers_model.pth",
        map_location="cpu"
    )
)

model.eval()

# MRI image path
image_path = input(
    "Enter MRI image path: "
).strip('"')

# Load image
rgb_img = Image.open(
    image_path
).convert("RGB")

rgb_img = rgb_img.resize(
    (224, 224)
)

img_np = np.array(
    rgb_img
) / 255.0

transform = transforms.Compose([
    transforms.ToTensor()
])

input_tensor = transform(
    rgb_img
).unsqueeze(0)

# Grad-CAM
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

# Save heatmap
cv2.imwrite(
    r"C:\Users\masre\OneDrive\Desktop\Mini Project Code\Alzhimers\outputs\heatmap_alzhimers.jpg",
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print("Heatmap saved!")