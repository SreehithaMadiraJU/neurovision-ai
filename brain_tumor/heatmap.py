from pathlib import Path

import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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

output_dir = BASE_DIR / "outputs"

output_dir.mkdir(
    exist_ok=True
)

heatmap_path = output_dir / "heatmap.jpg"

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

rgb_img = Image.open(
    image_path
).convert(
    "RGB"
)

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

# --------------------------------
# GRAD-CAM
# --------------------------------

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

cv2.imwrite(
    str(
        heatmap_path
    ),
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print()

print(
    "Heatmap saved successfully!"
)

print(
    "Location:",
    heatmap_path
)
