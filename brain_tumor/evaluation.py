from pathlib import Path

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model():

    # -----------------------------------
    # PATHS
    # -----------------------------------

    BASE_DIR = Path(__file__).resolve().parent

    test_dir = (
        BASE_DIR
        / "brain-tumor-mri-dataset"
        / "Testing"
    )

    model_path = (
        BASE_DIR
        / "models"
        / "tumor_model.pth"
    )

    # -----------------------------------
    # TRANSFORM
    # -----------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # -----------------------------------
    # DATASET
    # -----------------------------------

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,      # Increased batch size
        shuffle=False,
        num_workers=4        # Faster loading
    )

    # -----------------------------------
    # MODEL
    # -----------------------------------

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

    # -----------------------------------
    # PREDICTIONS
    # -----------------------------------

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predicted.numpy()
            )

    # -----------------------------------
    # METRICS
    # -----------------------------------

    return {

        "accuracy":
        accuracy_score(
            y_true,
            y_pred
        ) * 100,

        "precision":
        precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ) * 100,

        "recall":
        recall_score(
            y_true,
            y_pred,
            average="weighted"
        ) * 100,

        "f1":
        f1_score(
            y_true,
            y_pred,
            average="weighted"
        ) * 100

    }


if __name__ == "__main__":

    metrics = evaluate_model()

    print(
        f"Accuracy : {metrics['accuracy']:.2f}%"
    )

    print(
        f"Precision : {metrics['precision']:.2f}%"
    )

    print(
        f"Recall : {metrics['recall']:.2f}%"
    )

    print(
        f"F1 Score : {metrics['f1']:.2f}%"
    )
