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

    test_dir = BASE_DIR / "dataset" / "Testing"

    model_path = BASE_DIR / "models" / "tumor_model.pth"

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
        batch_size=16,
        shuffle=False
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

            _, predicted = torch.max(
                outputs,
                1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predicted.numpy()
            )

    # -----------------------------------
    # METRICS
    # -----------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return {

        "accuracy": accuracy * 100,

        "precision": precision * 100,

        "recall": recall * 100,

        "f1": f1 * 100

    }


if __name__ == "__main__":

    metrics = evaluate_model()

    print()

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
