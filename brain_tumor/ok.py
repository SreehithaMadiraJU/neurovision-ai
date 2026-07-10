from torchvision import datasets

dataset = datasets.ImageFolder(
    "brain_tumor/brain-tumor-mri-dataset/Training"
)

print(dataset.class_to_idx)