import torchvision.datasets as ds
from torchvision import transforms

dataset_mapping = {
    "flowers102": ds.Flowers102,
}

def create_image_transform(resize_size):

    return transforms.Compose([
        transforms.Resize(resize_size,antialias=True),
        transforms.ToTensor()
    ])

def get_dataset_mapping(dataset_name):
    if dataset_name in dataset_mapping:
        return {dataset_name: dataset_mapping[dataset_name]}

    raise KeyError(f"Unsupported image dataset: {dataset_name}")

def load_dataset_tvt(name, dataset_mapping, root="../root", transform=None, download=True):
    dataset_class = dataset_mapping[name]

    train_dataset = dataset_class(
        root=root,
        split="train",
        transform=transform,
        download=download
    )

    val_dataset = dataset_class(
        root=root,
        split="val",
        transform=transform,
        download=download
    )

    test_dataset = dataset_class(
        root=root,
        split="test",
        transform=transform,
        download=download
    )

    return train_dataset, val_dataset, test_dataset