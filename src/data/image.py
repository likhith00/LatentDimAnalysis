import numpy as np
import torch

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_dataset_labels(dataset):

    if hasattr(dataset, "_labels"):
        return np.asarray(dataset._labels)

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)

    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)

    raise AttributeError(f"Could not find labels for {type(dataset).__name__}")


def get_concat_labels(dataset):

    if isinstance(dataset, ConcatDataset):

        labels = []

        for ds in dataset.datasets:
            labels.append(get_dataset_labels(ds))

        return np.concatenate(labels)

    return get_dataset_labels(dataset)


def limit_dataset_indices(labels, max_samples=10000, random_state=42):
    labels = np.asarray(labels)

    all_indices = np.arange(len(labels))

    if len(labels) <= max_samples:
        return all_indices

    selected_indices, _ = train_test_split(
        all_indices,
        train_size=max_samples,
        stratify=labels,
        random_state=random_state
    )

    return selected_indices


def create_stratified_indices(
    indices,
    labels,
    train_size=0.60,
    val_size=0.20,
    test_size=0.20,
    random_state=42
):

    labels = np.asarray(labels)

    selected_labels = labels[indices]

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        stratify=selected_labels,
        random_state=random_state
    )

    temp_labels = labels[temp_idx]

    relative_val_size = (
        val_size /
        (val_size + test_size)
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_size,
        stratify=temp_labels,
        random_state=random_state
    )

    return train_idx, val_idx, test_idx

def subset_to_features(subset, batch_size=128, num_workers=4):

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    X_batches = []
    y_batches = []

    for images, labels in loader:
        images = images.flatten(start_dim=1)

        X_batches.append(images)
        y_batches.append(labels)

    X = torch.cat(X_batches, dim=0)
    y = torch.cat(y_batches, dim=0)

    return X, y


def preprocess_torchvision_dataset(
    dataset,
    max_samples=10000,
    train_size=0.60,
    val_size=0.20,
    test_size=0.20,
    batch_size=128,
    num_workers=4,
    random_state=42
):


    print("(Step 1/6) Retrieving Labels ....")
    labels = get_concat_labels(dataset)

    print("(Step 2/6) Encoding Labels....")
    label_encoder = LabelEncoder()

    labels_encoded = label_encoder.fit_transform(
        labels
    )

    print('(Step 3/6) Creating Stratified indices....')
    selected_indices = limit_dataset_indices(labels_encoded, max_samples=max_samples, random_state=random_state)


    train_idx, val_idx, test_idx = (
        create_stratified_indices(
            selected_indices,
            labels_encoded,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state
        )
    )

    print("(Step 4/6) Creating Subsets...")
    train_subset = Subset(
        dataset,
        train_idx
    )

    val_subset = Subset(
        dataset,
        val_idx
    )

    test_subset = Subset(
        dataset,
        test_idx
    )

    print("(Step 5/6) Transforming subset to features ....")
    X_train_tensor, y_train_tensor = (
        subset_to_features(
            train_subset,
            batch_size=batch_size,
            num_workers=num_workers
        )
    )

    X_val_tensor, y_val_tensor = (
        subset_to_features(
            val_subset,
            batch_size=batch_size,
            num_workers=num_workers
        )
    )

    X_test_tensor, y_test_tensor = (
        subset_to_features(
            test_subset,
            batch_size=batch_size,
            num_workers=num_workers
        )
    )

    print("(Step 6/6) Converting to numpy arrays...")

    X_train = X_train_tensor.numpy()
    X_val = X_val_tensor.numpy()
    X_test = X_test_tensor.numpy()

    y_train = y_train_tensor.numpy()
    y_val = y_val_tensor.numpy()
    y_test = y_test_tensor.numpy()

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "X_train_tensor": X_train_tensor,
        "X_val_tensor": X_val_tensor,
        "X_test_tensor": X_test_tensor,

        "y_train_tensor": y_train_tensor,
        "y_val_tensor": y_val_tensor,
        "y_test_tensor": y_test_tensor,

        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,

        "label_encoder": label_encoder,

        "input_dim": X_train.shape[1]
    }