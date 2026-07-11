import os 
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import BirdDataset
from sklearn.model_selection import train_test_split

def load_dataframe(DATASET_PATH):
    images = pd.read_csv(
        os.path.join(DATASET_PATH, "images.txt"),
        sep=" ",
        names=["image_id", "filepath"]
    )

    labels = pd.read_csv(
        os.path.join(DATASET_PATH, "image_class_labels.txt"),
        sep=" ",
        names=["image_id", "class_id"]
    )

    split = pd.read_csv(
        os.path.join(DATASET_PATH, "train_test_split.txt"),
        sep=" ",
        names=["image_id", "is_train"]
    )


    df = images.merge(labels, on="image_id")
    df = df.merge(split, on="image_id")

    full_train_df = df[df["is_train"] == 1]
    test_df = df[df["is_train"] == 0]


    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.2,
        random_state=42,
        stratify=full_train_df["class_id"]
    )

    return train_df, val_df, test_df


def create_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return train_transform, test_transform

def create_train_val_test(DATASET_PATH, train_df, val_df, test_df, train_transform, test_transform):
    train_dataset = BirdDataset(
        train_df,
        DATASET_PATH,
        transform=train_transform
    )

    val_dataset = BirdDataset(
        val_df,
        DATASET_PATH,
        transform=test_transform
    )

    test_dataset = BirdDataset(
        test_df,
        DATASET_PATH,
        transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset

def create_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


