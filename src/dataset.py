from PIL import Image
from torch.utils.data import Dataset
import os

class BirdDataset(Dataset):

    def __init__(self, dataframe, root_dir, transform=None):

        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_path = os.path.join(
            self.root_dir,
            "images",
            row["filepath"]
        )

        image = Image.open(img_path).convert("RGB")

        label = row["class_id"] - 1

        if self.transform:
            image = self.transform(image)

        return image, label
