import os
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, data_path: str, root_path: str, transforms: Optional[callable] = None):
        """
        Initializes the dataset by reading the image paths and labels from the provided data path.

        Args:
            data_path (str): Path to the text file containing image file paths. Labels are inferred from image file folder.
            root_path (str): Path to the root directory containing folders with images.
            transforms (callable, optional): Torchvision transforms.
        """
        self.images = []
        self.labels = []
        self.category_names = os.listdir(root_path)
        self.transforms = transforms

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                label = line.split("/")[-2]
                path = os.path.join(root_path, line)

                self.images.append(path)
                self.labels.append(self.category_names.index(label))
        self.samples = list(zip(self.images, self.labels))

    @staticmethod
    def load_image(path: str):
        image = Image.open(path)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        label = self.labels[idx]

        image = self.load_image(path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label
