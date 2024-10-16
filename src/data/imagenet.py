import os
import sys
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append("../..")


class BaseDataset(Dataset):

    def __init__(self, split, paths, size=None, random_crop=False, labels=None):
        """
        Initialize the BaseDataset.

        Args:
            split (str): The dataset split, either 'train' or 'val'.
            paths (list): List of image file paths.
            size (int, optional): The size to which images are resized.
            random_crop (bool, optional): Whether to apply random cropping.
            labels (dict, optional): A dictionary of labels associated with the images.
        """
        self.paths = sorted(paths)
        self.size = size
        self.random_crop = random_crop
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths

        if split == "train":
            self.transforms = transforms.Compose(
                [
                    (
                        transforms.RandomResizedCrop(size)
                        if self.random_crop
                        else transforms.Resize((size, size))
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def preprocess_image(self, image_path):
        """
        Preprocess an image by applying transformations.

        Args:
            image_path (str): The file path of the image to preprocess.

        Returns:
            Tensor: The transformed image tensor.
        """
        image = Image.open(image_path).convert("RGB")
        return self.transforms(image)

    def __getitem__(self, i):
        return {
            "input": self.preprocess_image(self.labels["file_path_"][i]),
            **{k: v[i] for k, v in self.labels.items()},
        }


class ImageNetDataset(BaseDataset):
    """ImageNet dataset class using BaseDataset."""

    def __init__(self, split: str, data_dir=None, size=256, random_crop=False):
        """
        Initialize the ImageNetDataset.

        Args:
            split (str): The dataset split, either 'train' or 'val'.
            data_dir (str, optional): The directory containing the ImageNet images.
            size (int, optional): The size to which images are resized.
            random_crop (bool, optional): Whether to apply random cropping.
        """
        self.split = split
        image_paths = sorted(
            glob(os.path.join(data_dir, "**", "*.JPEG"), recursive=True)
        )
        super().__init__(
            split=split, paths=image_paths, size=size, random_crop=random_crop
        )
