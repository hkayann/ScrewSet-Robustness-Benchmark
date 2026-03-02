"""
Dataset classes and helpers shared across all phases.
"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def is_valid_image(path):
    """Check if a file has a valid image extension."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return os.path.splitext(path)[1].lower() in valid_exts


class NumpyDataset(Dataset):
    """Wraps uint8 numpy images (N, H, W, 3) + labels for CIFAR-10-C / corruption arrays."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.to_pil(self.images[idx])
        return self.transform(img), int(self.labels[idx])


class SamplesDataset(Dataset):
    """Dataset from a list of (path, label) tuples — used for ImageNet-C severity subsets."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class PILImageFolder(Dataset):
    """ImageFolder that returns (PIL_Image, label) — for generative VLM evaluation."""

    def __init__(self, root, is_valid_file=None):
        self.folder = ImageFolder(root, transform=None, is_valid_file=is_valid_file)
        self.classes = self.folder.classes
        self.samples = self.folder.samples

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        path, label = self.folder.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return img, label


class PILNumpyDataset(Dataset):
    """Wraps uint8 numpy images (N, H, W, 3) + labels, returns PIL images."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.to_pil(self.images[idx])
        return img, int(self.labels[idx])
