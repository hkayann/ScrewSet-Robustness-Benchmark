"""
ImageNet utilities: class index, class names, ImageNet-A mapping.
"""
import json
import os
import urllib.request
from pathlib import Path

from src.config import DATA_DIR


def get_imagenet_class_index():
    """Download and cache the ImageNet 1K synset→index mapping JSON."""
    cache_path = DATA_DIR / "imagenet_class_index.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    print(f"[INFO] Downloading ImageNet class index from {url}")
    urllib.request.urlretrieve(url, str(cache_path))
    with open(cache_path) as f:
        return json.load(f)


def get_imagenet_class_names():
    """Get ordered list of 1000 ImageNet class names (human-readable)."""
    class_index = get_imagenet_class_index()
    names = []
    for i in range(1000):
        entry = class_index[str(i)]
        name = entry[1].replace("_", " ")
        names.append(name)
    return names


def build_imagenet_a_mapping(imagenet_a_dir):
    """Build mapping from ImageNet-A ImageFolder class idx → ImageNet-1K class idx.

    ImageNet-A has 200 classes. ImageFolder assigns indices 0..199 based on
    sorted folder names. This function returns a list of length 200,
    where mapping[i] = ImageNet-1K class index for ImageNet-A's i-th class.
    """
    class_index = get_imagenet_class_index()
    synset_to_idx = {v[0]: int(k) for k, v in class_index.items()}

    synset_folders = sorted(
        [d for d in os.listdir(imagenet_a_dir)
         if os.path.isdir(os.path.join(imagenet_a_dir, d)) and d.startswith("n")]
    )

    mapping = []
    for synset in synset_folders:
        if synset not in synset_to_idx:
            raise ValueError(f"Synset {synset} not found in ImageNet class index!")
        mapping.append(synset_to_idx[synset])

    print(f"[INFO] ImageNet-A mapping: {len(mapping)} classes → ImageNet-1K indices")
    return mapping
