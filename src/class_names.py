"""
Class-name helpers for ScrewSet, Lens, and CIFAR-10 datasets.
"""
import os
from pathlib import Path

from src.imagenet_utils import get_imagenet_class_index


def screwset_folder_to_text(folder_name):
    """Convert ScrewSet folder name to natural language for CLIP prompts.

    Examples:
        M3_10_Flat    → "M3 10mm flat head screw"
        7#_25_Black   → "number 7 25mm black screw"
        10#_19_4.6    → "number 10 19mm grade 4.6 screw"
        8#_16_Round   → "number 8 16mm round head screw"
    """
    parts = folder_name.split("_")
    # Parse size
    size = parts[0]
    if "#" in size:
        size_text = f"number {size.replace('#', '')}"
    else:
        size_text = size  # M3, M4, M4.2, M5, M6

    # Parse length
    length = parts[1] if len(parts) > 1 else ""

    # Parse attribute
    attr = parts[2] if len(parts) > 2 else ""

    if attr.lower() in ("flat", "round"):
        return f"{size_text} {length}mm {attr.lower()} head screw"
    elif attr.lower() in ("black", "yellow"):
        return f"{size_text} {length}mm {attr.lower()} screw"
    elif attr:
        return f"{size_text} {length}mm grade {attr} screw"
    else:
        return f"{size_text} {length}mm screw"


def get_screwset_class_names(split_dir):
    """Get ordered class names for ScrewSet, matching ImageFolder ordering.

    Returns:
        (class_names, folder_names): list of human-readable names, list of raw folder names.
    """
    train_dir = Path(split_dir) / "train"
    folders = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(train_dir / d)])
    return [screwset_folder_to_text(f) for f in folders], folders


def get_lens_class_names(lens_dir):
    """Get class names for Lens dataset by mapping synset IDs to names."""
    class_index = get_imagenet_class_index()
    synset_to_name = {v[0]: v[1] for v in class_index.values()}

    train_dir = Path(lens_dir) / "train"
    synsets = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(train_dir / d)])

    class_names = []
    for synset in synsets:
        name = synset_to_name.get(synset, synset)
        class_names.append(name.replace("_", " "))
    return class_names
