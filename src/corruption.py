"""
Corruption dataset discovery for ImageNet-C and Lens/ScrewSet-C.
"""
import os
from pathlib import Path

from src.config import IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA


def discover_imagenet_c_corruptions(imagenet_c_dir):
    """Discover available corruption directories under imagenet-c.

    ImageNet-C structure after extraction can be:
      imagenet-c/{corruption}/{severity}/{synsets}/ (flat)
    or
      imagenet-c/{category}/{corruption}/{severity}/{synsets}/ (nested under blur/noise/etc)

    This function handles both layouts and returns a dict {corruption_name: Path}.
    """
    imagenet_c_dir = Path(imagenet_c_dir)
    corruptions = {}

    all_known = IMAGENET_C_CORRUPTIONS_15 + IMAGENET_C_CORRUPTIONS_EXTRA
    for cname in all_known:
        # Direct layout: imagenet-c/{corruption}/1/n01440764/
        direct = imagenet_c_dir / cname
        if direct.exists() and (direct / "1").exists():
            corruptions[cname] = direct
            continue

        # Nested layout: imagenet-c/{category}/{corruption}/1/n01440764/
        for cat_dir in imagenet_c_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            nested = cat_dir / cname
            if nested.exists() and (nested / "1").exists():
                corruptions[cname] = nested
                break

    return corruptions


def find_corruption_leaf_dirs(corrupt_root):
    """Find leaf directories that contain synset class folders.

    Used for Lens/ScrewSet corruption datasets where the structure is:
      corrupt_root/{corruption_type}/{severity}/{synset_folders}/
    """
    leaf_dirs = []
    for root, dirs, _ in os.walk(str(corrupt_root), followlinks=True):
        p = Path(root)
        child_dirs = [p / d for d in dirs]
        if not child_dirs:
            continue
        if all(d.name.startswith("n") for d in child_dirs):
            leaf_dirs.append(p)
    return sorted(leaf_dirs)
