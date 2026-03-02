#!/usr/bin/env python3
"""Create a ScrewSet-style split layout for ImageNet-ES-Diverse.

This script creates:
  data/lens_split/
    train/         # class-balanced split from es-train
    validation/    # class-balanced split from es-train
    test/          # clean reference test set
    corrupted/
      auto_exposure -> source dir
      param_control -> source dir

`train` and `validation` are materialized as class folders so downstream
training scripts can consume a stable split directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


def count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_dir(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def list_class_files(root: Path) -> Dict[str, List[Path]]:
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    out: Dict[str, List[Path]] = {}
    for cls in classes:
        class_dir = root / cls
        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        out[cls] = files
    return out


def split_train_val(
    class_to_files: Dict[str, List[Path]],
    val_per_class: int,
    seed: int,
) -> tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    rng = np.random.default_rng(seed)
    train_split: Dict[str, List[Path]] = {}
    val_split: Dict[str, List[Path]] = {}
    for cls, files in class_to_files.items():
        n = len(files)
        if val_per_class <= 0 or val_per_class >= n:
            raise ValueError(f"Invalid val_per_class={val_per_class} for class '{cls}' with {n} files.")
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = set(idx[:val_per_class].tolist())
        val_split[cls] = [files[i] for i in range(n) if i in val_idx]
        train_split[cls] = [files[i] for i in range(n) if i not in val_idx]
    return train_split, val_split


def materialize_class_split(split: Dict[str, List[Path]], out_root: Path, mode: str) -> None:
    ensure_clean_dir(out_root)
    for cls, files in split.items():
        cls_out = out_root / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        for src_file in files:
            dst_file = cls_out / src_file.name
            if mode == "symlink":
                dst_file.symlink_to(src_file.resolve())
            else:
                shutil.copy2(src_file, dst_file)


def materialize_test_set(test_src: Path, test_dst: Path, mode: str) -> None:
    class_to_files = list_class_files(test_src)
    materialize_class_split(class_to_files, test_dst, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create lens_split layout from ImageNet-ES-Diverse.")
    parser.add_argument(
        "--src-root",
        default="data/lens/ImageNet-ES-Diverse",
        help="Path to extracted ImageNet-ES-Diverse root.",
    )
    parser.add_argument(
        "--out-root",
        default="data/lens_split",
        help="Output split root.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize train/validation/test files.",
    )
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=4,
        help="Validation images per class sampled from clean train (default: 4 -> 10% of 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()

    train_src = src_root / "es-train" / "tin_no_resize_sample_removed"
    test_clean_src = src_root / "es-diverse-test" / "sampled_tin_no_resize2"
    auto_src = src_root / "es-diverse-test" / "auto_exposure"
    param_src = src_root / "es-diverse-test" / "param_control"

    required = [train_src, test_clean_src, auto_src, param_src]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source paths: {missing}")

    class_to_files = list_class_files(train_src)
    classes = sorted(class_to_files.keys())
    if not classes:
        raise RuntimeError(f"No class folders found in {train_src}")
    n_per_class = {cls: len(files) for cls, files in class_to_files.items()}
    unique_counts = sorted(set(n_per_class.values()))
    if len(unique_counts) != 1:
        raise RuntimeError(f"Class imbalance in source train set: {unique_counts}")

    train_split, val_split = split_train_val(
        class_to_files=class_to_files,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )

    ensure_clean_dir(out_root)
    (out_root / "corrupted").mkdir(parents=True, exist_ok=True)

    materialize_class_split(train_split, out_root / "train", args.mode)
    materialize_class_split(val_split, out_root / "validation", args.mode)
    materialize_test_set(test_clean_src, out_root / "test", args.mode)

    # Keep corruption folders as directory-level links/copies (already nested and large).
    link_or_copy_dir(auto_src, out_root / "corrupted" / "auto_exposure", args.mode)
    link_or_copy_dir(param_src, out_root / "corrupted" / "param_control", args.mode)

    summary = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "materialization_mode": args.mode,
        "seed": args.seed,
        "num_classes": len(classes),
        "source_train_images_per_class": unique_counts[0],
        "val_per_class": args.val_per_class,
        "clean_train_images": count_files(out_root / "train"),
        "clean_validation_images": count_files(out_root / "validation"),
        "clean_test_images": count_files(test_clean_src),
        "corrupted_auto_exposure_images": count_files(auto_src),
        "corrupted_param_control_images": count_files(param_src),
    }
    summary["corrupted_total_images"] = (
        summary["corrupted_auto_exposure_images"] + summary["corrupted_param_control_images"]
    )
    summary["clean_total_images"] = (
        summary["clean_train_images"] + summary["clean_validation_images"] + summary["clean_test_images"]
    )

    summary_path = out_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Created lens split:")
    print(f"  train      -> {out_root / 'train'}")
    print(f"  validation -> {out_root / 'validation'}")
    print(f"  test       -> {out_root / 'test'}")
    print(f"  corrupted  -> {out_root / 'corrupted'}")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
