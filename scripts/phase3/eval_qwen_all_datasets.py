#!/usr/bin/env python3
"""
Comprehensive Qwen VLM Evaluation on ALL Datasets
===================================================
Evaluates Qwen2.5-VL-7B and Qwen3-VL-8B (zero-shot) on:

  SS    : ScrewSet clean accuracy
  SS-C  : ScrewSet-C mean corruption accuracy (6 real corruptions)
  SS-S  : ScrewSet-S simulated corruptions (19 types × 5 severities, mem-mapped)
  C10   : CIFAR-10 clean accuracy
  C10-C : CIFAR-10-C mean corruption accuracy (19 types × 5 severities)
  Lens  : Lens clean accuracy
  Lens-C: Lens-C mean corruption accuracy (auto_exposure + param_control)
  IN-val: ImageNet validation accuracy
  IN-C  : ImageNet-C mean accuracy (19 types × 5 severities)
  IN-A  : ImageNet-A accuracy

Per-class accuracy is logged for every evaluation.

Usage:
    python eval_qwen_all_datasets.py --model qwen2.5    # Qwen2.5-VL only
    python eval_qwen_all_datasets.py --model qwen3       # Qwen3-VL only
    python eval_qwen_all_datasets.py --model both        # Both models
    python eval_qwen_all_datasets.py --smoke-test        # Quick check
    python eval_qwen_all_datasets.py --dataset screwset_c  # Single dataset
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED, CIFAR10_CLASSES,
    IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA,
)
from src.utils import patch_ipv4, set_seed
from src.datasets import is_valid_image, PILImageFolder, PILNumpyDataset
from src.imagenet_utils import (
    get_imagenet_class_names, build_imagenet_a_mapping,
)
from src.class_names import get_screwset_class_names, get_lens_class_names

patch_ipv4()

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "phase3"
SPLIT_DIR = DATA_DIR / "screwset_split"

ALL_CORRUPTIONS = IMAGENET_C_CORRUPTIONS_15 + IMAGENET_C_CORRUPTIONS_EXTRA

# ScrewSet-C corruption folders
SCREWSET_C_CORRUPTIONS = [
    "screwset_multi_object",
    "screwset_occlusion_bottom_right",
    "screwset_occlusion_top_left",
    "screwset_reflection",
    "screwset_scrap_paper",
    "screwset_shadow",
]

# All dataset keys this script handles
ALL_DATASET_KEYS = [
    "screwset", "screwset_c", "screwset_s",
    "cifar10", "cifar10_c",
    "lens", "lens_c",
    "imagenet_val", "imagenet_c", "imagenet_a",
]

# Model configs
MODEL_CONFIGS = {
    "qwen2.5": {
        "key": "qwen2_5_vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "Qwen2.5-VL",
        "params": "7B",
        "loader": "qwen2_5",
        "batch_size": 16,
    },
    "qwen3": {
        "key": "qwen3_vl_8b",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "Qwen3-VL",
        "params": "8B",
        "loader": "qwen3",
        "batch_size": 12,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(config, device):
    """Load a Qwen model. Returns (model, processor)."""
    model_id = config["model_id"]
    print(f"\n[MODEL] Loading {config['family']}: {model_id} ...")

    if config["loader"] == "qwen2_5":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model_cls = Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=28 * 28 * 4, max_pixels=448 * 448
    )
    processor.tokenizer.padding_side = "left"
    model = model_cls.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {config['key']}: {n_params:,} params loaded on {device}")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Response Matching
# ═══════════════════════════════════════════════════════════════════════════════

def _match_response_to_class(response, target_name, class_names_lower, target_idx):
    """Match a generative model's text response to the target class name."""
    from difflib import SequenceMatcher

    if not response:
        return False
    if response == target_name:
        return True
    if target_name in response:
        return True
    response_words = set(response.split())
    target_words = set(target_name.split())
    generic_tokens = {
        "screw", "screws", "head", "number", "grade", "mm",
        "flat", "round",
    }
    meaningful_target_words = [
        w for w in target_words
        if len(w) > 3 and w not in generic_tokens
    ]
    if meaningful_target_words and any(w in response_words for w in meaningful_target_words):
        return True
    best_ratio = 0.0
    best_idx = -1
    for idx, cn in enumerate(class_names_lower):
        ratio = SequenceMatcher(None, response, cn).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    if best_idx == target_idx and best_ratio > 0.4:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Core Evaluation (with per-class tracking)
# ═══════════════════════════════════════════════════════════════════════════════

def qwen_evaluate(model, processor, class_names, dataset, device,
                  desc="Qwen", batch_size=12, class_mapping=None,
                  max_debug=5):
    """Generative zero-shot evaluation with per-class accuracy tracking.

    Returns:
        (accuracy, per_class_acc_dict)
        per_class_acc_dict: {class_name: {"correct": int, "total": int, "acc": float}}
    """
    class_names_lower = [c.lower() for c in class_names]

    # Build prompt
    if len(class_names) <= 40:
        class_list = ", ".join(class_names)
        user_text = (
            f"Classify this image into exactly one of these categories: "
            f"{class_list}. Answer with only the category name, nothing else."
        )
    else:
        user_text = (
            "What is the main subject of this image? "
            "Answer with a single word or very short phrase."
        )

    # Per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    correct, total = 0, 0
    n = len(dataset)
    _debug_count = 0

    for i in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, n)
        batch_images = []
        batch_labels = []
        for j in range(i, batch_end):
            img, label = dataset[j]
            batch_images.append(img)
            batch_labels.append(label)

        conversations = []
        for img in batch_images:
            conversation = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ]},
            ]
            conversations.append(conversation)

        try:
            prompts = [
                processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]

            from qwen_vl_utils import process_vision_info
            image_inputs_list = []
            video_inputs_list = []
            for conv in conversations:
                img_inputs, vid_inputs = process_vision_info(conv)
                image_inputs_list.extend(img_inputs if img_inputs else [])
                video_inputs_list.extend(vid_inputs if vid_inputs else [])

            inputs = processor(
                text=prompts,
                images=image_inputs_list if image_inputs_list else None,
                videos=video_inputs_list if video_inputs_list else None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                input_len = inputs["input_ids"].shape[-1]
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )

            generated_ids = outputs[:, input_len:]
            generated = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for text, label in zip(generated, batch_labels):
                text_lower = text.lower().strip().rstrip(".")

                if class_mapping is not None:
                    target_idx = class_mapping[label]
                else:
                    target_idx = label

                if _debug_count < max_debug:
                    print(f"  [DEBUG] response={text_lower!r}  "
                          f"target={class_names_lower[target_idx]!r}")
                    _debug_count += 1

                target_name = class_names_lower[target_idx]
                is_correct = _match_response_to_class(
                    text_lower, target_name, class_names_lower, target_idx
                )
                if is_correct:
                    correct += 1
                    class_correct[target_idx] += 1
                class_total[target_idx] += 1
                total += 1

        except Exception as e:
            print(f"  [WARN] batch {i} error: {e}")
            for label in batch_labels:
                target_idx = class_mapping[label] if class_mapping else label
                class_total[target_idx] += 1
            total += len(batch_labels)

    acc = correct / total if total else 0.0

    # Build per-class dict
    per_class = {}
    for idx in sorted(class_total.keys()):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        c = class_correct[idx]
        t = class_total[idx]
        per_class[name] = {
            "correct": c,
            "total": t,
            "acc": c / t if t else 0.0,
        }

    return acc, per_class


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_screwset(model, processor, config, device):
    """SS: ScrewSet clean accuracy."""
    tag = config["key"]
    print(f"\n{'='*70}\n  ScrewSet Clean — {tag}\n{'='*70}")

    class_names, folder_names = get_screwset_class_names(SPLIT_DIR)
    test_ds = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
    print(f"[INFO] ScrewSet test: {len(test_ds)} images, {len(class_names)} classes")

    acc, per_class = qwen_evaluate(
        model, processor, class_names, test_ds, device,
        desc=f"[{tag}] SS", batch_size=config["batch_size"],
    )
    print(f"[{tag}] ScrewSet Clean Acc: {acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "screwset", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names), "test_acc": acc,
        "per_class_accuracy": per_class,
        "class_name_mapping": dict(zip(folder_names, class_names)),
        "seed": SEED,
    }
    return result


def run_screwset_c(model, processor, config, device):
    """SS-C: ScrewSet-C mean corruption accuracy (6 real natural corruptions)."""
    tag = config["key"]
    print(f"\n{'='*70}\n  ScrewSet-C — {tag}\n{'='*70}")

    ss_c_dir = DATA_DIR / "screwset_c"
    class_names, folder_names = get_screwset_class_names(SPLIT_DIR)

    corruption_results = {}
    all_accs = []

    for corruption in SCREWSET_C_CORRUPTIONS:
        corr_dir = ss_c_dir / corruption
        if not corr_dir.exists():
            print(f"  [WARN] {corruption} not found, skipping")
            continue

        ds = PILImageFolder(str(corr_dir), is_valid_file=is_valid_image)
        print(f"  [{corruption}] {len(ds)} images")

        acc, per_class = qwen_evaluate(
            model, processor, class_names, ds, device,
            desc=f"[{tag}] SS-C {corruption}", batch_size=config["batch_size"],
            max_debug=2,
        )
        corruption_results[corruption] = {
            "acc": acc, "num_images": len(ds), "per_class_accuracy": per_class,
        }
        all_accs.append(acc)
        print(f"  [{tag}] SS-C {corruption}: {acc:.4f}")

    mean_acc = np.mean(all_accs) if all_accs else 0.0
    print(f"[{tag}] ScrewSet-C Mean Corruption Acc: {mean_acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "screwset_c", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "corruption_results": corruption_results,
        "mean_corrupt_acc": float(mean_acc),
        "seed": SEED,
    }
    return result


def run_screwset_s(model, processor, config, device):
    """SS-S: ScrewSet-S simulated corruptions (19 types × 5 severities).

    Uses memory-mapped loading to avoid 87GB RAM per file.
    """
    tag = config["key"]
    print(f"\n{'='*70}\n  ScrewSet-S — {tag}\n{'='*70}")

    ss_s_dir = DATA_DIR / "screwset_s"
    class_names, folder_names = get_screwset_class_names(SPLIT_DIR)
    labels_all = np.load(ss_s_dir / "labels.npy")
    n_per_severity = len(labels_all) // 5  # 20480 images per severity

    corruption_results = {}
    all_mean_accs = []

    for corruption in ALL_CORRUPTIONS:
        npy_path = ss_s_dir / f"{corruption}.npy"
        if not npy_path.exists():
            print(f"  [WARN] {corruption}.npy not found, skipping")
            continue

        # Memory-map the file — NEVER copy full severity into RAM (18 GB each)
        images_mmap = np.load(str(npy_path), mmap_mode='r')
        severity_accs = {}
        severity_per_class = {}

        for sev in range(1, 6):
            start = (sev - 1) * n_per_severity
            end = sev * n_per_severity
            # Pass mmap slice directly — PILNumpyDataset reads one image at a time
            sev_images = images_mmap[start:end]
            sev_labels = labels_all[start:end]

            ds = PILNumpyDataset(sev_images, sev_labels)
            acc, per_class = qwen_evaluate(
                model, processor, class_names, ds, device,
                desc=f"[{tag}] SS-S {corruption} s{sev}",
                batch_size=config["batch_size"], max_debug=1,
            )
            severity_accs[f"severity_{sev}"] = acc
            severity_per_class[f"severity_{sev}"] = per_class
            print(f"  [{tag}] SS-S {corruption} sev{sev}: {acc:.4f}")

            import gc
            gc.collect()

        mean_corr = np.mean(list(severity_accs.values()))
        corruption_results[corruption] = {
            "severity_accs": severity_accs,
            "mean_acc": float(mean_corr),
            "per_class_by_severity": severity_per_class,
        }
        all_mean_accs.append(mean_corr)
        print(f"  [{tag}] SS-S {corruption} mean: {mean_corr:.4f}")
        del images_mmap

    overall_mean = np.mean(all_mean_accs) if all_mean_accs else 0.0
    print(f"[{tag}] ScrewSet-S Overall Mean Acc: {overall_mean:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "screwset_s", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "corruption_results": corruption_results,
        "overall_mean_acc": float(overall_mean),
        "seed": SEED,
    }
    return result


def run_cifar10(model, processor, config, device):
    """C10: CIFAR-10 clean accuracy."""
    tag = config["key"]
    print(f"\n{'='*70}\n  CIFAR-10 Clean — {tag}\n{'='*70}")

    from torchvision import datasets as tv_datasets
    class_names = CIFAR10_CLASSES
    test_ds = tv_datasets.CIFAR10(str(DATA_DIR / "cifar10"), train=False, download=False)
    test_ds_pil = [(test_ds[i][0], test_ds[i][1]) for i in range(len(test_ds))]
    print(f"[INFO] CIFAR-10 test: {len(test_ds_pil)} images, {len(class_names)} classes")

    acc, per_class = qwen_evaluate(
        model, processor, class_names, test_ds_pil, device,
        desc=f"[{tag}] C10", batch_size=config["batch_size"],
    )
    print(f"[{tag}] CIFAR-10 Clean Acc: {acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "cifar10", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names), "test_acc": acc,
        "per_class_accuracy": per_class,
        "seed": SEED,
    }
    return result


def run_cifar10_c(model, processor, config, device):
    """C10-C: CIFAR-10-C mean corruption accuracy."""
    tag = config["key"]
    print(f"\n{'='*70}\n  CIFAR-10-C — {tag}\n{'='*70}")

    c10c_dir = DATA_DIR / "cifar10_c" / "CIFAR-10-C"
    class_names = CIFAR10_CLASSES
    labels_all = np.load(c10c_dir / "labels.npy")
    n_per_severity = len(labels_all) // 5  # 10000 per severity

    corruption_results = {}
    all_mean_accs = []

    for corruption in ALL_CORRUPTIONS:
        npy_path = c10c_dir / f"{corruption}.npy"
        if not npy_path.exists():
            print(f"  [WARN] {corruption}.npy not found, skipping")
            continue

        images = np.load(str(npy_path))
        severity_accs = {}
        severity_per_class = {}

        for sev in range(1, 6):
            start = (sev - 1) * n_per_severity
            end = sev * n_per_severity
            sev_images = images[start:end]
            sev_labels = labels_all[start:end]

            ds = PILNumpyDataset(sev_images, sev_labels)
            acc, per_class = qwen_evaluate(
                model, processor, class_names, ds, device,
                desc=f"[{tag}] C10-C {corruption} s{sev}",
                batch_size=config["batch_size"], max_debug=1,
            )
            severity_accs[f"severity_{sev}"] = acc
            severity_per_class[f"severity_{sev}"] = per_class
            print(f"  [{tag}] C10-C {corruption} sev{sev}: {acc:.4f}")

        mean_corr = np.mean(list(severity_accs.values()))
        corruption_results[corruption] = {
            "severity_accs": severity_accs,
            "mean_acc": float(mean_corr),
            "per_class_by_severity": severity_per_class,
        }
        all_mean_accs.append(mean_corr)
        print(f"  [{tag}] C10-C {corruption} mean: {mean_corr:.4f}")
        del images

    overall_mean = np.mean(all_mean_accs) if all_mean_accs else 0.0
    print(f"[{tag}] CIFAR-10-C Overall Mean Acc: {overall_mean:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "cifar10_c", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "corruption_results": corruption_results,
        "overall_mean_acc": float(overall_mean),
        "seed": SEED,
    }
    return result


def run_lens(model, processor, config, device):
    """Lens: Lens clean accuracy."""
    tag = config["key"]
    print(f"\n{'='*70}\n  Lens Clean — {tag}\n{'='*70}")

    lens_dir = DATA_DIR / "lens_split"
    class_names = get_lens_class_names(lens_dir)
    test_ds = PILImageFolder(str(lens_dir / "test"))
    print(f"[INFO] Lens test: {len(test_ds)} images, {len(class_names)} classes")

    acc, per_class = qwen_evaluate(
        model, processor, class_names, test_ds, device,
        desc=f"[{tag}] Lens", batch_size=config["batch_size"],
    )
    print(f"[{tag}] Lens Clean Acc: {acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "lens", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names), "test_acc": acc,
        "per_class_accuracy": per_class,
        "seed": SEED,
    }
    return result


def run_lens_c(model, processor, config, device):
    """Lens-C: Lens-C mean corruption accuracy (auto_exposure + param_control)."""
    tag = config["key"]
    print(f"\n{'='*70}\n  Lens-C — {tag}\n{'='*70}")

    lens_dir = DATA_DIR / "lens_split"
    corr_dir = lens_dir / "corrupted"
    class_names = get_lens_class_names(lens_dir)

    corruption_results = {}
    all_accs = []

    # Two corruption categories, each with l1/l5 levels
    for category in ["auto_exposure", "param_control"]:
        cat_dir = corr_dir / category
        if not cat_dir.exists():
            print(f"  [WARN] {category} not found, skipping")
            continue

        for level in ["l1", "l5"]:
            level_dir = cat_dir / level
            if not level_dir.exists():
                continue

            # Each level has param_1..param_5 subfolders, each is an ImageFolder
            for param_subdir in sorted(os.listdir(str(level_dir))):
                param_path = level_dir / param_subdir
                if not param_path.is_dir():
                    continue

                key = f"{category}/{level}/{param_subdir}"
                ds = PILImageFolder(str(param_path))
                print(f"  [{key}] {len(ds)} images")

                acc, per_class = qwen_evaluate(
                    model, processor, class_names, ds, device,
                    desc=f"[{tag}] Lens-C {key}",
                    batch_size=config["batch_size"], max_debug=1,
                )
                corruption_results[key] = {
                    "acc": acc, "num_images": len(ds),
                    "per_class_accuracy": per_class,
                }
                all_accs.append(acc)
                print(f"  [{tag}] Lens-C {key}: {acc:.4f}")

    mean_acc = np.mean(all_accs) if all_accs else 0.0
    print(f"[{tag}] Lens-C Mean Corruption Acc: {mean_acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "lens_c", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "corruption_results": corruption_results,
        "mean_corrupt_acc": float(mean_acc),
        "seed": SEED,
    }
    return result


def run_imagenet_val(model, processor, config, device):
    """IN-val: ImageNet validation accuracy."""
    tag = config["key"]
    print(f"\n{'='*70}\n  ImageNet Val — {tag}\n{'='*70}")

    in_dir = DATA_DIR / "imagenet-val"
    class_names = get_imagenet_class_names()
    ds = PILImageFolder(str(in_dir), is_valid_file=is_valid_image)
    print(f"[INFO] ImageNet Val: {len(ds)} images, {len(class_names)} classes")

    acc, per_class = qwen_evaluate(
        model, processor, class_names, ds, device,
        desc=f"[{tag}] IN-val", batch_size=config["batch_size"],
    )
    print(f"[{tag}] ImageNet Val Acc: {acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "imagenet_val", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names), "num_images": len(ds),
        "val_acc": acc,
        "per_class_accuracy": per_class,
        "seed": SEED,
    }
    return result


def run_imagenet_c(model, processor, config, device):
    """IN-C: ImageNet-C mean accuracy (19 corruptions × 5 severities)."""
    tag = config["key"]
    print(f"\n{'='*70}\n  ImageNet-C — {tag}\n{'='*70}")

    in_c_dir = DATA_DIR / "imagenet-c"
    class_names = get_imagenet_class_names()

    corruption_results = {}
    all_mean_accs = []

    for corruption in ALL_CORRUPTIONS:
        corr_dir = in_c_dir / corruption
        if not corr_dir.exists():
            print(f"  [WARN] {corruption} not found, skipping")
            continue

        severity_accs = {}
        severity_per_class = {}

        for sev in range(1, 6):
            sev_dir = corr_dir / str(sev)
            if not sev_dir.exists():
                continue

            ds = PILImageFolder(str(sev_dir), is_valid_file=is_valid_image)
            acc, per_class = qwen_evaluate(
                model, processor, class_names, ds, device,
                desc=f"[{tag}] IN-C {corruption} s{sev}",
                batch_size=config["batch_size"], max_debug=1,
            )
            severity_accs[f"severity_{sev}"] = acc
            severity_per_class[f"severity_{sev}"] = per_class
            print(f"  [{tag}] IN-C {corruption} sev{sev}: {acc:.4f}")

        mean_corr = np.mean(list(severity_accs.values())) if severity_accs else 0.0
        corruption_results[corruption] = {
            "severity_accs": severity_accs,
            "mean_acc": float(mean_corr),
            "per_class_by_severity": severity_per_class,
        }
        all_mean_accs.append(mean_corr)
        print(f"  [{tag}] IN-C {corruption} mean: {mean_corr:.4f}")

    overall_mean = np.mean(all_mean_accs) if all_mean_accs else 0.0
    print(f"[{tag}] ImageNet-C Overall Mean Acc: {overall_mean:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "imagenet_c", "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "corruption_results": corruption_results,
        "overall_mean_acc": float(overall_mean),
        "seed": SEED,
    }
    return result


def run_imagenet_a(model, processor, config, device):
    """IN-A: ImageNet-A accuracy."""
    tag = config["key"]
    print(f"\n{'='*70}\n  ImageNet-A — {tag}\n{'='*70}")

    in_a_dir = DATA_DIR / "imagenet-a"
    class_names = get_imagenet_class_names()
    class_mapping = build_imagenet_a_mapping(str(in_a_dir))
    ds = PILImageFolder(str(in_a_dir), is_valid_file=is_valid_image)
    print(f"[INFO] ImageNet-A: {len(ds)} images, {len(class_mapping)} classes")

    acc, per_class = qwen_evaluate(
        model, processor, class_names, ds, device,
        desc=f"[{tag}] IN-A", batch_size=config["batch_size"],
        class_mapping=class_mapping,
    )
    print(f"[{tag}] ImageNet-A Acc: {acc:.4f}")

    result = {
        "model": tag, "model_family": config["family"],
        "dataset": "imagenet_a", "evaluation_mode": "zero_shot",
        "num_imagenet_a_classes": len(class_mapping),
        "imagenet_a_acc": acc,
        "per_class_accuracy": per_class,
        "seed": SEED,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Runner registry
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_RUNNERS = {
    "screwset": run_screwset,
    "screwset_c": run_screwset_c,
    "screwset_s": run_screwset_s,
    "cifar10": run_cifar10,
    "cifar10_c": run_cifar10_c,
    "lens": run_lens,
    "lens_c": run_lens_c,
    "imagenet_val": run_imagenet_val,
    "imagenet_c": run_imagenet_c,
    "imagenet_a": run_imagenet_a,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test(config, device):
    """Quick check: 20 images from ScrewSet + 20 from CIFAR-10 + 10 from ScrewSet-C."""
    tag = config["key"]
    print(f"\n{'='*70}\n  SMOKE TEST — {tag}\n{'='*70}")

    model, processor = load_model(config, device)
    import random
    random.seed(SEED)

    results = {}

    # 1. ScrewSet clean — 20 images
    class_names_ss, _ = get_screwset_class_names(SPLIT_DIR)
    full_ds = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
    indices = random.sample(range(len(full_ds)), min(20, len(full_ds)))
    subset = [(full_ds[i][0], full_ds[i][1]) for i in indices]
    t0 = time.time()
    acc, pc = qwen_evaluate(model, processor, class_names_ss, subset, device,
                            desc="Smoke SS", batch_size=4, max_debug=3)
    print(f"  SS: acc={acc:.4f} ({len(subset)} imgs, {time.time()-t0:.1f}s)")
    print(f"    Per-class sample: {dict(list(pc.items())[:3])}")
    results["screwset"] = acc

    # 2. CIFAR-10 clean — 20 images
    from torchvision import datasets as tv_datasets
    c10_ds = tv_datasets.CIFAR10(str(DATA_DIR / "cifar10"), train=False, download=False)
    indices = random.sample(range(len(c10_ds)), 20)
    subset = [(c10_ds[i][0], c10_ds[i][1]) for i in indices]
    t0 = time.time()
    acc, pc = qwen_evaluate(model, processor, CIFAR10_CLASSES, subset, device,
                            desc="Smoke C10", batch_size=4, max_debug=3)
    print(f"  C10: acc={acc:.4f} ({len(subset)} imgs, {time.time()-t0:.1f}s)")
    results["cifar10"] = acc

    # 3. ScrewSet-C — 10 images from first corruption
    ss_c_dir = DATA_DIR / "screwset_c" / SCREWSET_C_CORRUPTIONS[0]
    if ss_c_dir.exists():
        ds = PILImageFolder(str(ss_c_dir), is_valid_file=is_valid_image)
        indices = random.sample(range(len(ds)), min(10, len(ds)))
        subset = [(ds[i][0], ds[i][1]) for i in indices]
        t0 = time.time()
        acc, pc = qwen_evaluate(model, processor, class_names_ss, subset, device,
                                desc="Smoke SS-C", batch_size=4, max_debug=2)
        print(f"  SS-C ({SCREWSET_C_CORRUPTIONS[0]}): acc={acc:.4f} "
              f"({len(subset)} imgs, {time.time()-t0:.1f}s)")
        results["screwset_c"] = acc

    # 4. ScrewSet-S — 10 images from brightness sev1 (memory-mapped)
    ss_s_dir = DATA_DIR / "screwset_s"
    labels = np.load(ss_s_dir / "labels.npy")
    n_per_sev = len(labels) // 5
    imgs_mmap = np.load(str(ss_s_dir / "brightness.npy"), mmap_mode='r')
    sev1_imgs = np.array(imgs_mmap[:min(10, n_per_sev)])
    sev1_labels = labels[:min(10, n_per_sev)]
    ds = PILNumpyDataset(sev1_imgs, sev1_labels)
    t0 = time.time()
    acc, pc = qwen_evaluate(model, processor, class_names_ss, ds, device,
                            desc="Smoke SS-S", batch_size=4, max_debug=2)
    print(f"  SS-S (brightness s1): acc={acc:.4f} ({len(ds)} imgs, {time.time()-t0:.1f}s)")
    results["screwset_s"] = acc
    del imgs_mmap, sev1_imgs

    # 5. CIFAR-10-C — 10 images from gaussian_noise sev1
    c10c_dir = DATA_DIR / "cifar10_c" / "CIFAR-10-C"
    c10c_imgs = np.load(str(c10c_dir / "gaussian_noise.npy"))
    c10c_labels = np.load(str(c10c_dir / "labels.npy"))
    n_per_sev_c10 = len(c10c_labels) // 5
    ds = PILNumpyDataset(c10c_imgs[:10], c10c_labels[:10])
    t0 = time.time()
    acc, pc = qwen_evaluate(model, processor, CIFAR10_CLASSES, ds, device,
                            desc="Smoke C10-C", batch_size=4, max_debug=2)
    print(f"  C10-C (gaussian_noise s1): acc={acc:.4f} ({len(ds)} imgs, {time.time()-t0:.1f}s)")
    results["cifar10_c"] = acc
    del c10c_imgs

    # 6. Lens-C — 10 images from auto_exposure/l1/param_1
    lens_c_path = DATA_DIR / "lens_split" / "corrupted" / "auto_exposure" / "l1" / "param_1"
    if lens_c_path.exists():
        class_names_lens = get_lens_class_names(DATA_DIR / "lens_split")
        ds = PILImageFolder(str(lens_c_path))
        indices = random.sample(range(len(ds)), min(10, len(ds)))
        subset = [(ds[i][0], ds[i][1]) for i in indices]
        t0 = time.time()
        acc, pc = qwen_evaluate(model, processor, class_names_lens, subset, device,
                                desc="Smoke Lens-C", batch_size=4, max_debug=2)
        print(f"  Lens-C (AE/l1/p1): acc={acc:.4f} ({len(subset)} imgs, {time.time()-t0:.1f}s)")
        results["lens_c"] = acc

    print(f"\n  SMOKE TEST SUMMARY:")
    for k, v in results.items():
        status = "PASS" if v >= 0 else "FAIL"
        print(f"    {k:15s}: {v:.4f} [{status}]")

    all_pass = all(v >= 0 for v in results.values())
    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    del model
    torch.cuda.empty_cache()
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Qwen VLM Evaluation on ALL Datasets",
    )
    parser.add_argument("--model", type=str, default="both",
                        choices=["qwen2.5", "qwen3", "both"],
                        help="Which model(s) to evaluate")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=ALL_DATASET_KEYS + ["all"],
                        help="Dataset to evaluate (default: all)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick integrity check")
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even if results exist")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3) if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] Seed: {SEED}")

    set_seed()

    # Select models
    if args.model == "both":
        models_to_run = ["qwen2.5", "qwen3"]
    else:
        models_to_run = [args.model]

    # Select datasets
    if args.dataset in (None, "all"):
        datasets_to_run = ALL_DATASET_KEYS
    else:
        datasets_to_run = [args.dataset]

    # Smoke test
    if args.smoke_test:
        for model_name in models_to_run:
            config = MODEL_CONFIGS[model_name]
            success = run_smoke_test(config, device)
            if not success:
                print(f"[FATAL] Smoke test FAILED for {model_name}")
                sys.exit(1)
        print("\n[OK] All smoke tests passed!")
        sys.exit(0)

    # Full evaluation
    for model_name in models_to_run:
        config = MODEL_CONFIGS[model_name]
        tag = config["key"]

        print(f"\n{'#'*70}")
        print(f"  MODEL: {config['family']} ({config['params']})")
        print(f"{'#'*70}")

        model, processor = load_model(config, device)
        failed = []

        for ds_name in datasets_to_run:
            # Result file naming
            out_path = RESULTS_DIR / f"{tag}_{ds_name}_full.json"

            if out_path.exists() and not args.force:
                print(f"\n[SKIP] {tag} × {ds_name} — {out_path.name} exists")
                continue

            runner = DATASET_RUNNERS[ds_name]

            set_seed()
            try:
                result = runner(model, processor, config, device)
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=4)
                print(f"[DONE] Results saved to {out_path}")
            except Exception as e:
                print(f"[ERROR] {tag} × {ds_name} failed: {e}")
                import traceback
                traceback.print_exc()
                failed.append(ds_name)
            finally:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        n_total = len(datasets_to_run)
        print(f"\n{'='*70}")
        print(f"  {tag}: {n_total - len(failed)}/{n_total} completed")
        if failed:
            print(f"  FAILED: {', '.join(failed)}")
        print(f"{'='*70}")

    print("\n[COMPLETE] All evaluations finished.")


if __name__ == "__main__":
    main()
