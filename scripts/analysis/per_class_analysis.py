#!/usr/bin/env python3
"""
Per-Class Accuracy & Confusion Matrix Analysis for ScrewSet
============================================================
Loads saved Phase 1 (CNN) and Phase 2 (ViT) checkpoints, runs inference on
clean test and ScrewSet-C corruptions, and produces:

  1. Per-class accuracy heatmap (models × classes, clean + per-corruption)
  2. Confusion matrices for selected models (clean + worst corruption)
  3. Class-difficulty ranking (which classes are hardest under corruption)
  4. CSV tables with all per-class numbers

Usage:
    python3 scripts/analysis/per_class_analysis.py
    python3 scripts/analysis/per_class_analysis.py --model resnet18
    python3 scripts/analysis/per_class_analysis.py --phase 1
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DATA_DIR, SEED,
    CNN_TIMM_PRETRAINED_IDS,
    VIT_TIMM_PRETRAINED_IDS,
)
from src.utils import set_seed
from src.datasets import is_valid_image

# ── Output directories ───────────────────────────────────────────────────────
RESULTS_DIR = REPO_ROOT / "results" / "per_class"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset paths ────────────────────────────────────────────────────────────
SPLIT_DIR = DATA_DIR / "screwset_split"
CORRUPT_ROOT = DATA_DIR / "screwset_c"

# ── ScrewSet normalization (Phase 1 CNNs) ────────────────────────────────────
SCREWSET_NORM = {"mean": [0.7750, 0.7343, 0.6862], "std": [0.0802, 0.0838, 0.0871]}
SCREWSET_RESIZE = (240, 320)

# ── Phase 2 ViT preprocessing (per model) ────────────────────────────────────
MODEL_PREPROCESS = {
    "vit_tiny_patch16_224": {
        "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
        "input_size": 224, "crop_pct": 0.9,
    },
    "vit_small_patch16_224": {
        "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
        "input_size": 224, "crop_pct": 0.9,
    },
    "deit_tiny_patch16_224": {
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "input_size": 224, "crop_pct": 0.9,
    },
    "deit_small_patch16_224": {
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "input_size": 224, "crop_pct": 0.9,
    },
    "swin_tiny_patch4_window7_224": {
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "input_size": 224, "crop_pct": 0.9,
    },
    "mobilevit_s": {
        "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0),
        "input_size": 256, "crop_pct": 0.9,
    },
    "efficientformer_l1": {
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "input_size": 224, "crop_pct": 0.95,
    },
    "convnext_tiny": {
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "input_size": 224, "crop_pct": 0.875,
    },
}

# ── Model lists ──────────────────────────────────────────────────────────────
PHASE1_MODELS = [
    "resnet18", "squeezenet1_1", "mobilenet_v3_large", "mobilenetv4_conv_small",
    "shufflenet_v2_x1_0", "efficientnetv2_rw_s", "ghostnetv2_100", "convnextv2_atto",
]
PHASE2_MODELS = [
    "vit_tiny_patch16_224", "vit_small_patch16_224",
    "deit_tiny_patch16_224", "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224", "mobilevit_s",
    "efficientformer_l1", "convnext_tiny",
]

CORRUPTION_TYPES = [
    "screwset_multi_object", "screwset_occlusion_bottom_right",
    "screwset_occlusion_top_left", "screwset_reflection",
    "screwset_scrap_paper", "screwset_shadow",
]
CORRUPTION_SHORT = ["Multi-Obj", "Occl-BR", "Occl-TL", "Reflect", "Scrap-P", "Shadow"]

# ── Display names ────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "resnet18": "ResNet-18",
    "squeezenet1_1": "SqueezeNet",
    "mobilenet_v3_large": "MobileNetV3-L",
    "mobilenetv4_conv_small": "MobileNetV4-S",
    "shufflenet_v2_x1_0": "ShuffleNetV2",
    "efficientnetv2_rw_s": "EfficientNetV2-S",
    "ghostnetv2_100": "GhostNetV2",
    "convnextv2_atto": "ConvNeXtV2-A",
    "vit_tiny_patch16_224": "ViT-Tiny",
    "vit_small_patch16_224": "ViT-Small",
    "deit_tiny_patch16_224": "DeiT-Tiny",
    "deit_small_patch16_224": "DeiT-Small",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "mobilevit_s": "MobileViT-S",
    "efficientformer_l1": "EffFormer-L1",
    "convnext_tiny": "ConvNeXt-Tiny",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory (matches Phase 1 / Phase 2 exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def create_phase1_model(name, num_classes):
    """Create a Phase 1 CNN model (same architecture as training)."""
    is_timm = name in CNN_TIMM_PRETRAINED_IDS

    if is_timm:
        model = timm.create_model(name, pretrained=False, num_classes=num_classes)
        return model

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown Phase 1 model: {name}")
    return model


def create_phase2_model(name, num_classes):
    """Create a Phase 2 ViT model (same architecture as training)."""
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def get_eval_transform(model_name, phase):
    """Get the correct eval transform matching training preprocessing.

    For Phase 2, reads actual normalization & resize from the saved JSON
    (the training code may have used different normalization than MODEL_PREPROCESS).
    """
    if phase == 1:
        return transforms.Compose([
            transforms.Resize(SCREWSET_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=SCREWSET_NORM["mean"], std=SCREWSET_NORM["std"]),
        ])
    else:
        # Read actual training config from saved Phase 2 JSON
        json_path = REPO_ROOT / "results" / "phase2" / f"{model_name}_screwset_baselines.json"
        if json_path.exists():
            with open(json_path) as f:
                saved = json.load(f)
            norm_mean = saved.get("normalization_mean", list(MODEL_PREPROCESS[model_name]["mean"]))
            norm_std = saved.get("normalization_std", list(MODEL_PREPROCESS[model_name]["std"]))

            # Parse resize string e.g. "Resize(248)+CenterCrop(224)"
            resize_str = saved.get("resize", "")
            import re
            m_resize = re.search(r"Resize\((\d+)\)", str(resize_str))
            m_crop = re.search(r"CenterCrop\((\d+)\)", str(resize_str))
            if m_resize and m_crop:
                resize_short = int(m_resize.group(1))
                img_size = int(m_crop.group(1))
            else:
                pcfg = MODEL_PREPROCESS[model_name]
                img_size = pcfg["input_size"]
                resize_short = int(math.floor(img_size / pcfg["crop_pct"]))

            print(f"  [PREPROCESS] norm_mean={norm_mean}, resize={resize_short}, crop={img_size}")
        else:
            # Fallback to MODEL_PREPROCESS
            pcfg = MODEL_PREPROCESS[model_name]
            img_size = pcfg["input_size"]
            resize_short = int(math.floor(img_size / pcfg["crop_pct"]))
            norm_mean = pcfg["mean"]
            norm_std = pcfg["std"]

        return transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Class Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_per_class(model, loader, num_classes, device):
    """Evaluate model and return per-class accuracy + confusion matrix.

    Returns:
        per_class_acc: np.array of shape (num_classes,)
        confusion_matrix: np.array of shape (num_classes, num_classes)
                          confusion_matrix[true, pred] = count
    """
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=True):
                outputs = model(inputs)

            _, preds = outputs.max(1)

            # Update confusion matrix (on CPU for numpy)
            t_np = targets.cpu().numpy()
            p_np = preds.cpu().numpy()
            for t, p in zip(t_np, p_np):
                confusion[t, p] += 1

    # Per-class accuracy = diagonal / row sum
    row_sums = confusion.sum(axis=1)
    per_class_acc = np.where(row_sums > 0, confusion.diagonal() / row_sums, 0.0)

    return per_class_acc, confusion


# ═══════════════════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_analysis(model_name, phase, device, batch_size=256):
    """Run per-class analysis for one model on clean test + all corruptions."""
    print(f"\n{'='*60}")
    print(f"  {MODEL_DISPLAY.get(model_name, model_name)} (Phase {phase})")
    print(f"{'='*60}")

    # Find checkpoint
    ckpt_dir = REPO_ROOT / "results" / f"phase{phase}" / "models"
    ckpt_path = ckpt_dir / f"{model_name}_screwset_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    # Get class names from test folder
    test_dir = SPLIT_DIR / "test"
    class_names = sorted([d for d in os.listdir(test_dir) if (test_dir / d).is_dir()])
    num_classes = len(class_names)
    print(f"  Classes: {num_classes}")

    # Create eval transform
    eval_transform = get_eval_transform(model_name, phase)

    # Load model
    if phase == 1:
        model = create_phase1_model(model_name, num_classes)
    else:
        model = create_phase2_model(model_name, num_classes)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path.name}")

    results = {
        "model": model_name,
        "phase": phase,
        "num_classes": num_classes,
        "class_names": class_names,
    }

    # ── Clean test ────────────────────────────────────────────────────────
    test_ds = ImageFolder(str(test_dir), transform=eval_transform,
                          is_valid_file=is_valid_image)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    print(f"  Evaluating clean test ({len(test_ds)} images)...")
    clean_pca, clean_cm = evaluate_per_class(model, test_loader, num_classes, device)
    results["clean_per_class_acc"] = clean_pca.tolist()
    results["clean_confusion_matrix"] = clean_cm.tolist()
    results["clean_acc"] = float(clean_pca.mean())
    print(f"    Clean mean acc: {clean_pca.mean()*100:.2f}%  "
          f"min: {clean_pca.min()*100:.1f}% ({class_names[clean_pca.argmin()]})  "
          f"max: {clean_pca.max()*100:.1f}% ({class_names[clean_pca.argmax()]})")

    # ── ScrewSet-C corruptions ────────────────────────────────────────────
    results["corrupt_per_class_acc"] = {}
    results["corrupt_confusion_matrices"] = {}

    for corr_type in CORRUPTION_TYPES:
        corr_dir = CORRUPT_ROOT / corr_type
        if not corr_dir.exists():
            print(f"  [SKIP] {corr_type} not found")
            continue

        corr_ds = ImageFolder(str(corr_dir), transform=eval_transform,
                              is_valid_file=is_valid_image)
        corr_loader = DataLoader(corr_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=8, pin_memory=True)

        print(f"  Evaluating {corr_type} ({len(corr_ds)} images)...")
        corr_pca, corr_cm = evaluate_per_class(model, corr_loader, num_classes, device)
        results["corrupt_per_class_acc"][corr_type] = corr_pca.tolist()
        results["corrupt_confusion_matrices"][corr_type] = corr_cm.tolist()
        print(f"    {corr_type}: mean={corr_pca.mean()*100:.2f}%  "
              f"min={corr_pca.min()*100:.1f}%  max={corr_pca.max()*100:.1f}%")

    # Mean corrupt per-class
    all_corr_pca = np.array([results["corrupt_per_class_acc"][c] for c in CORRUPTION_TYPES
                             if c in results["corrupt_per_class_acc"]])
    if len(all_corr_pca) > 0:
        results["mean_corrupt_per_class_acc"] = all_corr_pca.mean(axis=0).tolist()

    # Save JSON
    out_path = RESULTS_DIR / f"{model_name}_per_class.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figures(all_results):
    """Generate all per-class analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

    if not all_results:
        print("[WARN] No results to plot")
        return

    # Get class names from first result
    class_names = all_results[0]["class_names"]
    num_classes = len(class_names)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1: Per-class accuracy heatmap (models × classes) — CLEAN
    # ══════════════════════════════════════════════════════════════════════
    print("\n[FIG] Per-class clean accuracy heatmap...")
    model_names = [r["model"] for r in all_results]
    display_names = [MODEL_DISPLAY.get(m, m) for m in model_names]
    clean_matrix = np.array([r["clean_per_class_acc"] for r in all_results])

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(clean_matrix * 100, aspect="auto", cmap="RdYlGn", vmin=80, vmax=100)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=6)
    ax.set_title("Per-Class Clean Test Accuracy (%)")
    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"per_class_clean_heatmap.{fmt}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 2: Per-class MEAN CORRUPT accuracy heatmap
    # ══════════════════════════════════════════════════════════════════════
    print("[FIG] Per-class mean corrupt accuracy heatmap...")
    corrupt_matrix = np.array([r.get("mean_corrupt_per_class_acc",
                                     [0]*num_classes) for r in all_results])

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(corrupt_matrix * 100, aspect="auto", cmap="RdYlGn", vmin=0, vmax=80)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=6)
    ax.set_title("Per-Class Mean SS-C Accuracy (%)")
    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"per_class_corrupt_heatmap.{fmt}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 3: Class difficulty ranking (avg across all models)
    # ══════════════════════════════════════════════════════════════════════
    print("[FIG] Class difficulty ranking...")
    mean_clean_by_class = clean_matrix.mean(axis=0)
    mean_corr_by_class = corrupt_matrix.mean(axis=0)
    gap_by_class = mean_clean_by_class - mean_corr_by_class

    # Sort by corruption accuracy (ascending = hardest first)
    sort_idx = np.argsort(mean_corr_by_class)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: bar chart of mean corrupt accuracy per class
    ax = axes[0]
    colors = plt.cm.RdYlGn(mean_corr_by_class[sort_idx])
    ax.barh(range(num_classes), mean_corr_by_class[sort_idx] * 100, color=colors)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([class_names[i] for i in sort_idx], fontsize=7)
    ax.set_xlabel("Mean SS-C Accuracy (%)")
    ax.set_title("Class Difficulty Ranking\n(avg across 16 models)")
    ax.set_xlim(0, 80)
    ax.axvline(x=mean_corr_by_class.mean()*100, color="red", linestyle="--",
               alpha=0.7, label=f"Overall mean: {mean_corr_by_class.mean()*100:.1f}%")
    ax.legend(fontsize=8)

    # Right: robustness gap per class
    ax = axes[1]
    colors_gap = plt.cm.Reds(gap_by_class[sort_idx] / gap_by_class.max())
    ax.barh(range(num_classes), gap_by_class[sort_idx] * 100, color=colors_gap)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([class_names[i] for i in sort_idx], fontsize=7)
    ax.set_xlabel("Robustness Gap (Clean − SS-C) in pp")
    ax.set_title("Per-Class Robustness Gap")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"per_class_difficulty_ranking.{fmt}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 4: Per-corruption × per-class heatmap (averaged over models)
    # ══════════════════════════════════════════════════════════════════════
    print("[FIG] Corruption × class heatmap...")
    # Shape: (n_corruptions, n_classes)
    corr_class_matrix = np.zeros((len(CORRUPTION_TYPES), num_classes))
    for ci, corr in enumerate(CORRUPTION_TYPES):
        vals = []
        for r in all_results:
            if corr in r.get("corrupt_per_class_acc", {}):
                vals.append(r["corrupt_per_class_acc"][corr])
        if vals:
            corr_class_matrix[ci] = np.mean(vals, axis=0)

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(corr_class_matrix * 100, aspect="auto", cmap="RdYlGn", vmin=0, vmax=80)
    ax.set_yticks(range(len(CORRUPTION_SHORT)))
    ax.set_yticklabels(CORRUPTION_SHORT)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=6)
    ax.set_title("Corruption Type × Class Accuracy (% avg over 16 models)")
    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"per_class_corruption_heatmap.{fmt}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 5: Confusion matrices for selected models (clean + worst corr)
    # ══════════════════════════════════════════════════════════════════════
    print("[FIG] Confusion matrices...")
    # Select 4 representative models
    selected = ["resnet18", "efficientnetv2_rw_s", "vit_tiny_patch16_224", "convnext_tiny"]
    sel_results = [r for r in all_results if r["model"] in selected]

    for r in sel_results:
        model_name = r["model"]
        disp = MODEL_DISPLAY.get(model_name, model_name)

        # Clean confusion matrix
        cm_clean = np.array(r["clean_confusion_matrix"])
        # Normalize by row (true class)
        row_sums = cm_clean.sum(axis=1, keepdims=True)
        cm_clean_norm = np.where(row_sums > 0, cm_clean / row_sums, 0)

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # Clean
        ax = axes[0]
        sns.heatmap(cm_clean_norm * 100, ax=ax, cmap="Blues", vmin=0, vmax=100,
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
                    linewidths=0.1, linecolor="white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{disp} — Clean Test")
        ax.tick_params(axis="both", labelsize=5)

        # Find worst corruption for this model
        worst_corr = None
        worst_acc = 1.0
        for corr in CORRUPTION_TYPES:
            if corr in r.get("corrupt_per_class_acc", {}):
                acc = np.mean(r["corrupt_per_class_acc"][corr])
                if acc < worst_acc:
                    worst_acc = acc
                    worst_corr = corr

        if worst_corr and worst_corr in r.get("corrupt_confusion_matrices", {}):
            cm_corr = np.array(r["corrupt_confusion_matrices"][worst_corr])
            row_sums_c = cm_corr.sum(axis=1, keepdims=True)
            cm_corr_norm = np.where(row_sums_c > 0, cm_corr / row_sums_c, 0)

            ax = axes[1]
            short_name = CORRUPTION_SHORT[CORRUPTION_TYPES.index(worst_corr)]
            sns.heatmap(cm_corr_norm * 100, ax=ax, cmap="Reds", vmin=0, vmax=100,
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
                        linewidths=0.1, linecolor="white")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{disp} — {short_name} (worst, {worst_acc*100:.1f}%)")
            ax.tick_params(axis="both", labelsize=5)

        fig.suptitle(f"Confusion Matrices: {disp}", fontsize=14, y=1.01)
        fig.tight_layout()
        for fmt in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.{fmt}")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 6: CNN vs ViT per-class comparison
    # ══════════════════════════════════════════════════════════════════════
    print("[FIG] CNN vs ViT per-class comparison...")
    cnn_results = [r for r in all_results if r["model"] in PHASE1_MODELS]
    vit_results = [r for r in all_results if r["model"] in PHASE2_MODELS]

    if cnn_results and vit_results:
        cnn_mean_corr = np.mean([r.get("mean_corrupt_per_class_acc", [0]*num_classes)
                                 for r in cnn_results], axis=0)
        vit_mean_corr = np.mean([r.get("mean_corrupt_per_class_acc", [0]*num_classes)
                                 for r in vit_results], axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(cnn_mean_corr * 100, vit_mean_corr * 100, s=30, alpha=0.8,
                   edgecolors="black", linewidth=0.5)

        # Annotate each class
        for i, name in enumerate(class_names):
            ax.annotate(name, (cnn_mean_corr[i]*100, vit_mean_corr[i]*100),
                       fontsize=5, alpha=0.7, ha="center", va="bottom")

        lims = [0, max(cnn_mean_corr.max(), vit_mean_corr.max()) * 100 + 5]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("CNN Mean SS-C Accuracy (%)")
        ax.set_ylabel("ViT Mean SS-C Accuracy (%)")
        ax.set_title("Per-Class: CNN vs ViT Corruption Robustness")
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        for fmt in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"per_class_cnn_vs_vit.{fmt}")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # CSV: Full per-class results
    # ══════════════════════════════════════════════════════════════════════
    print("[CSV] Writing per-class summary...")
    csv_dir = FIGURES_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Summary table: class × {clean_avg, corrupt_avg, gap, per_corruption_avg}
    lines = ["class," + ",".join(["clean_avg", "corrupt_avg", "gap"] +
                                 [s.replace(",", "") for s in CORRUPTION_SHORT])]
    for ci, cname in enumerate(class_names):
        clean_avg = clean_matrix[:, ci].mean()
        corr_avg = corrupt_matrix[:, ci].mean()
        gap = clean_avg - corr_avg
        per_corr_vals = [f"{corr_class_matrix[j, ci]*100:.2f}" for j in range(len(CORRUPTION_TYPES))]
        lines.append(f"{cname},{clean_avg*100:.2f},{corr_avg*100:.2f},{gap*100:.2f}," +
                     ",".join(per_corr_vals))

    with open(csv_dir / "per_class_summary.csv", "w") as f:
        f.write("\n".join(lines))

    # Full matrix: model × class (corrupt)
    lines2 = ["model," + ",".join(class_names)]
    for r in all_results:
        m = r["model"]
        corr_vals = r.get("mean_corrupt_per_class_acc", [0]*num_classes)
        lines2.append(f"{m}," + ",".join(f"{v*100:.2f}" for v in corr_vals))
    with open(csv_dir / "per_class_corrupt_matrix.csv", "w") as f:
        f.write("\n".join(lines2))

    print(f"\n[DONE] All figures saved to {FIGURES_DIR}")
    print(f"[DONE] CSVs saved to {csv_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Per-class accuracy & confusion matrix analysis")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all'")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="Phase to run (0=both, 1=CNN only, 2=ViT only)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation (only compute per-class stats)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name()}")

    # Build model list
    models_to_run = []
    if args.model != "all":
        if args.model in PHASE1_MODELS:
            models_to_run = [(args.model, 1)]
        elif args.model in PHASE2_MODELS:
            models_to_run = [(args.model, 2)]
        else:
            print(f"[ERROR] Unknown model: {args.model}")
            sys.exit(1)
    else:
        if args.phase in (0, 1):
            models_to_run += [(m, 1) for m in PHASE1_MODELS]
        if args.phase in (0, 2):
            models_to_run += [(m, 2) for m in PHASE2_MODELS]

    print(f"[INFO] Will evaluate {len(models_to_run)} models\n")

    all_results = []
    for i, (model_name, phase) in enumerate(models_to_run):
        print(f"\n  ── Model {i+1}/{len(models_to_run)} ──")
        result = run_model_analysis(model_name, phase, device, args.batch_size)
        if result is not None:
            all_results.append(result)

    # Also load any previously computed results for models not in this run
    for f in sorted(RESULTS_DIR.glob("*_per_class.json")):
        model_name = f.stem.replace("_per_class", "")
        if model_name not in [r["model"] for r in all_results]:
            print(f"  Loading cached: {f.name}")
            with open(f) as fh:
                all_results.append(json.load(fh))

    # Sort: Phase 1 first, then Phase 2
    phase_order = {1: 0, 2: 1}
    all_results.sort(key=lambda r: (phase_order.get(r["phase"], 2),
                                     PHASE1_MODELS.index(r["model"])
                                     if r["model"] in PHASE1_MODELS
                                     else len(PHASE1_MODELS) +
                                     (PHASE2_MODELS.index(r["model"])
                                      if r["model"] in PHASE2_MODELS else 99)))

    if not args.skip_figures:
        generate_figures(all_results)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(all_results)} models evaluated")
    print(f"{'='*60}")
    class_names = all_results[0]["class_names"] if all_results else []
    num_classes = len(class_names)

    if all_results:
        all_corr_pca = np.array([r.get("mean_corrupt_per_class_acc", [0]*num_classes)
                                 for r in all_results])
        avg_by_class = all_corr_pca.mean(axis=0)
        hardest_idx = avg_by_class.argmin()
        easiest_idx = avg_by_class.argmax()
        print(f"  Hardest class: {class_names[hardest_idx]} ({avg_by_class[hardest_idx]*100:.1f}%)")
        print(f"  Easiest class: {class_names[easiest_idx]} ({avg_by_class[easiest_idx]*100:.1f}%)")
        print(f"  Class std:     {avg_by_class.std()*100:.1f}pp")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
