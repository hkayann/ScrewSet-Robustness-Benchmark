#!/usr/bin/env python3
"""
Phase 2: FP32 Vision Transformer Baseline Training & Evaluation
================================================================
Trains (or evaluates pretrained) ViT models on all available datasets
and evaluates robustness against corruptions.

Same evaluation protocol as Phase 1 (CNNs) but with ViT-appropriate
training hyperparameters: AdamW, cosine LR schedule with warmup,
higher num_workers, larger prefetch, torch.compile, and AMP.

Models (all via timm):
    vit_tiny_patch16_224, vit_small_patch16_224,
    deit_tiny_patch16_224, deit_small_patch16_224,
    swin_tiny_patch4_window7_224, mobilevit_s,
    efficientformer_l1, convnext_tiny

Datasets:
    cifar10      — Fine-tune from pretrained, eval CIFAR-10-C
    screwset     — Fine-tune from pretrained, eval ScrewSet-C
    imagenet_a   — Pretrained eval only (200-class subset)
    imagenet_val — Pretrained eval only (1000-class clean)
    imagenet_c   — Pretrained eval only (19 corruptions × 5 severities)
    lens         — Fine-tune from pretrained, eval corrupted

Usage:
    python3 phase2_vit_baselines.py --model vit_small_patch16_224 --dataset cifar10
    python3 phase2_vit_baselines.py --model all --dataset all
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED, ALL_DATASETS,
    VIT_TIMM_PRETRAINED_IDS as TIMM_PRETRAINED_IDS,
    IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA,
)
from src.utils import patch_ipv4, set_seed, make_generator, seed_worker
from src.datasets import is_valid_image, NumpyDataset, SamplesDataset
from src.imagenet_utils import get_imagenet_class_index, build_imagenet_a_mapping
from src.corruption import discover_imagenet_c_corruptions, find_corruption_leaf_dirs

patch_ipv4()

# ═══════════════════════════════════════════════════════════════════════════════
# Phase-specific Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "phase2"
MODELS_SAVE_DIR = RESULTS_DIR / "models"

ALL_MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
    "mobilevit_s",
    "efficientformer_l1",
    "convnext_tiny",
]

# Model-specific parameter counts (for display)
MODEL_INFO = {
    "vit_tiny_patch16_224": {"params": "5.7M", "family": "ViT"},
    "vit_small_patch16_224": {"params": "22M", "family": "ViT"},
    "deit_tiny_patch16_224": {"params": "5.7M", "family": "DeiT"},
    "deit_small_patch16_224": {"params": "22M", "family": "DeiT"},
    "swin_tiny_patch4_window7_224": {"params": "28M", "family": "Swin"},
    "mobilevit_s": {"params": "5.6M", "family": "MobileViT"},
    "efficientformer_l1": {"params": "12M", "family": "EfficientFormer"},
    "convnext_tiny": {"params": "28M", "family": "ConvNeXt"},
}

# Model-specific preprocessing configs (from timm.data.resolve_model_data_config)
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

# ═══════════════════════════════════════════════════════════════════════════════
# GPU & Performance Settings
# ═══════════════════════════════════════════════════════════════════════════════
# Maximize GPU utilization on RTX PRO 6000 (95 GB VRAM)
NUM_WORKERS = 12           # High worker count for data loading
PREFETCH_FACTOR = 4        # Aggressive prefetching
PERSISTENT_WORKERS = True  # Keep workers alive across epochs
USE_AMP = True             # Mixed precision for speed
USE_COMPILE = False        # torch.compile — disable by default (can cause issues with some timm models)
USE_CHANNELS_LAST = True   # Memory format optimization
PIN_MEMORY = True


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(name, num_classes, pretrained=False):
    """Create a ViT model via timm with proper head replacement.

    All Phase 2 models are from timm. timm.create_model handles head replacement
    automatically when num_classes differs from the pretrained model.
    """
    if name not in TIMM_PRETRAINED_IDS:
        raise ValueError(f"Unknown model: {name}. Available: {list(TIMM_PRETRAINED_IDS.keys())}")

    model_id = TIMM_PRETRAINED_IDS[name] if pretrained else name
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {name}: {total_params:,} params ({trainable_params:,} trainable)")

    return model


def prepare_model_for_gpu(model, device):
    """Move model to GPU with performance optimizations."""
    model = model.to(device)

    # Channels-last memory format (can speed up convolution-heavy models)
    if USE_CHANNELS_LAST:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass  # Some models may not support channels_last

    # torch.compile for speed (PyTorch 2.x)
    if USE_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed, continuing without: {e}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_loader(dataset, batch_size):
    """Create a training DataLoader maximized for GPU throughput."""
    gen = make_generator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        worker_init_fn=seed_worker,
        generator=gen,
        drop_last=True,  # Avoid small last batch for stable BN/LN stats
    )


def make_eval_loader(dataset, batch_size):
    """Create an evaluation DataLoader maximized for GPU throughput."""
    return DataLoader(
        dataset,
        batch_size=batch_size * 2,  # Double batch size for eval (no gradients)
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LR Scheduler: Cosine with Linear Warmup
# ═══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for linear warmup.
        total_epochs: Total number of training epochs.
        min_lr: Minimum learning rate at the end of cosine decay.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


# ═══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device, desc="Evaluating"):
    """Standard evaluation with AMP: returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total if total else 0.0, correct / total if total else 0.0


def evaluate_imagenet_a(model, loader, criterion, device, class_mapping, desc="ImageNet-A"):
    """Evaluate on ImageNet-A with proper class mapping."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
            _, preds = outputs.max(1)
            mapped_targets = torch.tensor(
                [class_mapping[t.item()] for t in targets],
                device=device, dtype=torch.long,
            )
            correct += preds.eq(mapped_targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total else 0.0
    return acc


def train_model(model, train_loader, val_loader, device, args, model_tag):
    """Train a ViT model with AdamW + cosine warmup + AMP + early stopping."""
    criterion = nn.CrossEntropyLoss()

    # AdamW with weight decay — standard for ViTs
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine schedule with warmup
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        min_lr=args.min_lr,
    )

    # AMP scaler
    scaler = GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"
    no_improve = 0
    epoch_times = []

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, targets in tqdm(
            train_loader,
            desc=f"[{model_tag}] Epoch {epoch}/{args.num_epochs} (lr={optimizer.param_groups[0]['lr']:.2e})",
            leave=False,
        ):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            # Gradient clipping — important for ViT stability
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_acc = correct / total
        train_loss /= total

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            desc=f"[{model_tag}] Val {epoch}/{args.num_epochs}",
        )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"[{model_tag}] Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"Time={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"[{model_tag}] Early stopping after {epoch} epochs "
                    f"(no improvement for {args.patience})"
                )
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(
        f"[{model_tag}] Training complete. Best val acc: {best_val_acc:.4f} | "
        f"Avg epoch: {avg_epoch_time:.1f}s"
    )
    return best_val_acc, str(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_cifar10(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  CIFAR-10 — {model_name} ({MODEL_INFO[model_name]['family']})")
    print(f"{'='*70}")

    CIFAR_ROOT = DATA_DIR / "cifar10"
    CIFAR_C_DIR = DATA_DIR / "CIFAR-10-C"

    # Use model-specific preprocessing (normalization + input size) from timm
    pcfg = MODEL_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    # Load CIFAR-10
    full_train = datasets.CIFAR10(str(CIFAR_ROOT), train=True, download=False,
                                   transform=train_transform)
    test_ds = datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False,
                                transform=eval_transform)

    # Stratified train/val split (45000 / 5000)
    set_seed()
    labels = np.array(full_train.targets)
    train_idx, val_idx = [], []
    for cls in range(10):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = 500
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())

    train_ds = Subset(full_train, train_idx)

    full_val = datasets.CIFAR10(str(CIFAR_ROOT), train=True, download=False,
                                 transform=eval_transform)
    val_ds = Subset(full_val, val_idx)

    train_loader = make_train_loader(train_ds, args.batch_size)
    val_loader = make_eval_loader(val_ds, args.batch_size)
    test_loader = make_eval_loader(test_ds, args.batch_size)

    NUM_CLASSES = 10
    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Create model
    set_seed()
    model = create_model(model_name, NUM_CLASSES, pretrained=True)
    model = prepare_model_for_gpu(model, device)
    model_tag = f"{model_name}_cifar10"

    # Train
    best_val_acc, ckpt_path = train_model(model, train_loader, val_loader,
                                           device, args, model_tag)

    # Evaluate clean test
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                    desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test: Loss={test_loss:.4f} Acc={test_acc:.4f}")

    # Evaluate CIFAR-10-C
    corrupt_results = {}
    if CIFAR_C_DIR.exists():
        corr_labels = np.load(str(CIFAR_C_DIR / "labels.npy"))
        for fname in sorted(os.listdir(CIFAR_C_DIR)):
            if not fname.endswith(".npy") or fname == "labels.npy":
                continue
            images = np.load(str(CIFAR_C_DIR / fname))
            ds = NumpyDataset(images, corr_labels, eval_transform)
            loader = make_eval_loader(ds, args.batch_size)
            _, acc = evaluate(model, loader, criterion, device,
                               desc=f"[{model_tag}] C-{fname}")
            corrupt_results[fname.replace(".npy", "")] = acc
            print(f"  CIFAR-10-C {fname.replace('.npy', '')}: {acc:.4f}")
    else:
        print("[WARN] CIFAR-10-C directory not found, skipping corruption eval")

    # Save results
    result = {
        "model": model_name,
        "model_family": MODEL_INFO[model_name]["family"],
        "dataset": "cifar10",
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": [IMG_SIZE, IMG_SIZE],
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "scheduler": "CosineWarmup",
        "amp": USE_AMP,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_cifar10_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ScrewSet pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_screwset(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  ScrewSet — {model_name} ({MODEL_INFO[model_name]['family']})")
    print(f"{'='*70}")

    SPLIT_DIR = DATA_DIR / "screwset_split"
    CORRUPT_ROOT = DATA_DIR / "screwset_c"

    # Use model-specific preprocessing (normalization + input size) from timm
    pcfg = MODEL_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    RESIZE_SHORT = int(math.floor(IMG_SIZE / pcfg["crop_pct"]))
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=train_transform,
                            is_valid_file=is_valid_image)
    val_ds = ImageFolder(str(SPLIT_DIR / "validation"), transform=eval_transform,
                          is_valid_file=is_valid_image)
    test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_transform,
                           is_valid_file=is_valid_image)

    NUM_CLASSES = len(train_ds.classes)
    print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = make_train_loader(train_ds, args.batch_size)
    val_loader = make_eval_loader(val_ds, args.batch_size)
    test_loader = make_eval_loader(test_ds, args.batch_size)

    # Create model
    set_seed()
    model = create_model(model_name, NUM_CLASSES, pretrained=True)
    model = prepare_model_for_gpu(model, device)
    model_tag = f"{model_name}_screwset"

    # Train
    best_val_acc, ckpt_path = train_model(model, train_loader, val_loader,
                                           device, args, model_tag)

    # Evaluate clean test
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                    desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test: Loss={test_loss:.4f} Acc={test_acc:.4f}")

    # Evaluate ScrewSet-C
    corrupt_results = {}
    if CORRUPT_ROOT.exists():
        for corrupt_type in sorted(os.listdir(CORRUPT_ROOT)):
            corrupt_dir = CORRUPT_ROOT / corrupt_type
            if not corrupt_dir.is_dir():
                continue
            try:
                ds = ImageFolder(str(corrupt_dir), transform=eval_transform,
                                  is_valid_file=is_valid_image)
                loader = make_eval_loader(ds, args.batch_size)
                _, acc = evaluate(model, loader, criterion, device,
                                   desc=f"[{model_tag}] {corrupt_type}")
                corrupt_results[corrupt_type] = acc
                print(f"  ScrewSet-C {corrupt_type}: {acc:.4f}")
            except Exception as e:
                print(f"  [ERROR] {corrupt_type}: {e}")
                corrupt_results[corrupt_type] = None
    else:
        print("[WARN] ScrewSet-C directory not found, skipping corruption eval")

    # Save results
    result = {
        "model": model_name,
        "model_family": MODEL_INFO[model_name]["family"],
        "dataset": "screwset",
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": f"Resize({RESIZE_SHORT})+CenterCrop({IMG_SIZE})",
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "scheduler": "CosineWarmup",
        "amp": USE_AMP,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_screwset_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet-A pipeline (pretrained eval only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_a(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet-A — {model_name} ({MODEL_INFO[model_name]['family']})")
    print(f"{'='*70}")

    IMAGENET_A_DIR = DATA_DIR / "imagenet-a"
    if not IMAGENET_A_DIR.exists():
        print("[ERROR] ImageNet-A directory not found — skipping")
        return None

    # Use model-specific preprocessing (normalization + input size) from timm
    pcfg = MODEL_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    RESIZE_SIZE = int(math.floor(IMG_SIZE / pcfg["crop_pct"]))
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    class_mapping = build_imagenet_a_mapping(str(IMAGENET_A_DIR))

    ds = ImageFolder(str(IMAGENET_A_DIR), transform=eval_transform,
                      is_valid_file=is_valid_image)
    loader = make_eval_loader(ds, args.batch_size)

    print(f"[INFO] ImageNet-A: {len(ds)} images, {len(ds.classes)} classes")

    set_seed()
    model = create_model(model_name, num_classes=1000, pretrained=True)
    model = prepare_model_for_gpu(model, device)
    model_tag = f"{model_name}_imagenet_a"

    criterion = nn.CrossEntropyLoss()
    acc = evaluate_imagenet_a(model, loader, criterion, device, class_mapping,
                               desc=f"[{model_tag}] Evaluating")
    print(f"[{model_tag}] ImageNet-A Accuracy: {acc:.4f}")

    result = {
        "model": model_name,
        "model_family": MODEL_INFO[model_name]["family"],
        "dataset": "imagenet_a",
        "pretrained": True,
        "num_imagenet_a_classes": len(class_mapping),
        "imagenet_a_acc": acc,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": RESIZE_SIZE,
        "crop": IMG_SIZE,
        "batch_size": args.batch_size,
        "amp": USE_AMP,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_imagenet_a_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet Validation (clean, 1000-class, eval-only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_val(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet Val — {model_name} ({MODEL_INFO[model_name]['family']})")
    print(f"{'='*70}")

    IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
    if not IMAGENET_VAL_DIR.exists():
        print("[ERROR] ImageNet val directory not found — skipping")
        return None

    # Use model-specific preprocessing (normalization + input size) from timm
    pcfg = MODEL_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    RESIZE_SIZE = int(math.floor(IMG_SIZE / pcfg["crop_pct"]))
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    ds = ImageFolder(str(IMAGENET_VAL_DIR), transform=eval_transform,
                      is_valid_file=is_valid_image)
    loader = make_eval_loader(ds, args.batch_size)

    print(f"[INFO] ImageNet Val: {len(ds)} images, {len(ds.classes)} classes")

    set_seed()
    model = create_model(model_name, num_classes=1000, pretrained=True)
    model = prepare_model_for_gpu(model, device)
    model_tag = f"{model_name}_imagenet_val"

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, loader, criterion, device,
                                  desc=f"[{model_tag}] Evaluating")
    print(f"[{model_tag}] ImageNet Val: Loss={val_loss:.4f} Acc={val_acc:.4f}")

    # Per-class accuracy
    model.eval()
    class_correct = {}
    class_total = {}
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"[{model_tag}] Per-class", leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
            _, preds = outputs.max(1)
            for t, p in zip(targets, preds):
                t_item = t.item()
                class_total[t_item] = class_total.get(t_item, 0) + 1
                class_correct[t_item] = class_correct.get(t_item, 0) + (1 if p.item() == t_item else 0)

    per_class_acc = {k: class_correct.get(k, 0) / class_total[k]
                     for k in sorted(class_total.keys())}
    mean_per_class_acc = sum(per_class_acc.values()) / len(per_class_acc) if per_class_acc else 0.0

    result = {
        "model": model_name,
        "model_family": MODEL_INFO[model_name]["family"],
        "dataset": "imagenet_val",
        "pretrained": True,
        "num_classes": len(ds.classes),
        "num_images": len(ds),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "mean_per_class_acc": mean_per_class_acc,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": RESIZE_SIZE,
        "crop": IMG_SIZE,
        "batch_size": args.batch_size,
        "amp": USE_AMP,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_imagenet_val_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet-C (19 corruptions × 5 severities, eval-only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_c(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet-C — {model_name} ({MODEL_INFO[model_name]['family']})")
    print(f"{'='*70}")

    IMAGENET_C_DIR = DATA_DIR / "imagenet-c"
    if not IMAGENET_C_DIR.exists():
        print("[ERROR] ImageNet-C directory not found — skipping")
        return None

    corruptions = discover_imagenet_c_corruptions(IMAGENET_C_DIR)
    if not corruptions:
        print("[ERROR] No corruption directories found in ImageNet-C — skipping")
        return None

    print(f"[INFO] Found {len(corruptions)} corruptions: {list(corruptions.keys())}")

    # Use model-specific preprocessing (normalization + input size) from timm
    pcfg = MODEL_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    RESIZE_SIZE = int(math.floor(IMG_SIZE / pcfg["crop_pct"]))
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    set_seed()
    model = create_model(model_name, num_classes=1000, pretrained=True)
    model = prepare_model_for_gpu(model, device)
    model_tag = f"{model_name}_imagenet_c"

    criterion = nn.CrossEntropyLoss()
    corruption_results = {}
    severity_levels = [1, 2, 3, 4, 5]

    for cname, cpath in sorted(corruptions.items()):
        corruption_results[cname] = {}
        for sev in severity_levels:
            sev_dir = cpath / str(sev)
            if not sev_dir.exists():
                print(f"  [WARN] {cname}/severity-{sev} not found, skipping")
                continue

            ds = ImageFolder(str(sev_dir), transform=eval_transform,
                              is_valid_file=is_valid_image)
            loader = make_eval_loader(ds, args.batch_size)

            _, acc = evaluate(model, loader, criterion, device,
                               desc=f"[{model_tag}] {cname}/s{sev}")
            corruption_results[cname][str(sev)] = acc
            print(f"  {cname} sev-{sev}: {acc:.4f}")

        accs = [v for v in corruption_results[cname].values() if v is not None]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        corruption_results[cname]["mean"] = mean_acc
        print(f"  {cname} mean: {mean_acc:.4f}")

    # Aggregate metrics
    std15_accs = []
    for cname in IMAGENET_C_CORRUPTIONS_15:
        if cname in corruption_results and "mean" in corruption_results[cname]:
            std15_accs.append(corruption_results[cname]["mean"])
    mean_acc_15 = sum(std15_accs) / len(std15_accs) if std15_accs else 0.0
    mce_15 = 1.0 - mean_acc_15

    all_accs = [cr["mean"] for cr in corruption_results.values()
                if "mean" in cr and cr["mean"] is not None]
    mean_acc_all = sum(all_accs) / len(all_accs) if all_accs else 0.0
    mce_all = 1.0 - mean_acc_all

    print(f"\n[{model_tag}] Mean Acc (15 std): {mean_acc_15:.4f}  mCE: {mce_15:.4f}")
    print(f"[{model_tag}] Mean Acc (all {len(all_accs)}): {mean_acc_all:.4f}  mCE: {mce_all:.4f}")

    result = {
        "model": model_name,
        "model_family": MODEL_INFO[model_name]["family"],
        "dataset": "imagenet_c",
        "pretrained": True,
        "num_corruptions_evaluated": len(corruption_results),
        "corruption_results": corruption_results,
        "mean_acc_15_std": mean_acc_15,
        "mce_15_std": mce_15,
        "mean_acc_all": mean_acc_all,
        "mce_all": mce_all,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": RESIZE_SIZE,
        "crop": IMG_SIZE,
        "batch_size": args.batch_size,
        "amp": USE_AMP,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_imagenet_c_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Lens / ImageNet-ES pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_lens(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  Lens / ImageNet-ES — {model_name} ({MODEL_INFO[model_name]['family']})")
    print(f"{'='*70}")

    LENS_DIR = DATA_DIR / "lens_split"
    if not LENS_DIR.exists():
        print("[ERROR] Lens split directory not found — skipping")
        return None

    # Use model-specific preprocessing (normalization + input size) from timm
    pcfg = MODEL_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    RESIZE_SIZE = int(math.floor(IMG_SIZE / pcfg["crop_pct"]))
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    train_ds = ImageFolder(str(LENS_DIR / "train"), transform=train_transform)
    val_ds = ImageFolder(str(LENS_DIR / "validation"), transform=eval_transform)
    test_ds = ImageFolder(str(LENS_DIR / "test"), transform=eval_transform)

    NUM_CLASSES = len(train_ds.classes)
    print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        print("[ERROR] Class mismatch between train/val/test splits!")
        return None

    train_loader = make_train_loader(train_ds, args.batch_size)
    val_loader = make_eval_loader(val_ds, args.batch_size)
    test_loader = make_eval_loader(test_ds, args.batch_size)

    set_seed()
    model = create_model(model_name, NUM_CLASSES, pretrained=True)
    model = prepare_model_for_gpu(model, device)
    model_tag = f"{model_name}_lens"

    best_val_acc, ckpt_path = train_model(model, train_loader, val_loader,
                                           device, args, model_tag)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                    desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test: Loss={test_loss:.4f} Acc={test_acc:.4f}")

    # Evaluate corrupted subsets
    corrupt_root = LENS_DIR / "corrupted"
    corrupt_results = {}
    if corrupt_root.exists():
        leaf_dirs = find_corruption_leaf_dirs(corrupt_root)
        print(f"[INFO] Found {len(leaf_dirs)} corrupted subsets")
        for leaf in leaf_dirs:
            rel = str(leaf.relative_to(corrupt_root))
            try:
                ds = ImageFolder(str(leaf), transform=eval_transform)
                if ds.classes != train_ds.classes:
                    print(f"  [WARN] Class mismatch for {rel}, skipping")
                    continue
                loader = make_eval_loader(ds, args.batch_size)
                _, acc = evaluate(model, loader, criterion, device,
                                   desc=f"[{model_tag}] {rel}")
                corrupt_results[rel] = acc
                print(f"  Lens corrupt {rel}: {acc:.4f}")
            except Exception as e:
                print(f"  [ERROR] {rel}: {e}")
                corrupt_results[rel] = None
    else:
        print("[WARN] Lens corrupted directory not found")

    result = {
        "model": model_name,
        "model_family": MODEL_INFO[model_name]["family"],
        "dataset": "lens",
        "pretrained_finetune": True,
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": RESIZE_SIZE,
        "crop": IMG_SIZE,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "scheduler": "CosineWarmup",
        "amp": USE_AMP,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_lens_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_RUNNERS = {
    "cifar10": run_cifar10,
    "screwset": run_screwset,
    "imagenet_a": run_imagenet_a,
    "imagenet_val": run_imagenet_val,
    "imagenet_c": run_imagenet_c,
    "lens": run_lens,
}

# Map dataset key → JSON suffix for skip logic
DATASET_JSON_SUFFIX = {
    "cifar10": "cifar10",
    "screwset": "screwset",
    "imagenet_a": "imagenet_a",
    "imagenet_val": "imagenet_val",
    "imagenet_c": "imagenet_c",
    "lens": "lens",
}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: FP32 ViT Baseline Training & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=ALL_MODELS + ["all"],
                        help="ViT architecture to use")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=ALL_DATASETS + ["all"],
                        help="Dataset to train/evaluate on")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (eval uses 2x)")
    parser.add_argument("--num-epochs", type=int, default=30,
                        help="Max training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum LR at end of cosine decay")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    # Setup directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] timm: {timm.__version__}")
    print(f"[INFO] AMP: {USE_AMP} | Channels-Last: {USE_CHANNELS_LAST} | "
          f"Compile: {USE_COMPILE}")
    print(f"[INFO] Workers: {NUM_WORKERS} | Prefetch: {PREFETCH_FACTOR}")
    print(f"[INFO] Optimizer: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
    print(f"[INFO] Scheduler: CosineWarmup (warmup={args.warmup_epochs}, "
          f"min_lr={args.min_lr})")

    model_list = ALL_MODELS if args.model == "all" else [args.model]
    dataset_list = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    total_runs = len(model_list) * len(dataset_list)
    run_idx = 0
    failed_runs = []

    for ds_name in dataset_list:
        runner = DATASET_RUNNERS[ds_name]
        ds_suffix = DATASET_JSON_SUFFIX[ds_name]
        for m_name in model_list:
            run_idx += 1

            # ── Auto-resume: skip if JSON already exists ──
            json_path = RESULTS_DIR / f"{m_name}_{ds_suffix}_baselines.json"
            if json_path.exists():
                print(f"\n[SKIP] RUN {run_idx}/{total_runs}: "
                      f"{m_name} × {ds_name} — {json_path.name} exists")
                continue

            print(f"\n{'#'*70}")
            print(f"  RUN {run_idx}/{total_runs}: {m_name} × {ds_name}")
            print(f"  Family: {MODEL_INFO[m_name]['family']} | "
                  f"Params: {MODEL_INFO[m_name]['params']}")
            print(f"{'#'*70}")

            set_seed()
            try:
                runner(m_name, args, device)
            except Exception as e:
                print(f"[ERROR] {m_name} × {ds_name} failed: {e}")
                import traceback
                traceback.print_exc()
                failed_runs.append(f"{m_name} × {ds_name}")

    print(f"\n{'='*70}")
    print(f"  ALL RUNS COMPLETE ({run_idx - len(failed_runs)}/{total_runs} succeeded)")
    if failed_runs:
        print(f"  FAILED: {', '.join(failed_runs)}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
