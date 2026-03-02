#!/usr/bin/env python3
"""
Comprehensive Augmentation Baselines — ScrewSet
=================================================
Trains 4 representative models on ScrewSet with DIFFERENT augmentation
strategies, then evaluates on ScrewSet-C corruptions.  Results are
compared against the no-augmentation Phase 1/2 baselines and AugMix.

Models (same 4 as AugMix experiment):
    resnet18             — Worst CNN on SS-C baseline  (9.23 %)
    efficientnetv2_rw_s  — Best CNN on SS-C baseline   (11.28 %)
    vit_tiny_patch16_224 — Small ViT                   (21.17 %)
    convnext_tiny        — Best supervised on SS-C      (45.39 %)

Augmentation methods (spanning 2018-2025):

  TRAINING-TIME:
    cutmix_mixup      — CutMix (ICCV 2019) + MixUp (ICLR 2018)
                         via timm.data.Mixup; standard recipe in
                         ConvNeXt / DeiT / modern ViT training (2023-2025)
    randaugment        — RandAugment (NeurIPS 2020)
                         torchvision.transforms.RandAugment
    trivialaugment     — TrivialAugmentWide (ICCV 2021)
                         torchvision.transforms.TrivialAugmentWide
    3augment           — 3Augment from DeiT-III (ECCV 2022),
                         de-facto standard ViT recipe 2023-2025.
                         Grayscale + Solarize + GaussianBlur (each p=0.1)

  TEST-TIME (inference only, no re-training):
    tta                — Test-Time Augmentation (widely adopted 2024-2025):
                         5-crop × original+flip  = 10-view ensemble.
                         Applied to existing Phase 1/2 baseline checkpoints.

Usage:
    python3 augmentation_baselines.py --method trivialaugment --model resnet18
    python3 augmentation_baselines.py --method cutmix_mixup   --model all
    python3 augmentation_baselines.py --method tta             --model all
    python3 augmentation_baselines.py --method all             --model all
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
import torch.nn.functional as F
import torch.optim as optim
import timm
from timm.data import Mixup
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED,
    CNN_TIMM_PRETRAINED_IDS,
    VIT_TIMM_PRETRAINED_IDS,
)
from src.utils import patch_ipv4, set_seed, make_generator, seed_worker
from src.datasets import is_valid_image
from src.evaluation import evaluate

patch_ipv4()

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "augmentation"
MODELS_SAVE_DIR = RESULTS_DIR / "models"
SPLIT_DIR = DATA_DIR / "screwset_split"
CORRUPT_ROOT = DATA_DIR / "screwset_c"

# Phase 1/2 model checkpoint directories (for TTA)
PHASE1_MODELS = REPO_ROOT / "results" / "phase1" / "models"
PHASE2_MODELS = REPO_ROOT / "results" / "phase2" / "models"

# CNN models use fixed normalization; ViTs use model-specific
SCREWSET_NORM = {"mean": [0.7750, 0.7343, 0.6862], "std": [0.0802, 0.0838, 0.0871]}

CNN_MODELS = ["resnet18", "efficientnetv2_rw_s"]
VIT_MODELS = ["vit_tiny_patch16_224", "convnext_tiny"]
ALL_MODELS = CNN_MODELS + VIT_MODELS

TRAINING_METHODS = ["cutmix_mixup", "randaugment", "trivialaugment", "3augment"]
ALL_METHODS = TRAINING_METHODS + ["tta"]

# ViT model-specific preprocessing (from Phase 2)
VIT_PREPROCESS = {
    "vit_tiny_patch16_224": {
        "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
        "input_size": 224, "crop_pct": 0.9,
    },
    "convnext_tiny": {
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "input_size": 224, "crop_pct": 0.875,
    },
}

# GPU settings
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
USE_AMP = True
USE_CHANNELS_LAST = True
PIN_MEMORY = True


# ═══════════════════════════════════════════════════════════════════════════════
# Soft Cross-Entropy for CutMix/MixUp (soft target labels)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss for soft (one-hot) targets from CutMix/MixUp."""
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory (shared with AugMix script)
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(name, num_classes, pretrained=False):
    """Create CNN (Phase 1) or ViT (Phase 2) model."""
    if name in VIT_TIMM_PRETRAINED_IDS:
        model_id = VIT_TIMM_PRETRAINED_IDS[name] if pretrained else name
        return timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
    if name in CNN_TIMM_PRETRAINED_IDS:
        model_id = CNN_TIMM_PRETRAINED_IDS[name] if pretrained else name
        return timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")


def prepare_model(model, device, is_vit=False):
    """Move model to GPU with optional ViT optimizations."""
    model = model.to(device)
    if is_vit and USE_CHANNELS_LAST:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3Augment Transform (DeiT-III recipe, ECCV 2022 → standard 2023-2025)
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeAugment:
    """DeiT-III '3Augment' recipe: randomly apply one of three augmentations.

    Each of Grayscale / Solarize / GaussianBlur is applied independently
    with probability p=0.1. This simple recipe was shown to match or exceed
    RandAugment and AutoAugment for ViT training.

    Reference: Touvron et al., "DeiT III: Revenge of the ViT" (ECCV 2022)
    Widely adopted as the default augmentation in ViT training recipes (2023-2025).
    """
    def __init__(self, p_each=0.1):
        self.grayscale = transforms.RandomGrayscale(p=p_each)
        self.solarize = transforms.RandomSolarize(threshold=128, p=p_each)
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.p_blur = p_each

    def __call__(self, img):
        img = self.grayscale(img)
        img = self.solarize(img)
        if torch.rand(1).item() < self.p_blur:
            img = self.blur(img)
        return img


# ═══════════════════════════════════════════════════════════════════════════════
# Transform Builders per Method
# ═══════════════════════════════════════════════════════════════════════════════

def _get_norm_and_sizes(model_name):
    """Return normalization params and image sizes for model."""
    is_vit = model_name in VIT_MODELS
    if is_vit:
        pcfg = VIT_PREPROCESS[model_name]
        img_size = pcfg["input_size"]
        resize_short = int(math.floor(img_size / pcfg["crop_pct"]))
        norm_mean, norm_std = pcfg["mean"], pcfg["std"]
    else:
        resize_short = 240
        img_size = 240
        norm_mean = SCREWSET_NORM["mean"]
        norm_std = SCREWSET_NORM["std"]
    return is_vit, img_size, resize_short, norm_mean, norm_std


def build_transforms_randaugment(model_name, num_ops=2, magnitude=9):
    """RandAugment (Cubuk et al., NeurIPS 2020)."""
    is_vit, img_size, resize_short, norm_mean, norm_std = _get_norm_and_sizes(model_name)

    aug = transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)

    if is_vit:
        train_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomHorizontalFlip(),
            aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    return train_tf, eval_tf, norm_mean, norm_std


def build_transforms_trivialaugment(model_name):
    """TrivialAugmentWide (Müller & Hutter, ICCV 2021)."""
    is_vit, img_size, resize_short, norm_mean, norm_std = _get_norm_and_sizes(model_name)

    aug = transforms.TrivialAugmentWide()

    if is_vit:
        train_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomHorizontalFlip(),
            aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    return train_tf, eval_tf, norm_mean, norm_std


def build_transforms_3augment(model_name):
    """3Augment (Touvron et al., DeiT-III, ECCV 2022 — standard 2023-2025)."""
    is_vit, img_size, resize_short, norm_mean, norm_std = _get_norm_and_sizes(model_name)

    aug = ThreeAugment(p_each=0.1)

    if is_vit:
        train_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomHorizontalFlip(),
            aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    return train_tf, eval_tf, norm_mean, norm_std


def build_transforms_cutmix_mixup(model_name):
    """CutMix+MixUp — transforms are vanilla; mixing happens in-batch.

    Returns the same eval transform and a VANILLA train transform
    (just resize + flip). CutMix/MixUp is applied at the batch level
    in the training loop via timm.data.Mixup.
    """
    is_vit, img_size, resize_short, norm_mean, norm_std = _get_norm_and_sizes(model_name)

    if is_vit:
        train_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    return train_tf, eval_tf, norm_mean, norm_std


def build_transforms_for_method(method, model_name):
    """Dispatch to the appropriate transform builder."""
    if method == "randaugment":
        return build_transforms_randaugment(model_name)
    elif method == "trivialaugment":
        return build_transforms_trivialaugment(model_name)
    elif method == "3augment":
        return build_transforms_3augment(model_name)
    elif method == "cutmix_mixup":
        return build_transforms_cutmix_mixup(model_name)
    else:
        raise ValueError(f"Unknown method for transforms: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loops
# ═══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup (for ViTs)."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs]


def train_cnn(model, train_loader, val_loader, device, args, model_tag,
              mixup_fn=None, criterion_train=None):
    """Phase 1 training: Adam + early stopping, no AMP."""
    criterion_eval = nn.CrossEntropyLoss()
    if criterion_train is None:
        criterion_train = criterion_eval
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"
    no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, targets in tqdm(train_loader,
                                     desc=f"[{model_tag}] Epoch {epoch}/{args.num_epochs}",
                                     leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # CutMix/MixUp if active
            if mixup_fn is not None:
                inputs, targets_mixed = mixup_fn(inputs, targets)
            else:
                targets_mixed = targets

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_train(outputs, targets_mixed)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            if mixup_fn is not None:
                # For mixed targets, use argmax for accuracy estimate
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
            else:
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total
        train_loss /= total

        val_loss, val_acc = evaluate(model, val_loader, criterion_eval, device,
                                      desc=f"[{model_tag}] Val {epoch}")

        print(f"[{model_tag}] Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[{model_tag}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"[{model_tag}] Best val acc: {best_val_acc:.4f}")
    return best_val_acc, str(ckpt_path)


def train_vit(model, train_loader, val_loader, device, args, model_tag,
              mixup_fn=None, criterion_train=None):
    """Phase 2 training: AdamW + cosine warmup + AMP + early stopping."""
    criterion_eval = nn.CrossEntropyLoss()
    if criterion_train is None:
        criterion_train = criterion_eval
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                       total_epochs=args.num_epochs, min_lr=args.min_lr)
    scaler = GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"
    no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, targets in tqdm(
            train_loader,
            desc=f"[{model_tag}] Epoch {epoch}/{args.num_epochs} "
                 f"(lr={optimizer.param_groups[0]['lr']:.2e})",
            leave=False,
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # CutMix/MixUp if active
            if mixup_fn is not None:
                inputs, targets_mixed = mixup_fn(inputs, targets)
            else:
                targets_mixed = targets

            if USE_CHANNELS_LAST:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion_train(outputs, targets_mixed)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            if mixup_fn is not None:
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
            else:
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss /= total

        val_loss, val_acc = evaluate(model, val_loader, criterion_eval, device,
                                      desc=f"[{model_tag}] Val {epoch}",
                                      use_amp=USE_AMP,
                                      use_channels_last=USE_CHANNELS_LAST)

        elapsed = time.time() - epoch_start
        print(f"[{model_tag}] Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
              f"LR={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[{model_tag}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"[{model_tag}] Best val acc: {best_val_acc:.4f}")
    return best_val_acc, str(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Corruption Evaluation (shared)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_corruptions(model, eval_transform, device, args, model_tag, is_vit):
    """Evaluate a model on all ScrewSet-C corruptions."""
    criterion = nn.CrossEntropyLoss()
    corrupt_results = {}

    if not CORRUPT_ROOT.exists():
        print("[WARN] ScrewSet-C directory not found, skipping corruption eval")
        return corrupt_results

    for corrupt_type in sorted(os.listdir(CORRUPT_ROOT)):
        corrupt_dir = CORRUPT_ROOT / corrupt_type
        if not corrupt_dir.is_dir():
            continue
        try:
            ds = ImageFolder(str(corrupt_dir), transform=eval_transform,
                              is_valid_file=is_valid_image)
            if is_vit:
                loader = DataLoader(
                    ds, batch_size=args.batch_size * 2, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                    prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
                )
                _, acc = evaluate(model, loader, criterion, device,
                                  desc=f"[{model_tag}] {corrupt_type}",
                                  use_amp=USE_AMP,
                                  use_channels_last=USE_CHANNELS_LAST)
            else:
                loader = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True,
                )
                _, acc = evaluate(model, loader, criterion, device,
                                  desc=f"[{model_tag}] {corrupt_type}")
            corrupt_results[corrupt_type] = acc
            print(f"  SS-C {corrupt_type}: {acc:.4f}")
        except Exception as e:
            print(f"  [ERROR] {corrupt_type}: {e}")
            corrupt_results[corrupt_type] = None

    return corrupt_results


# ═══════════════════════════════════════════════════════════════════════════════
# Training-Time Method Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_training_method(method, model_name, args, device):
    """Train model on ScrewSet with given augmentation, then evaluate."""
    is_vit = model_name in VIT_MODELS
    phase = "Phase2-ViT" if is_vit else "Phase1-CNN"

    print(f"\n{'='*70}")
    print(f"  {method.upper()} ScrewSet — {model_name}  [{phase}]")
    print(f"{'='*70}")

    # Build transforms
    train_tf, eval_tf, norm_mean, norm_std = build_transforms_for_method(method, model_name)

    # Load datasets
    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=train_tf,
                            is_valid_file=is_valid_image)
    val_ds = ImageFolder(str(SPLIT_DIR / "validation"), transform=eval_tf,
                          is_valid_file=is_valid_image)
    test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_tf,
                           is_valid_file=is_valid_image)
    NUM_CLASSES = len(train_ds.classes)
    print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    # DataLoaders
    gen = make_generator()
    if is_vit:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
            worker_init_fn=seed_worker, generator=gen, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker, generator=gen,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    # Create model
    set_seed()
    pretrained = is_vit  # Phase 1 CNNs: scratch, Phase 2 ViTs: pretrained
    model = create_model(model_name, NUM_CLASSES, pretrained=pretrained)
    model = prepare_model(model, device, is_vit=is_vit)
    model_tag = f"{model_name}_screwset_{method}"

    # CutMix/MixUp: setup batch-level mixing
    mixup_fn = None
    criterion_train = None
    if method == "cutmix_mixup":
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=NUM_CLASSES,
        )
        criterion_train = SoftTargetCrossEntropy()
        print(f"[INFO] CutMix+MixUp: mixup_alpha=0.8, cutmix_alpha=1.0, "
              f"switch_prob=0.5, label_smoothing=0.1")

    # Train
    start_time = time.time()
    if is_vit:
        best_val_acc, ckpt_path = train_vit(
            model, train_loader, val_loader, device, args, model_tag,
            mixup_fn=mixup_fn, criterion_train=criterion_train)
    else:
        best_val_acc, ckpt_path = train_cnn(
            model, train_loader, val_loader, device, args, model_tag,
            mixup_fn=mixup_fn, criterion_train=criterion_train)
    train_time = time.time() - start_time

    # Evaluate clean test
    criterion = nn.CrossEntropyLoss()
    if is_vit:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                        desc=f"[{model_tag}] Clean Test",
                                        use_amp=USE_AMP,
                                        use_channels_last=USE_CHANNELS_LAST)
    else:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                        desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test: Loss={test_loss:.4f} Acc={test_acc:.4f}")

    # Evaluate ScrewSet-C
    corrupt_results = evaluate_corruptions(model, eval_tf, device, args, model_tag, is_vit)

    # Compute mean corrupt accuracy
    valid_accs = [v for v in corrupt_results.values() if v is not None]
    mean_corrupt_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

    # Build method-specific metadata
    method_meta = {"method": method}
    if method == "cutmix_mixup":
        method_meta.update({
            "mixup_alpha": 0.8, "cutmix_alpha": 1.0,
            "switch_prob": 0.5, "label_smoothing": 0.1,
            "year": "2018/2019",
            "reference": "Zhang+2018 (MixUp, ICLR); Yun+2019 (CutMix, ICCV)",
        })
    elif method == "randaugment":
        method_meta.update({
            "num_ops": 2, "magnitude": 9,
            "year": "2020",
            "reference": "Cubuk+2020 (RandAugment, NeurIPS)",
        })
    elif method == "trivialaugment":
        method_meta.update({
            "year": "2021",
            "reference": "Müller & Hutter 2021 (TrivialAugmentWide, ICCV)",
        })
    elif method == "3augment":
        method_meta.update({
            "p_each": 0.1,
            "components": "Grayscale+Solarize+GaussianBlur",
            "year": "2022",
            "reference": "Touvron+2022 (DeiT-III, ECCV); standard ViT recipe 2023-2025",
        })

    # Save results
    result = {
        "model": model_name,
        "dataset": "screwset",
        "augmentation": method,
        "augmentation_meta": method_meta,
        "phase": phase,
        "pretrained": pretrained,
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "mean_corrupt_acc": mean_corrupt_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "training_time_sec": round(train_time, 1),
        "normalization_mean": list(norm_mean) if isinstance(norm_mean, tuple) else norm_mean,
        "normalization_std": list(norm_std) if isinstance(norm_std, tuple) else norm_std,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": SEED,
    }
    if is_vit:
        result.update({
            "optimizer": "AdamW",
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "scheduler": "CosineWarmup",
            "amp": USE_AMP,
        })
    else:
        result["optimizer"] = "Adam"

    out_path = RESULTS_DIR / f"{model_name}_screwset_{method}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[DONE] Results saved to {out_path}")
    print(f"  Clean Acc:  {test_acc:.4f}")
    print(f"  Mean SS-C:  {mean_corrupt_acc:.4f}")
    print(f"  Train Time: {train_time/60:.1f} min")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Test-Time Augmentation (TTA) — inference only, 2024-2025 standard
# ═══════════════════════════════════════════════════════════════════════════════

def _tta_predict(model, image_tensor, device, is_vit):
    """Produce 10-view TTA prediction for a single image tensor.

    Views: 5-crop (TL, TR, BL, BR, center) × (original + horizontal-flip)
    Returns averaged softmax probabilities.
    """
    # image_tensor shape: (C, H, W)
    C, H, W = image_tensor.shape

    if is_vit:
        crop_size = 224
    else:
        crop_size_h, crop_size_w = min(H, 224), min(W, 300)
        crop_size = (crop_size_h, crop_size_w)

    # Use torchvision FiveCrop + flips
    if isinstance(crop_size, int):
        five_crop = transforms.FiveCrop(crop_size)
    else:
        five_crop = transforms.FiveCrop(crop_size)

    # FiveCrop returns 5 crops as PIL or tensor
    crops = five_crop(image_tensor)  # tuple of 5 tensors

    all_views = []
    for crop in crops:
        all_views.append(crop.unsqueeze(0))
        # Horizontal flip
        all_views.append(torch.flip(crop, dims=[-1]).unsqueeze(0))

    batch = torch.cat(all_views, dim=0).to(device)  # (10, C, crop_h, crop_w)
    if is_vit and USE_CHANNELS_LAST:
        try:
            batch = batch.to(memory_format=torch.channels_last)
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=USE_AMP if is_vit else False):
            logits = model(batch)  # (10, num_classes)
    probs = F.softmax(logits, dim=-1)
    return probs.mean(dim=0)  # (num_classes,)


def evaluate_tta(model, loader, device, is_vit, desc="TTA"):
    """Evaluate with TTA: 10-view ensemble per image."""
    model.eval()
    correct, total = 0, 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        for i in range(inputs.size(0)):
            avg_prob = _tta_predict(model, inputs[i], device, is_vit)
            pred = avg_prob.argmax().item()
            if pred == targets[i].item():
                correct += 1
            total += 1
    return correct / total if total else 0.0


def run_tta(model_name, args, device):
    """Run TTA on existing Phase 1/2 baseline model checkpoints."""
    is_vit = model_name in VIT_MODELS
    phase = "Phase2-ViT" if is_vit else "Phase1-CNN"

    print(f"\n{'='*70}")
    print(f"  TTA ScrewSet — {model_name}  [{phase}]")
    print(f"  10-view ensemble (5-crop × flip)")
    print(f"{'='*70}")

    # Find baseline checkpoint
    if is_vit:
        ckpt_dir = PHASE2_MODELS
        ckpt_pattern = f"{model_name}_screwset_best.pth"
    else:
        ckpt_dir = PHASE1_MODELS
        ckpt_pattern = f"{model_name}_screwset_best.pth"

    ckpt_path = ckpt_dir / ckpt_pattern
    if not ckpt_path.exists():
        # Try alternate naming
        possible = list(ckpt_dir.glob(f"*{model_name}*screwset*best*.pth"))
        if possible:
            ckpt_path = possible[0]
        else:
            print(f"[ERROR] Cannot find checkpoint for {model_name} in {ckpt_dir}")
            print(f"  Looked for: {ckpt_pattern}")
            return None

    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    # Determine num_classes from train set
    _, eval_tf, norm_mean, norm_std = build_transforms_cutmix_mixup(model_name)
    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=eval_tf,
                            is_valid_file=is_valid_image)
    NUM_CLASSES = len(train_ds.classes)

    # Create model and load weights
    pretrained_flag = False  # loading from checkpoint
    model = create_model(model_name, NUM_CLASSES, pretrained=pretrained_flag)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = prepare_model(model, device, is_vit=is_vit)
    model.eval()
    model_tag = f"{model_name}_screwset_tta"

    # Test and corrupt datasets with eval_tf (NO TTA in transforms - TTA is manual)
    test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_tf,
                           is_valid_file=is_valid_image)
    # Batch size 1 for TTA (each image goes through 10 views)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Evaluate clean with TTA
    start_time = time.time()
    test_acc_tta = evaluate_tta(model, test_loader, device, is_vit,
                                desc=f"[{model_tag}] Clean Test TTA")
    print(f"[{model_tag}] Clean Test (TTA): {test_acc_tta:.4f}")

    # Also get plain accuracy for comparison
    criterion = nn.CrossEntropyLoss()
    if is_vit:
        _, test_acc_plain = evaluate(model, test_loader, criterion, device,
                                      desc=f"[{model_tag}] Clean Test (plain)",
                                      use_amp=USE_AMP,
                                      use_channels_last=USE_CHANNELS_LAST)
    else:
        _, test_acc_plain = evaluate(model, test_loader, criterion, device,
                                      desc=f"[{model_tag}] Clean Test (plain)")

    # Evaluate ScrewSet-C with TTA
    corrupt_results_tta = {}
    corrupt_results_plain = {}
    if CORRUPT_ROOT.exists():
        for corrupt_type in sorted(os.listdir(CORRUPT_ROOT)):
            corrupt_dir = CORRUPT_ROOT / corrupt_type
            if not corrupt_dir.is_dir():
                continue
            try:
                ds = ImageFolder(str(corrupt_dir), transform=eval_tf,
                                  is_valid_file=is_valid_image)
                loader = DataLoader(ds, batch_size=32, shuffle=False,
                                     num_workers=4, pin_memory=True)

                # TTA eval
                acc_tta = evaluate_tta(model, loader, device, is_vit,
                                       desc=f"[{model_tag}] {corrupt_type} TTA")
                corrupt_results_tta[corrupt_type] = acc_tta

                # Plain eval
                if is_vit:
                    _, acc_plain = evaluate(model, loader, criterion, device,
                                            desc=f"[{model_tag}] {corrupt_type} plain",
                                            use_amp=USE_AMP,
                                            use_channels_last=USE_CHANNELS_LAST)
                else:
                    _, acc_plain = evaluate(model, loader, criterion, device,
                                            desc=f"[{model_tag}] {corrupt_type} plain")
                corrupt_results_plain[corrupt_type] = acc_plain

                print(f"  SS-C {corrupt_type}: plain={acc_plain:.4f} → TTA={acc_tta:.4f} "
                      f"(Δ={acc_tta - acc_plain:+.4f})")
            except Exception as e:
                print(f"  [ERROR] {corrupt_type}: {e}")

    eval_time = time.time() - start_time

    valid_tta = [v for v in corrupt_results_tta.values() if v is not None]
    valid_plain = [v for v in corrupt_results_plain.values() if v is not None]
    mean_corrupt_tta = sum(valid_tta) / len(valid_tta) if valid_tta else 0.0
    mean_corrupt_plain = sum(valid_plain) / len(valid_plain) if valid_plain else 0.0

    result = {
        "model": model_name,
        "dataset": "screwset",
        "augmentation": "tta",
        "augmentation_meta": {
            "method": "tta",
            "num_views": 10,
            "strategy": "5-crop × (original + horizontal-flip)",
            "year": "2024-2025",
            "reference": "Standard TTA; widely adopted in robustness evaluation 2024-2025",
            "training_required": False,
        },
        "phase": phase,
        "pretrained": is_vit,
        "num_classes": NUM_CLASSES,
        "test_acc": test_acc_tta,
        "test_acc_plain": test_acc_plain,
        "mean_corrupt_acc": mean_corrupt_tta,
        "mean_corrupt_acc_plain": mean_corrupt_plain,
        "corrupt_results": corrupt_results_tta,
        "corrupt_results_plain": corrupt_results_plain,
        "checkpoint_used": str(ckpt_path),
        "evaluation_time_sec": round(eval_time, 1),
        "normalization_mean": list(norm_mean) if isinstance(norm_mean, tuple) else norm_mean,
        "normalization_std": list(norm_std) if isinstance(norm_std, tuple) else norm_std,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_screwset_tta.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[DONE] TTA results saved to {out_path}")
    print(f"  Clean: plain={test_acc_plain:.4f} → TTA={test_acc_tta:.4f}")
    print(f"  SS-C:  plain={mean_corrupt_plain:.4f} → TTA={mean_corrupt_tta:.4f}")
    print(f"  Eval Time: {eval_time/60:.1f} min")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Augmentation Baselines — ScrewSet (2018-2025)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", type=str, required=True,
                        choices=ALL_METHODS + ["all"],
                        help="Augmentation method")
    parser.add_argument("--model", type=str, required=True,
                        choices=ALL_MODELS + ["all"],
                        help="Model to train/evaluate")
    # Training hyperparams
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    # ViT-specific
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    args = parser.parse_args()

    # Setup
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

    method_list = ALL_METHODS if args.method == "all" else [args.method]
    model_list = ALL_MODELS if args.model == "all" else [args.model]

    total_runs = len(method_list) * len(model_list)
    run_idx = 0

    for method in method_list:
        for m_name in model_list:
            run_idx += 1
            is_vit = m_name in VIT_MODELS

            # Auto-set hyperparams
            run_args = argparse.Namespace(**vars(args))
            if run_args.learning_rate is None:
                run_args.learning_rate = 5e-4 if is_vit else 1e-3
            if is_vit and args.num_epochs == 20:
                run_args.num_epochs = 30
            if is_vit and args.patience == 5:
                run_args.patience = 7

            print(f"\n{'#'*70}")
            print(f"  RUN {run_idx}/{total_runs}: {method.upper()} × {m_name}")
            print(f"{'#'*70}")

            set_seed()
            try:
                if method == "tta":
                    run_tta(m_name, run_args, device)
                else:
                    run_training_method(method, m_name, run_args, device)
            except Exception as e:
                print(f"[ERROR] {method} × {m_name} failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  ALL AUGMENTATION RUNS COMPLETE ({total_runs} runs)")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
