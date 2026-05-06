#!/usr/bin/env python3
"""
Augmentation Models → ScrewSet-S Evaluation Pipeline
=====================================================
For each of 4 models × 6 augmentation strategies:
  1. Train with augmentation on ScrewSet clean split (if checkpoint missing)
  2. Evaluate on ScrewSet-S (19 corruptions × 5 severities)
  3. Save results to results/screwset_s/augmentation/

Models: resnet18, efficientnetv2_rw_s, vit_tiny_patch16_224, convnext_tiny
Methods: augmix, cutmix_mixup, randaugment, trivialaugment, 3augment, tta

Usage:
    python train_and_eval_screwset_s.py                            # all 24
    python train_and_eval_screwset_s.py --model resnet18           # 6 methods
    python train_and_eval_screwset_s.py --method augmix            # 4 models
    python train_and_eval_screwset_s.py --model resnet18 --method augmix  # 1 run
"""

import argparse
import gc
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
    IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA,
)
from src.utils import patch_ipv4, set_seed, make_generator, seed_worker
from src.datasets import is_valid_image, NumpyDataset
from src.evaluation import evaluate

patch_ipv4()


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "screwset_s" / "augmentation"
CKPT_DIR = REPO_ROOT / "results" / "augmentation" / "models"
CKPT_DIR_AUGMIX = REPO_ROOT / "results" / "augmix" / "models"
BASELINE_CKPT_DIR = REPO_ROOT / "results" / "screwset_s" / "models"

SPLIT_DIR = DATA_DIR / "screwset_split"
CORRUPT_ROOT = DATA_DIR / "screwset_c"
SS_DIR = DATA_DIR / "screwset_s"

ALL_CORRUPTIONS = IMAGENET_C_CORRUPTIONS_15 + IMAGENET_C_CORRUPTIONS_EXTRA

SCREWSET_NORM = {"mean": [0.7750, 0.7343, 0.6862], "std": [0.0802, 0.0838, 0.0871]}

CNN_MODELS = ["resnet18", "efficientnetv2_rw_s"]
VIT_MODELS = ["vit_tiny_patch16_224", "convnext_tiny"]
ALL_MODELS = CNN_MODELS + VIT_MODELS

TRAINING_METHODS = ["augmix", "cutmix_mixup", "randaugment", "trivialaugment", "3augment"]
ALL_METHODS = TRAINING_METHODS + ["tta"]

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

# GPU settings — optimised for RTX 5090 (32 GB VRAM) + 28-core CPU + 31 GB RAM
# AMP enabled for ALL models (CNNs + ViTs) to use Tensor Cores + halve VRAM
# Note: system RAM is limited (31 GB). Eval workers use 'spawn' context to avoid
# inheriting CUDA's virtual address space (which causes 19 GB RSS per worker via fork).
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_SS_S = 0   # must be 0: CUDA virtual address space + fork = OOM on 31 GB system
USE_AMP = True
USE_CHANNELS_LAST = True
PIN_MEMORY = True
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2

# Enable cuDNN autotuner — picks fastest conv algorithm for each input shape
torch.backends.cudnn.benchmark = True

# Per-model batch sizes (re-profiled with AMP on 32 GB)
# resnet18       AMP: train bs=1024 → ~18 GB  |  eval bs=2048 → ~10 GB
# efficientnetv2 AMP: train bs=256  → ~28 GB  |  eval bs=1024 → ~12 GB
# vit_tiny       AMP: train bs=1024 → ~26 GB  |  eval bs=2048 → ~5 GB
# convnext_tiny  AMP: train bs=512  → ~30 GB  |  eval bs=1024 → ~14 GB
TRAIN_BATCH_SIZE = {
    "resnet18": 1024,
    "efficientnetv2_rw_s": 256,
    "vit_tiny_patch16_224": 1024,
    "convnext_tiny": 512,
}
EVAL_BATCH_SIZE = {
    "resnet18": 512,
    "efficientnetv2_rw_s": 256,
    "vit_tiny_patch16_224": 512,
    "convnext_tiny": 256,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Soft Cross-Entropy for CutMix/MixUp
# ═══════════════════════════════════════════════════════════════════════════════

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory
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
    model = model.to(device)
    if is_vit and USE_CHANNELS_LAST:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3Augment Transform (DeiT-III recipe)
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeAugment:
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
# Transform Builders
# ═══════════════════════════════════════════════════════════════════════════════

def _get_norm_and_sizes(model_name):
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


def build_eval_transform(model_name):
    """Build evaluation-only transform for a model."""
    is_vit, img_size, resize_short, norm_mean, norm_std = _get_norm_and_sizes(model_name)
    if is_vit:
        return transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]), norm_mean, norm_std
    else:
        return transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]), norm_mean, norm_std


def build_transforms_for_method(method, model_name):
    """Build (train_tf, eval_tf, norm_mean, norm_std) for a method."""
    is_vit, img_size, resize_short, norm_mean, norm_std = _get_norm_and_sizes(model_name)

    # Augmentation-specific transform
    if method == "augmix":
        aug = transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)
    elif method == "randaugment":
        aug = transforms.RandAugment(num_ops=2, magnitude=9)
    elif method == "trivialaugment":
        aug = transforms.TrivialAugmentWide()
    elif method == "3augment":
        aug = ThreeAugment(p_each=0.1)
    elif method == "cutmix_mixup":
        aug = None  # CutMix/MixUp applied at batch level
    else:
        raise ValueError(f"Unknown method: {method}")

    if is_vit:
        base_train = [
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ]
        base_eval = [
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
        ]
    else:
        base_train = [
            transforms.Resize((240, 320)),
            transforms.RandomHorizontalFlip(),
        ]
        base_eval = [
            transforms.Resize((240, 320)),
        ]

    if aug is not None:
        base_train.append(aug)

    base_train += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]
    base_eval += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]

    return transforms.Compose(base_train), transforms.Compose(base_eval), norm_mean, norm_std


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loops
# ═══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
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


def train_cnn(model, train_loader, val_loader, device, model_tag, ckpt_path,
              num_epochs=20, lr=1e-3, patience=5, mixup_fn=None, criterion_train=None):
    """Phase 1 training: Adam + AMP + early stopping."""
    criterion_eval = nn.CrossEntropyLoss()
    if criterion_train is None:
        criterion_train = criterion_eval
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader,
                                     desc=f"[{model_tag}] Epoch {epoch}/{num_epochs}",
                                     leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                inputs, targets_mixed = mixup_fn(inputs, targets)
            else:
                targets_mixed = targets
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion_train(outputs, targets_mixed)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total
        train_loss /= total
        val_loss, val_acc = evaluate(model, val_loader, criterion_eval, device,
                                      desc=f"[{model_tag}] Val {epoch}",
                                      use_amp=USE_AMP)
        elapsed = time.time() - epoch_start
        print(f"[{model_tag}] Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_tag}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"[{model_tag}] Best val acc: {best_val_acc:.4f}")
    return best_val_acc


def train_vit(model, train_loader, val_loader, device, model_tag, ckpt_path,
              num_epochs=30, lr=5e-4, weight_decay=0.05,
              warmup_epochs=5, min_lr=1e-6, patience=7,
              mixup_fn=None, criterion_train=None):
    """Phase 2 training: AdamW + cosine warmup + AMP + early stopping."""
    criterion_eval = nn.CrossEntropyLoss()
    if criterion_train is None:
        criterion_train = criterion_eval
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs, num_epochs, min_lr)
    scaler = GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in tqdm(
            train_loader,
            desc=f"[{model_tag}] Epoch {epoch}/{num_epochs} "
                 f"(lr={optimizer.param_groups[0]['lr']:.2e})",
            leave=False,
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
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
            if no_improve >= patience:
                print(f"[{model_tag}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"[{model_tag}] Best val acc: {best_val_acc:.4f}")
    return best_val_acc


# ═══════════════════════════════════════════════════════════════════════════════
# ScrewSet-S Evaluation (single-pass per corruption, mmap, multi-worker)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_screwset_s(model, eval_transform, device, batch_size,
                        model_tag, use_amp=False, model_name=None):
    """Evaluate on ScrewSet-S (19 corruptions × 5 severities).

    Evaluates all 102,400 images per corruption in ONE DataLoader pass,
    then splits predictions by severity. Avoids repeated worker init overhead.

    Returns dict: {corruption: {severity_1: acc, ..., severity_5: acc, mean: acc}}
    """
    labels = np.load(str(SS_DIR / "labels.npy"))
    n_total = len(labels)
    n_per_sev = n_total // 5  # 20480

    model.eval()
    results = {}

    for cname in ALL_CORRUPTIONS:
        fpath = SS_DIR / f"{cname}.npy"
        if not fpath.exists():
            print(f"  [WARN] {cname}.npy not found, skipping")
            continue

        print(f"  Evaluating {cname}...")
        images_mmap = np.load(str(fpath), mmap_mode="r")

        # Single DataLoader for ALL severities (102,400 images)
        ds = NumpyDataset(images_mmap, labels, eval_transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS_SS_S,
                            pin_memory=False,
                            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS_SS_S > 0 else None,
                            persistent_workers=False)

        # Collect all predictions in one pass
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(loader,
                                         desc=f"[{model_tag}] {cname}",
                                         leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if USE_CHANNELS_LAST:
                    try:
                        inputs = inputs.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                with torch.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                _, preds = outputs.max(1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        del ds, loader

        # Split by severity
        sev_accs = {}
        for sev in range(5):
            start = sev * n_per_sev
            end = start + n_per_sev
            correct = all_preds[start:end].eq(all_targets[start:end]).sum().item()
            acc = correct / n_per_sev
            sev_accs[f"severity_{sev+1}"] = round(acc, 6)

        sev_mean = sum(sev_accs.values()) / len(sev_accs)
        sev_accs["mean"] = round(sev_mean, 6)
        results[cname] = sev_accs
        print(f"    {cname} mean: {sev_mean:.4f}")
        del images_mmap, all_preds, all_targets
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _eval_supervised(model, loader, criterion, device, desc="Eval", use_amp=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            with torch.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total if total else 0.0, correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TTA Evaluation on ScrewSet-S
# ═══════════════════════════════════════════════════════════════════════════════

def _tta_predict(model, image_tensor, device, is_vit):
    """10-view TTA: 5-crop × (original + horizontal-flip)."""
    C, H, W = image_tensor.shape
    crop_size = 224 if is_vit else (min(H, 224), min(W, 300))

    five_crop = transforms.FiveCrop(crop_size)
    crops = five_crop(image_tensor)

    all_views = []
    for crop in crops:
        all_views.append(crop.unsqueeze(0))
        all_views.append(torch.flip(crop, dims=[-1]).unsqueeze(0))

    batch = torch.cat(all_views, dim=0).to(device)
    if is_vit and USE_CHANNELS_LAST:
        try:
            batch = batch.to(memory_format=torch.channels_last)
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=USE_AMP):
            logits = model(batch)
    probs = F.softmax(logits, dim=-1)
    return probs.mean(dim=0)


def evaluate_screwset_s_tta(model, eval_transform, device, batch_size,
                             model_tag, is_vit):
    """TTA evaluation on ScrewSet-S — single pass per corruption."""
    labels = np.load(str(SS_DIR / "labels.npy"))
    n_total = len(labels)
    n_per_sev = n_total // 5

    model.eval()
    results = {}

    for cname in ALL_CORRUPTIONS:
        fpath = SS_DIR / f"{cname}.npy"
        if not fpath.exists():
            print(f"  [WARN] {cname}.npy not found, skipping")
            continue

        print(f"  TTA evaluating {cname}...")
        images_mmap = np.load(str(fpath), mmap_mode="r")

        # Single DataLoader for ALL severities
        ds = NumpyDataset(images_mmap, labels, eval_transform)
        loader = DataLoader(ds, batch_size=128, shuffle=False,
                            num_workers=NUM_WORKERS_SS_S,
                            pin_memory=False,
                            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS_SS_S > 0 else None,
                            persistent_workers=False)

        all_preds = []
        all_targets = []
        for inputs, targets in tqdm(loader,
                                     desc=f"[{model_tag}] {cname} TTA",
                                     leave=False):
            for i in range(inputs.size(0)):
                avg_prob = _tta_predict(model, inputs[i], device, is_vit)
                pred = avg_prob.argmax().item()
                all_preds.append(pred)
                all_targets.append(targets[i].item())

        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)
        del ds, loader

        sev_accs = {}
        for sev in range(5):
            start = sev * n_per_sev
            end = start + n_per_sev
            correct = all_preds[start:end].eq(all_targets[start:end]).sum().item()
            acc = correct / n_per_sev
            sev_accs[f"severity_{sev+1}"] = round(acc, 6)

        sev_mean = sum(sev_accs.values()) / len(sev_accs)
        sev_accs["mean"] = round(sev_mean, 6)
        results[cname] = sev_accs
        print(f"    {cname} TTA mean: {sev_mean:.4f}")
        del images_mmap, all_preds, all_targets
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main Runner: Training-Time Methods
# ═══════════════════════════════════════════════════════════════════════════════

def find_or_train_checkpoint(method, model_name, device, num_classes):
    """Find existing augmentation checkpoint or train from scratch.

    Returns: (model_loaded_on_device, ckpt_path_str, best_val_acc, train_time, was_trained)
    """
    is_vit = model_name in VIT_MODELS
    model_tag = f"{model_name}_screwset_{method}"

    # Check if augmentation checkpoint exists
    if method == "augmix":
        ckpt_path = CKPT_DIR_AUGMIX / f"{model_tag}_best.pth"
    else:
        ckpt_path = CKPT_DIR / f"{model_tag}_best.pth"

    if ckpt_path.exists():
        print(f"  [INFO] Found existing checkpoint: {ckpt_path}")
        pretrained_flag = False
        model = create_model(model_name, num_classes, pretrained=pretrained_flag)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = prepare_model(model, device, is_vit=is_vit)
        model.eval()
        return model, str(ckpt_path), -1.0, 0.0, False

    # No checkpoint found — need to train
    print(f"  [INFO] No checkpoint found at {ckpt_path}, training from scratch...")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Build transforms
    train_tf, eval_tf, _, _ = build_transforms_for_method(method, model_name)

    # Load datasets
    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=train_tf,
                            is_valid_file=is_valid_image)
    val_ds = ImageFolder(str(SPLIT_DIR / "validation"), transform=eval_tf,
                          is_valid_file=is_valid_image)
    print(f"  [INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Classes: {num_classes}")

    gen = make_generator()
    bs_train = TRAIN_BATCH_SIZE.get(model_name, 128)
    bs_val = bs_train  # same batch size for validation
    print(f"  [INFO] Training batch_size={bs_train} (model={model_name})")
    # Unified DataLoader config for both CNNs and ViTs
    loader_kwargs = dict(
        num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs_train, shuffle=True,
        worker_init_fn=seed_worker, generator=gen,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False,
        **loader_kwargs,
    )

    # Create model
    set_seed()
    pretrained = is_vit
    model = create_model(model_name, num_classes, pretrained=pretrained)
    model = prepare_model(model, device, is_vit=is_vit)

    # CutMix/MixUp batch-level mixing
    mixup_fn = None
    criterion_train = None
    if method == "cutmix_mixup":
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
            switch_prob=0.5, mode="batch", label_smoothing=0.1,
            num_classes=num_classes,
        )
        criterion_train = SoftTargetCrossEntropy()

    # Train
    start_time = time.time()
    if is_vit:
        best_val_acc = train_vit(
            model, train_loader, val_loader, device, model_tag, ckpt_path,
            num_epochs=30, lr=5e-4, weight_decay=0.05,
            warmup_epochs=5, min_lr=1e-6, patience=7,
            mixup_fn=mixup_fn, criterion_train=criterion_train)
    else:
        best_val_acc = train_cnn(
            model, train_loader, val_loader, device, model_tag, ckpt_path,
            num_epochs=20, lr=1e-3, patience=5,
            mixup_fn=mixup_fn, criterion_train=criterion_train)
    train_time = time.time() - start_time

    # Cleanup train loaders
    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(2)  # Brief pause to let OS reclaim shared memory

    return model, str(ckpt_path), best_val_acc, train_time, True


def run_training_method_ss_s(method, model_name, device):
    """Train (if needed) + evaluate augmentation model on ScrewSet-S."""
    is_vit = model_name in VIT_MODELS
    phase = "Phase2-ViT" if is_vit else "Phase1-CNN"
    model_tag = f"{model_name}_screwset_{method}"

    # Check if SS-S result already exists
    out_path = RESULTS_DIR / f"{model_name}_screwset_{method}_ss_s.json"
    if out_path.exists():
        print(f"\n[SKIP] {out_path.name} already exists")
        return None

    print(f"\n{'='*70}")
    print(f"  {method.upper()} × {model_name}  [{phase}]  → ScrewSet-S")
    print(f"{'='*70}")

    # Get num_classes
    train_ds_tmp = ImageFolder(str(SPLIT_DIR / "train"), is_valid_file=is_valid_image)
    num_classes = len(train_ds_tmp.classes)
    del train_ds_tmp

    # Find or train checkpoint
    model, ckpt_path, best_val_acc, train_time, was_trained = \
        find_or_train_checkpoint(method, model_name, device, num_classes)

    # Build eval transform
    eval_tf, norm_mean, norm_std = build_eval_transform(model_name)

    # Evaluate clean test set
    test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_tf,
                           is_valid_file=is_valid_image)
    bs_eval = EVAL_BATCH_SIZE.get(model_name, 256)
    test_loader = DataLoader(test_ds, batch_size=bs_eval, shuffle=False,
                              num_workers=NUM_WORKERS_SS_S, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS_SS_S > 0 else None)
    criterion = nn.CrossEntropyLoss()
    _, test_acc = _eval_supervised(model, test_loader, criterion, device,
                                    desc=f"[{model_tag}] Clean Test",
                                    use_amp=USE_AMP)
    print(f"  Clean Test Acc: {test_acc:.4f}")
    del test_loader, test_ds
    gc.collect()

    # Evaluate ScrewSet-S
    ss_s_results = evaluate_screwset_s(
        model, eval_tf, device, bs_eval, model_tag,
        use_amp=USE_AMP, model_name=model_name)

    # Compute overall mean
    all_means = [v["mean"] for v in ss_s_results.values()]
    mean_acc_all = sum(all_means) / len(all_means) if all_means else 0.0

    # Build augmentation metadata
    method_meta = {"method": method}
    if method == "augmix":
        method_meta.update({"severity": 3, "width": 3, "depth": -1, "alpha": 1.0})
    elif method == "cutmix_mixup":
        method_meta.update({"mixup_alpha": 0.8, "cutmix_alpha": 1.0,
                            "switch_prob": 0.5, "label_smoothing": 0.1})
    elif method == "randaugment":
        method_meta.update({"num_ops": 2, "magnitude": 9})
    elif method == "3augment":
        method_meta.update({"p_each": 0.1})

    result = {
        "model": model_name,
        "dataset": "screwset_s",
        "augmentation": method,
        "augmentation_meta": method_meta,
        "phase": phase,
        "num_classes": num_classes,
        "test_acc": round(test_acc, 6),
        "mean_acc_all": round(mean_acc_all, 6),
        "corruption_results": ss_s_results,
        "model_path": ckpt_path,
        "was_retrained": was_trained,
        "training_time_sec": round(train_time, 1) if was_trained else None,
        "best_val_acc": round(best_val_acc, 6) if best_val_acc >= 0 else None,
        "normalization_mean": list(norm_mean) if isinstance(norm_mean, tuple) else norm_mean,
        "normalization_std": list(norm_std) if isinstance(norm_std, tuple) else norm_std,
        "seed": SEED,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[DONE] Saved {out_path.name}")
    print(f"  Clean Acc:    {test_acc:.4f}")
    print(f"  Mean SS-S:    {mean_acc_all:.4f}")
    if was_trained:
        print(f"  Train Time:   {train_time/60:.1f} min")

    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def _force_cuda_cleanup():
    """Aggressive CUDA cleanup between runs to prevent OOM cascading."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Log memory state
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  [MEM] Allocated: {alloc:.2f} GB, Reserved: {reserved:.2f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Runner: TTA on ScrewSet-S
# ═══════════════════════════════════════════════════════════════════════════════

def run_tta_ss_s(model_name, device):
    """Run TTA on baseline Phase 1/2 checkpoint, evaluate on ScrewSet-S."""
    is_vit = model_name in VIT_MODELS
    phase = "Phase2-ViT" if is_vit else "Phase1-CNN"
    model_tag = f"{model_name}_screwset_tta"

    out_path = RESULTS_DIR / f"{model_name}_screwset_tta_ss_s.json"
    if out_path.exists():
        print(f"\n[SKIP] {out_path.name} already exists")
        return None

    print(f"\n{'='*70}")
    print(f"  TTA × {model_name}  [{phase}]  → ScrewSet-S")
    print(f"  10-view ensemble (5-crop × flip)")
    print(f"{'='*70}")

    # Find baseline checkpoint (trained during SS-S baseline eval)
    ckpt_path = BASELINE_CKPT_DIR / f"{model_name}_screwset_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] Baseline checkpoint not found: {ckpt_path}")
        return None

    print(f"  [INFO] Loading baseline checkpoint: {ckpt_path}")

    # Get num_classes
    train_ds_tmp = ImageFolder(str(SPLIT_DIR / "train"), is_valid_file=is_valid_image)
    num_classes = len(train_ds_tmp.classes)
    del train_ds_tmp

    # Load model
    model = create_model(model_name, num_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = prepare_model(model, device, is_vit=is_vit)
    model.eval()

    # Build eval transform
    eval_tf, norm_mean, norm_std = build_eval_transform(model_name)

    # Evaluate clean test with TTA
    test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_tf,
                           is_valid_file=is_valid_image)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=NUM_WORKERS_SS_S, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS_SS_S > 0 else None)

    # Plain clean acc
    criterion = nn.CrossEntropyLoss()
    _, test_acc_plain = _eval_supervised(model, test_loader, criterion, device,
                                          desc=f"[{model_tag}] Clean plain",
                                          use_amp=USE_AMP)

    # TTA clean acc
    correct, total = 0, 0
    for inputs, targets in tqdm(test_loader, desc=f"[{model_tag}] Clean TTA", leave=False):
        for i in range(inputs.size(0)):
            avg_prob = _tta_predict(model, inputs[i], device, is_vit)
            pred = avg_prob.argmax().item()
            if pred == targets[i].item():
                correct += 1
            total += 1
    test_acc_tta = correct / total if total else 0.0
    print(f"  Clean plain={test_acc_plain:.4f} → TTA={test_acc_tta:.4f}")
    del test_loader, test_ds

    # Evaluate ScrewSet-S with TTA
    start_time = time.time()
    ss_s_results = evaluate_screwset_s_tta(
        model, eval_tf, device, 32, model_tag, is_vit)
    eval_time = time.time() - start_time

    # Compute overall mean
    all_means = [v["mean"] for v in ss_s_results.values()]
    mean_acc_all = sum(all_means) / len(all_means) if all_means else 0.0

    result = {
        "model": model_name,
        "dataset": "screwset_s",
        "augmentation": "tta",
        "augmentation_meta": {
            "method": "tta",
            "num_views": 10,
            "strategy": "5-crop × (original + horizontal-flip)",
            "training_required": False,
        },
        "phase": phase,
        "num_classes": num_classes,
        "test_acc": round(test_acc_tta, 6),
        "test_acc_plain": round(test_acc_plain, 6),
        "mean_acc_all": round(mean_acc_all, 6),
        "corruption_results": ss_s_results,
        "checkpoint_used": str(ckpt_path),
        "evaluation_time_sec": round(eval_time, 1),
        "normalization_mean": list(norm_mean) if isinstance(norm_mean, tuple) else norm_mean,
        "normalization_std": list(norm_std) if isinstance(norm_std, tuple) else norm_std,
        "seed": SEED,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[DONE] Saved {out_path.name}")
    print(f"  Clean: plain={test_acc_plain:.4f} → TTA={test_acc_tta:.4f}")
    print(f"  Mean SS-S TTA: {mean_acc_all:.4f}")
    print(f"  Eval Time: {eval_time/60:.1f} min")

    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train augmentation models + evaluate on ScrewSet-S",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", type=str, default="all",
                        choices=ALL_METHODS + ["all"],
                        help="Augmentation method (or 'all')")
    parser.add_argument("--model", type=str, default="all",
                        choices=ALL_MODELS + ["all"],
                        help="Model (or 'all')")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR_AUGMIX.mkdir(parents=True, exist_ok=True)

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
    completed = 0
    failed = 0

    print(f"\n{'#'*70}")
    print(f"  AUGMENTATION → ScrewSet-S PIPELINE")
    print(f"  Models: {model_list}")
    print(f"  Methods: {method_list}")
    print(f"  Total runs: {total_runs}")
    print(f"{'#'*70}")

    for method in method_list:
        for m_name in model_list:
            run_idx += 1
            print(f"\n{'#'*70}")
            print(f"  RUN {run_idx}/{total_runs}: {method.upper()} × {m_name}")
            print(f"{'#'*70}")

            set_seed()
            _force_cuda_cleanup()
            try:
                if method == "tta":
                    result = run_tta_ss_s(m_name, device)
                else:
                    result = run_training_method_ss_s(method, m_name, device)
                if result is not None:
                    completed += 1
                else:
                    # Skipped (already exists) or error
                    completed += 1  # count skips as completed
            except Exception as e:
                failed += 1
                print(f"[ERROR] {method} × {m_name} failed: {e}")
                import traceback
                traceback.print_exc()
                # Aggressive cleanup after failure to prevent OOM cascade
                _force_cuda_cleanup()

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total: {total_runs} | Completed/Skipped: {completed} | Failed: {failed}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
