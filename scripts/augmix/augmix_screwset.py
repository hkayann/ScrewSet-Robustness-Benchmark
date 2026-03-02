#!/usr/bin/env python3
"""
AugMix Robustness Improvement Baseline — ScrewSet
===================================================
Trains 4 representative models on ScrewSet WITH AugMix augmentation,
then evaluates on ScrewSet-C corruptions. Results are compared against
the no-augmentation Phase 1/2 baselines.

Models:
    resnet18            — Worst CNN on SS-C (9.23%)
    efficientnetv2_rw_s — Best CNN on SS-C (11.28%) + best Lens/IN-C
    vit_tiny_patch16_224 — Small ViT (21.17% SS-C)
    convnext_tiny        — Best ViT on SS-C (45.39%)

Method:
    torchvision.transforms.AugMix applied during training.
    Same hyperparameters as original phase baselines per model type.

Usage:
    python3 augmix_screwset.py --model resnet18
    python3 augmix_screwset.py --model all
    python3 augmix_screwset.py --model all --severity 5 --width 4
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
RESULTS_DIR = REPO_ROOT / "results" / "augmix"
MODELS_SAVE_DIR = RESULTS_DIR / "models"
SPLIT_DIR = DATA_DIR / "screwset_split"
CORRUPT_ROOT = DATA_DIR / "screwset_c"

# CNN models use fixed normalization; ViTs use model-specific
SCREWSET_NORM = {"mean": [0.7750, 0.7343, 0.6862], "std": [0.0802, 0.0838, 0.0871]}

CNN_MODELS = ["resnet18", "efficientnetv2_rw_s"]
VIT_MODELS = ["vit_tiny_patch16_224", "convnext_tiny"]
ALL_MODELS = CNN_MODELS + VIT_MODELS

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

# GPU settings for ViTs (from Phase 2)
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
USE_AMP = True
USE_CHANNELS_LAST = True
PIN_MEMORY = True


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(name, num_classes, pretrained=False):
    """Create CNN (Phase 1 style) or ViT (Phase 2 style) model."""
    # ViTs — all timm
    if name in VIT_TIMM_PRETRAINED_IDS:
        model_id = VIT_TIMM_PRETRAINED_IDS[name] if pretrained else name
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        return model

    # CNN — timm models
    if name in CNN_TIMM_PRETRAINED_IDS:
        model_id = CNN_TIMM_PRETRAINED_IDS[name] if pretrained else name
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        return model

    # CNN — torchvision models
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
# AugMix Transform Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_transforms(model_name, augmix_severity=3, augmix_width=3,
                     augmix_depth=-1, augmix_alpha=1.0):
    """Build train (with AugMix) and eval transforms.

    AugMix is inserted BEFORE ToTensor/Normalize so it operates on PIL images.

    Args:
        model_name: Model name to determine preprocessing.
        augmix_severity: AugMix severity (1-10, default 3).
        augmix_width: Number of augmentation chains (default 3).
        augmix_depth: Depth of each chain (-1 = random 1-3).
        augmix_alpha: Dirichlet mixing parameter.
    """
    is_vit = model_name in VIT_MODELS

    if is_vit:
        pcfg = VIT_PREPROCESS[model_name]
        img_size = pcfg["input_size"]
        resize_short = int(math.floor(img_size / pcfg["crop_pct"]))
        norm_mean, norm_std = pcfg["mean"], pcfg["std"]
    else:
        # CNN: Phase 1 ScrewSet settings
        resize_short = 240  # RESIZE_DIM = (240, 320) but use 240 for short edge
        img_size = 240      # No center crop for CNNs (they used Resize(240,320))
        norm_mean = SCREWSET_NORM["mean"]
        norm_std = SCREWSET_NORM["std"]

    # AugMix operates on PIL images — must come before ToTensor
    augmix_transform = transforms.AugMix(
        severity=augmix_severity,
        mixture_width=augmix_width,
        chain_depth=augmix_depth,
        alpha=augmix_alpha,
    )

    if is_vit:
        train_transform = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            augmix_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomHorizontalFlip(),
            augmix_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    return train_transform, eval_transform, norm_mean, norm_std


# ═══════════════════════════════════════════════════════════════════════════════
# Training: Phase 1 style (CNN) and Phase 2 style (ViT)
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
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def train_cnn(model, train_loader, val_loader, device, args, model_tag):
    """Phase 1 training: Adam + early stopping, no AMP."""
    criterion = nn.CrossEntropyLoss()
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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total
        train_loss /= total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
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


def train_vit(model, train_loader, val_loader, device, args, model_tag):
    """Phase 2 training: AdamW + cosine warmup + AMP + early stopping."""
    criterion = nn.CrossEntropyLoss()
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

        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
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
# Run Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_augmix_screwset(model_name, args, device):
    """Train model on ScrewSet with AugMix, then evaluate clean + corrupt."""
    is_vit = model_name in VIT_MODELS
    phase = "Phase2-ViT" if is_vit else "Phase1-CNN"

    print(f"\n{'='*70}")
    print(f"  AugMix ScrewSet — {model_name}  [{phase}]")
    print(f"  AugMix params: severity={args.augmix_severity}, "
          f"width={args.augmix_width}, depth={args.augmix_depth}")
    print(f"{'='*70}")

    # Build transforms
    train_transform, eval_transform, norm_mean, norm_std = build_transforms(
        model_name,
        augmix_severity=args.augmix_severity,
        augmix_width=args.augmix_width,
        augmix_depth=args.augmix_depth,
    )

    # Load datasets
    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=train_transform,
                            is_valid_file=is_valid_image)
    val_ds = ImageFolder(str(SPLIT_DIR / "validation"), transform=eval_transform,
                          is_valid_file=is_valid_image)
    test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_transform,
                           is_valid_file=is_valid_image)

    NUM_CLASSES = len(train_ds.classes)
    print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    # DataLoaders — Phase-specific settings
    if is_vit:
        gen = make_generator()
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
        gen = make_generator()
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

    # Create model — CNNs from scratch, ViTs pretrained+finetune
    set_seed()
    pretrained = is_vit  # Phase 1 CNNs: scratch, Phase 2 ViTs: pretrained
    model = create_model(model_name, NUM_CLASSES, pretrained=pretrained)
    model = prepare_model(model, device, is_vit=is_vit)
    model_tag = f"{model_name}_screwset_augmix"

    # Train
    start_time = time.time()
    if is_vit:
        best_val_acc, ckpt_path = train_vit(model, train_loader, val_loader,
                                             device, args, model_tag)
    else:
        best_val_acc, ckpt_path = train_cnn(model, train_loader, val_loader,
                                             device, args, model_tag)
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
    corrupt_results = {}
    if CORRUPT_ROOT.exists():
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
                print(f"  ScrewSet-C {corrupt_type}: {acc:.4f}")
            except Exception as e:
                print(f"  [ERROR] {corrupt_type}: {e}")
                corrupt_results[corrupt_type] = None
    else:
        print("[WARN] ScrewSet-C directory not found, skipping corruption eval")

    # Compute mean corrupt accuracy
    valid_accs = [v for v in corrupt_results.values() if v is not None]
    mean_corrupt_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

    # Save results
    result = {
        "model": model_name,
        "dataset": "screwset",
        "augmentation": "AugMix",
        "augmix_severity": args.augmix_severity,
        "augmix_width": args.augmix_width,
        "augmix_depth": args.augmix_depth,
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

    # Add ViT-specific hyperparams
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

    out_path = RESULTS_DIR / f"{model_name}_screwset_augmix.json"
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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AugMix Robustness Baseline — ScrewSet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=ALL_MODELS + ["all"],
                        help="Model to train")
    # Training hyperparams (defaults match original phase baselines)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=20,
                        help="Max epochs (CNNs default=20, ViTs default=30)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="LR (auto: CNN=1e-3, ViT=5e-4)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    # ViT-specific
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    # AugMix params
    parser.add_argument("--augmix-severity", type=int, default=3,
                        help="AugMix severity (1-10)")
    parser.add_argument("--augmix-width", type=int, default=3,
                        help="Number of augmentation chains")
    parser.add_argument("--augmix-depth", type=int, default=-1,
                        help="Chain depth (-1 = random 1-3)")

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

    model_list = ALL_MODELS if args.model == "all" else [args.model]
    total_runs = len(model_list)

    for run_idx, m_name in enumerate(model_list, 1):
        # Auto-set hyperparams by model type
        is_vit = m_name in VIT_MODELS
        run_args = argparse.Namespace(**vars(args))

        if run_args.learning_rate is None:
            run_args.learning_rate = 5e-4 if is_vit else 1e-3

        if is_vit and args.num_epochs == 20:
            run_args.num_epochs = 30  # ViTs need more epochs
        if is_vit and args.patience == 5:
            run_args.patience = 7    # ViTs: more patience

        print(f"\n{'#'*70}")
        print(f"  RUN {run_idx}/{total_runs}: {m_name}")
        print(f"{'#'*70}")

        set_seed()
        try:
            run_augmix_screwset(m_name, run_args, device)
        except Exception as e:
            print(f"[ERROR] {m_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  ALL AUGMIX RUNS COMPLETE ({total_runs} models)")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
