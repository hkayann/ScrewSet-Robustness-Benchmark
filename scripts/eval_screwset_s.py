#!/usr/bin/env python3
"""
ScrewSet-S Evaluation: All Models, All Phases
==============================================
Trains Phase 1 (CNN) and Phase 2 (ViT) models on ScrewSet (if no checkpoints),
then evaluates ALL models on ScrewSet-S (19 corruptions × 5 severities).

Phase 3 VLMs are zero-shot (no training needed).

Results saved per-model as JSON with per-severity + mean accuracy
for each corruption.

Usage:
    python eval_screwset_s.py --phase 1                  # Phase 1 CNNs
    python eval_screwset_s.py --phase 2                  # Phase 2 ViTs
    python eval_screwset_s.py --phase 3                  # Phase 3 VLMs
    python eval_screwset_s.py --phase all                # All phases
    python eval_screwset_s.py --phase 1 --model resnet18 # Single model
    python eval_screwset_s.py --eval-only --phase 1      # Skip training, load checkpoints
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED,
    CNN_TIMM_PRETRAINED_IDS,
    VIT_TIMM_PRETRAINED_IDS,
    IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA,
)
from src.utils import patch_ipv4, set_seed, make_generator, seed_worker
from src.datasets import is_valid_image, NumpyDataset, PILNumpyDataset
from src.class_names import get_screwset_class_names

patch_ipv4()


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "screwset_s"
MODELS_SAVE_DIR = RESULTS_DIR / "models"
SS_DIR = DATA_DIR / "screwset_s"
SPLIT_DIR = DATA_DIR / "screwset_split"

ALL_CORRUPTIONS = IMAGENET_C_CORRUPTIONS_15 + IMAGENET_C_CORRUPTIONS_EXTRA

# ── Phase 1 CNN models ──────────────────────────────────────────────────────
PHASE1_MODELS = [
    "resnet18", "squeezenet1_1", "mobilenet_v3_large", "mobilenetv4_conv_small",
    "shufflenet_v2_x1_0", "efficientnetv2_rw_s", "ghostnetv2_100", "convnextv2_atto",
]

PHASE1_NORMALIZATION = {"mean": [0.7750, 0.7343, 0.6862], "std": [0.0802, 0.0838, 0.0871]}
PHASE1_RESIZE = (240, 320)

# ── Phase 2 ViT models ──────────────────────────────────────────────────────
PHASE2_MODELS = [
    "vit_tiny_patch16_224", "vit_small_patch16_224",
    "deit_tiny_patch16_224", "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224", "mobilevit_s",
    "efficientformer_l1", "convnext_tiny",
]

PHASE2_PREPROCESS = {
    "vit_tiny_patch16_224":          {"mean": (0.5, 0.5, 0.5),           "std": (0.5, 0.5, 0.5),           "input_size": 224, "crop_pct": 0.9},
    "vit_small_patch16_224":         {"mean": (0.5, 0.5, 0.5),           "std": (0.5, 0.5, 0.5),           "input_size": 224, "crop_pct": 0.9},
    "deit_tiny_patch16_224":         {"mean": (0.485, 0.456, 0.406),     "std": (0.229, 0.224, 0.225),     "input_size": 224, "crop_pct": 0.9},
    "deit_small_patch16_224":        {"mean": (0.485, 0.456, 0.406),     "std": (0.229, 0.224, 0.225),     "input_size": 224, "crop_pct": 0.9},
    "swin_tiny_patch4_window7_224":  {"mean": (0.485, 0.456, 0.406),     "std": (0.229, 0.224, 0.225),     "input_size": 224, "crop_pct": 0.9},
    "mobilevit_s":                   {"mean": (0.0, 0.0, 0.0),           "std": (1.0, 1.0, 1.0),           "input_size": 256, "crop_pct": 0.9},
    "efficientformer_l1":            {"mean": (0.485, 0.456, 0.406),     "std": (0.229, 0.224, 0.225),     "input_size": 224, "crop_pct": 0.95},
    "convnext_tiny":                 {"mean": (0.485, 0.456, 0.406),     "std": (0.229, 0.224, 0.225),     "input_size": 224, "crop_pct": 0.875},
}

# ── Phase 3 VLM models ──────────────────────────────────────────────────────
PHASE3_MODELS_CLIP = [
    "clip_vit_b32", "clip_vit_b16", "clip_vit_l14",
    "openclip_vit_b16", "siglip_vit_b16", "eva02_clip_vit_b16",
]
PHASE3_MODELS_GENERATIVE = ["blip2", "llava", "qwen2_5_vl"]
PHASE3_MODELS = PHASE3_MODELS_CLIP + PHASE3_MODELS_GENERATIVE

PHASE3_MODEL_CFG = {
    "clip_vit_b32":        {"model_name": "ViT-B-32",       "pretrained": "openai",              "family": "CLIP (OpenAI)"},
    "clip_vit_b16":        {"model_name": "ViT-B-16",       "pretrained": "openai",              "family": "CLIP (OpenAI)"},
    "clip_vit_l14":        {"model_name": "ViT-L-14",       "pretrained": "openai",              "family": "CLIP (OpenAI)"},
    "openclip_vit_b16":    {"model_name": "ViT-B-16",       "pretrained": "laion2b_s34b_b88k",   "family": "OpenCLIP (LAION-2B)"},
    "siglip_vit_b16":      {"model_name": "ViT-B-16-SigLIP","pretrained": "webli",               "family": "SigLIP (Google)"},
    "eva02_clip_vit_b16":  {"model_name": "EVA02-B-16",     "pretrained": "merged2b_s8b_b131k",  "family": "EVA-02-CLIP"},
    "blip2":               {"model_id": "Salesforce/blip2-opt-2.7b",     "family": "BLIP-2"},
    "llava":               {"model_id": "llava-hf/llava-1.5-7b-hf",     "family": "LLaVA-1.5"},
    "qwen2_5_vl":          {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "Qwen2.5-VL"},
}

PHASE3_BATCH_SIZES = {
    "clip_vit_b32": 1024, "clip_vit_b16": 512, "clip_vit_l14": 256,
    "openclip_vit_b16": 512, "siglip_vit_b16": 512, "eva02_clip_vit_b16": 512,
    "blip2": 64, "llava": 4, "qwen2_5_vl": 16,
}

SIMPLE_TEMPLATES = ["a photo of a {}."]

# Phase 2 GPU performance settings
USE_AMP = True
USE_CHANNELS_LAST = True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: CNN Model Factory & Training
# ═══════════════════════════════════════════════════════════════════════════════

def create_phase1_model(name, num_classes, pretrained=False):
    """Create a Phase 1 CNN model (same logic as phase1_baselines.py)."""
    is_timm = name in CNN_TIMM_PRETRAINED_IDS

    if is_timm:
        model_id = CNN_TIMM_PRETRAINED_IDS[name] if pretrained else name
        return timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "squeezenet1_1":
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        if num_classes != 1000:
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
            model.num_classes = num_classes
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        if num_classes != 1000:
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown Phase 1 model: {name}")
    return model


def train_phase1(model, train_loader, val_loader, device, model_tag,
                 num_epochs=20, lr=1e-3, patience=5):
    """Train a Phase 1 CNN with Adam + early stopping (same as phase1_baselines.py)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader,
                                     desc=f"[{model_tag}] Epoch {epoch}/{num_epochs}",
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

        # Validate
        val_loss, val_acc = _evaluate_supervised(model, val_loader, criterion, device,
                                                  desc=f"[{model_tag}] Val {epoch}/{num_epochs}")
        print(f"[{model_tag}] Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_tag}] Early stopping at epoch {epoch} "
                      f"(no improve for {patience})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"[{model_tag}] Training done. Best val acc: {best_val_acc:.4f}")
    return best_val_acc, str(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: ViT Model Factory & Training
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
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                    for base_lr in self.base_lrs]


def create_phase2_model(name, num_classes, pretrained=True):
    """Create a Phase 2 ViT model via timm (same as phase2_vit_baselines.py)."""
    model_id = VIT_TIMM_PRETRAINED_IDS[name] if pretrained else name
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {name}: {n_params:,} params")
    return model


def train_phase2(model, train_loader, val_loader, device, model_tag,
                 num_epochs=30, lr=5e-4, weight_decay=0.05,
                 warmup_epochs=5, min_lr=1e-6, patience=7):
    """Train a Phase 2 ViT with AdamW + cosine warmup + AMP (same as phase2)."""
    from torch.cuda.amp import GradScaler, autocast

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs, num_epochs, min_lr)
    scaler = GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, targets in tqdm(
            train_loader,
            desc=f"[{model_tag}] Epoch {epoch}/{num_epochs} (lr={optimizer.param_groups[0]['lr']:.2e})",
            leave=False
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

        val_loss, val_acc = _evaluate_supervised(
            model, val_loader, criterion, device,
            desc=f"[{model_tag}] Val {epoch}/{num_epochs}",
            use_amp=True)

        print(f"[{model_tag}] Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

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
    print(f"[{model_tag}] Training done. Best val acc: {best_val_acc:.4f}")
    return best_val_acc, str(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_supervised(model, loader, criterion, device, desc="Eval",
                         use_amp=False):
    """Standard evaluation: returns (loss, accuracy)."""
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


def make_eval_loader(dataset, batch_size, num_workers=8):
    """Create evaluation DataLoader."""
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = (num_workers >= 4)
    return DataLoader(dataset, **kwargs)


def make_train_loader(dataset, batch_size, num_workers=8):
    """Create training DataLoader."""
    gen = make_generator()
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
        worker_init_fn=seed_worker, generator=gen,
        drop_last=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ScrewSet-S Evaluation Core
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_screwset_s_supervised(model, eval_transform, device, batch_size,
                                    model_tag, use_amp=False):
    """Evaluate a supervised model on ScrewSet-S (19 corruptions × 5 severities).

    Args:
        model: Trained PyTorch model (in eval mode).
        eval_transform: torchvision transforms for preprocessing.
        device: torch.device.
        batch_size: Batch size for DataLoader.
        model_tag: String tag for progress bars.
        use_amp: Enable automatic mixed precision.

    Returns:
        dict: {corruption_name: {severity_1: acc, ..., severity_5: acc, mean: acc}, ...}
    """
    labels = np.load(str(SS_DIR / "labels.npy"))
    n_total = len(labels)
    n_per_sev = n_total // 5  # 20480

    criterion = nn.CrossEntropyLoss()
    model.eval()
    results = {}

    for cname in ALL_CORRUPTIONS:
        fpath = SS_DIR / f"{cname}.npy"
        if not fpath.exists():
            print(f"  [WARN] {cname}.npy not found, skipping")
            continue

        print(f"  Evaluating {cname}...")
        images_mmap = np.load(str(fpath), mmap_mode="r")
        sev_accs = {}

        for sev in range(5):
            start = sev * n_per_sev
            end = start + n_per_sev
            # Copy slice to RAM to avoid mmap page-cache thrashing
            sev_images = np.array(images_mmap[start:end])
            sev_labels = labels[start:end]
            ds = NumpyDataset(sev_images, sev_labels, eval_transform)
            loader = make_eval_loader(ds, batch_size, num_workers=2)
            _, acc = _evaluate_supervised(
                model, loader, criterion, device,
                desc=f"[{model_tag}] {cname} sev{sev+1}",
                use_amp=use_amp)
            sev_accs[f"severity_{sev+1}"] = round(acc, 6)
            print(f"    severity {sev+1}: {acc:.4f}")
            del sev_images, ds, loader  # free RAM before next severity

        sev_mean = sum(sev_accs.values()) / len(sev_accs)
        sev_accs["mean"] = round(sev_mean, 6)
        results[cname] = sev_accs
        print(f"    mean: {sev_mean:.4f}")
        del images_mmap  # release mmap

    return results


def evaluate_screwset_s_clip(model, tokenizer, preprocess, text_features,
                              device, batch_size, model_tag):
    """Evaluate a CLIP-family model on ScrewSet-S (zero-shot)."""
    labels = np.load(str(SS_DIR / "labels.npy"))
    n_total = len(labels)
    n_per_sev = n_total // 5

    results = {}
    for cname in ALL_CORRUPTIONS:
        fpath = SS_DIR / f"{cname}.npy"
        if not fpath.exists():
            print(f"  [WARN] {cname}.npy not found, skipping")
            continue

        print(f"  Evaluating {cname}...")
        images_mmap = np.load(str(fpath), mmap_mode="r")
        sev_accs = {}

        for sev in range(5):
            start = sev * n_per_sev
            end = start + n_per_sev
            # Copy slice to RAM to avoid mmap page-cache thrashing
            sev_images = np.array(images_mmap[start:end])
            sev_labels = labels[start:end]
            ds = NumpyDataset(sev_images, sev_labels, preprocess)
            loader = make_eval_loader(ds, batch_size, num_workers=2)
            acc = _clip_evaluate(model, loader, text_features, device,
                                  desc=f"[{model_tag}] {cname} sev{sev+1}")
            sev_accs[f"severity_{sev+1}"] = round(acc, 6)
            print(f"    severity {sev+1}: {acc:.4f}")
            del sev_images, ds, loader  # free RAM before next severity

        sev_mean = sum(sev_accs.values()) / len(sev_accs)
        sev_accs["mean"] = round(sev_mean, 6)
        results[cname] = sev_accs
        print(f"    mean: {sev_mean:.4f}")
        del images_mmap  # release mmap

    return results


def _clip_evaluate(model, loader, text_features, device, desc="Eval"):
    """CLIP zero-shot eval on a DataLoader. Returns accuracy."""
    correct, total = 0, 0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def _tokenize_for_clip(tokenizer, texts, device):
    """Tokenize texts for open_clip, handling SigLIP's T5 tokenizer."""
    try:
        tokens = tokenizer(texts)
        if isinstance(tokens, dict):
            return tokens["input_ids"].to(device)
        return tokens.to(device)
    except AttributeError:
        hf_tok = getattr(tokenizer, "tokenizer", tokenizer)
        tok_out = hf_tok(
            texts, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )
        return tok_out["input_ids"].to(device)


def build_clip_text_features(model, tokenizer, class_names, templates, device):
    """Build text feature matrix using template ensembling."""
    print(f"[TEXT] Building text features: {len(class_names)} classes × {len(templates)} templates")
    all_features = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for template in templates:
            texts = [template.format(c) for c in class_names]
            tokens = _tokenize_for_clip(tokenizer, texts, device)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
            all_features.append(feats)
    text_features = torch.stack(all_features).mean(dim=0)
    text_features = F.normalize(text_features, dim=-1)
    return text_features


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train + Evaluate ScrewSet-S
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(model_name, device, eval_only=False, batch_size=256,
               num_epochs=20, lr=1e-3, patience=5):
    """Train Phase 1 CNN on ScrewSet, then evaluate on ScrewSet-S."""
    print(f"\n{'='*70}")
    print(f"  Phase 1: {model_name} — ScrewSet-S Evaluation")
    print(f"{'='*70}")

    # Check for existing results
    out_path = RESULTS_DIR / f"{model_name}_screwset_s.json"
    if out_path.exists():
        print(f"[SKIP] Results already exist: {out_path}")
        return json.load(open(out_path))

    eval_transform = transforms.Compose([
        transforms.Resize(PHASE1_RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=PHASE1_NORMALIZATION["mean"],
                              std=PHASE1_NORMALIZATION["std"]),
    ])

    # Count classes
    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=eval_transform,
                            is_valid_file=is_valid_image)
    NUM_CLASSES = len(train_ds.classes)

    # Check for existing checkpoint
    model_tag = f"{model_name}_screwset"
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"
    best_val_acc = None

    if ckpt_path.exists():
        print(f"[INFO] Loading existing checkpoint: {ckpt_path}")
        set_seed()
        model = create_phase1_model(model_name, NUM_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)
    elif eval_only:
        print(f"[ERROR] --eval-only but no checkpoint at {ckpt_path}")
        return None
    else:
        # Train
        train_transform = transforms.Compose([
            transforms.Resize(PHASE1_RESIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=PHASE1_NORMALIZATION["mean"],
                                  std=PHASE1_NORMALIZATION["std"]),
        ])
        train_ds_t = ImageFolder(str(SPLIT_DIR / "train"), transform=train_transform,
                                  is_valid_file=is_valid_image)
        val_ds = ImageFolder(str(SPLIT_DIR / "validation"), transform=eval_transform,
                              is_valid_file=is_valid_image)
        test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_transform,
                               is_valid_file=is_valid_image)

        print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds_t)}, "
              f"Val: {len(val_ds)}, Test: {len(test_ds)}")

        gen = make_generator()
        train_loader = DataLoader(train_ds_t, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True,
                                   worker_init_fn=seed_worker, generator=gen)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

        set_seed()
        model = create_phase1_model(model_name, NUM_CLASSES, pretrained=False)
        model = model.to(device)

        best_val_acc, ckpt_path_str = train_phase1(
            model, train_loader, val_loader, device, model_tag,
            num_epochs=num_epochs, lr=lr, patience=patience)

        # Evaluate clean test
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        _, test_acc = _evaluate_supervised(model, test_loader, criterion, device,
                                            desc=f"[{model_tag}] Clean Test")
        print(f"[{model_tag}] Clean Test Acc: {test_acc:.4f}")

    # ── Evaluate ScrewSet-S ──
    print(f"\n[{model_tag}] Evaluating ScrewSet-S (19 corruptions × 5 severities)...")
    ss_results = evaluate_screwset_s_supervised(
        model, eval_transform, device, batch_size, model_tag, use_amp=False)

    # Compute overall statistics
    all_means = [v["mean"] for v in ss_results.values()]
    overall_mean = sum(all_means) / len(all_means) if all_means else 0.0

    result = {
        "model": model_name,
        "phase": 1,
        "dataset": "screwset_s",
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "model_path": str(ckpt_path),
        "normalization_mean": PHASE1_NORMALIZATION["mean"],
        "normalization_std": PHASE1_NORMALIZATION["std"],
        "resize": list(PHASE1_RESIZE),
        "screwset_s_results": ss_results,
        "screwset_s_overall_mean": round(overall_mean, 6),
        "num_corruptions": len(ss_results),
        "num_severities": 5,
        "seed": SEED,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    print(f"[DONE] ScrewSet-S overall mean: {overall_mean:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Train + Evaluate ScrewSet-S
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(model_name, device, eval_only=False, batch_size=256,
               num_epochs=30, lr=5e-4, weight_decay=0.05,
               warmup_epochs=5, min_lr=1e-6, patience=7):
    """Train Phase 2 ViT on ScrewSet, then evaluate on ScrewSet-S."""
    print(f"\n{'='*70}")
    print(f"  Phase 2: {model_name} — ScrewSet-S Evaluation")
    print(f"{'='*70}")

    # Check for existing results
    out_path = RESULTS_DIR / f"{model_name}_screwset_s.json"
    if out_path.exists():
        print(f"[SKIP] Results already exist: {out_path}")
        return json.load(open(out_path))

    # Model-specific preprocessing
    pcfg = PHASE2_PREPROCESS[model_name]
    IMG_SIZE = pcfg["input_size"]
    RESIZE_SHORT = int(math.floor(IMG_SIZE / pcfg["crop_pct"]))
    NORM_MEAN, NORM_STD = pcfg["mean"], pcfg["std"]

    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    # Count classes
    train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=eval_transform,
                            is_valid_file=is_valid_image)
    NUM_CLASSES = len(train_ds.classes)

    model_tag = f"{model_name}_screwset"
    ckpt_path = MODELS_SAVE_DIR / f"{model_tag}_best.pth"

    if ckpt_path.exists():
        print(f"[INFO] Loading existing checkpoint: {ckpt_path}")
        set_seed()
        model = create_phase2_model(model_name, NUM_CLASSES, pretrained=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)
        if USE_CHANNELS_LAST:
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        best_val_acc = None
    elif eval_only:
        print(f"[ERROR] --eval-only but no checkpoint at {ckpt_path}")
        return None
    else:
        # Train
        train_transform = transforms.Compose([
            transforms.Resize(RESIZE_SHORT),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        train_ds_t = ImageFolder(str(SPLIT_DIR / "train"), transform=train_transform,
                                  is_valid_file=is_valid_image)
        val_ds = ImageFolder(str(SPLIT_DIR / "validation"), transform=eval_transform,
                              is_valid_file=is_valid_image)
        test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=eval_transform,
                               is_valid_file=is_valid_image)

        print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds_t)}, "
              f"Val: {len(val_ds)}, Test: {len(test_ds)}")

        train_loader = make_train_loader(train_ds_t, batch_size)
        val_loader = make_eval_loader(val_ds, batch_size)

        set_seed()
        model = create_phase2_model(model_name, NUM_CLASSES, pretrained=True)
        model = model.to(device)
        if USE_CHANNELS_LAST:
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        best_val_acc, ckpt_path_str = train_phase2(
            model, train_loader, val_loader, device, model_tag,
            num_epochs=num_epochs, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, min_lr=min_lr, patience=patience)

        # Evaluate clean test
        test_loader = make_eval_loader(test_ds, batch_size)
        criterion = nn.CrossEntropyLoss()
        _, test_acc = _evaluate_supervised(model, test_loader, criterion, device,
                                            desc=f"[{model_tag}] Clean Test",
                                            use_amp=True)
        print(f"[{model_tag}] Clean Test Acc: {test_acc:.4f}")

    # ── Evaluate ScrewSet-S ──
    print(f"\n[{model_tag}] Evaluating ScrewSet-S (19 corruptions × 5 severities)...")
    ss_results = evaluate_screwset_s_supervised(
        model, eval_transform, device, batch_size, model_tag, use_amp=True)

    all_means = [v["mean"] for v in ss_results.values()]
    overall_mean = sum(all_means) / len(all_means) if all_means else 0.0

    result = {
        "model": model_name,
        "phase": 2,
        "dataset": "screwset_s",
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "model_path": str(ckpt_path),
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "resize": f"Resize({RESIZE_SHORT})+CenterCrop({IMG_SIZE})",
        "screwset_s_results": ss_results,
        "screwset_s_overall_mean": round(overall_mean, 6),
        "num_corruptions": len(ss_results),
        "num_severities": 5,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_name}_screwset_s.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    print(f"[DONE] ScrewSet-S overall mean: {overall_mean:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: CLIP Zero-Shot ScrewSet-S Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase3_clip(model_key, device, batch_size=None):
    """Zero-shot CLIP-family evaluation on ScrewSet-S."""
    print(f"\n{'='*70}")
    print(f"  Phase 3: {model_key} — ScrewSet-S Zero-Shot")
    print(f"{'='*70}")

    out_path = RESULTS_DIR / f"{model_key}_screwset_s.json"
    if out_path.exists():
        print(f"[SKIP] Results already exist: {out_path}")
        return json.load(open(out_path))

    import open_clip

    cfg = PHASE3_MODEL_CFG[model_key]
    bs = batch_size or PHASE3_BATCH_SIZES.get(model_key, 256)

    # Load model
    print(f"[MODEL] Loading {cfg['model_name']} ({cfg['pretrained']}) via open_clip...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=cfg["pretrained"])
    tokenizer = open_clip.get_tokenizer(cfg["model_name"])
    model = model.to(device).eval()

    # Build text features for ScrewSet classes
    class_names, folder_names = get_screwset_class_names(SPLIT_DIR)
    text_features = build_clip_text_features(
        model, tokenizer, class_names, SIMPLE_TEMPLATES, device)

    # First evaluate clean test set for reference
    clean_test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=preprocess,
                                 is_valid_file=is_valid_image)
    clean_loader = make_eval_loader(clean_test_ds, bs)
    clean_acc = _clip_evaluate(model, clean_loader, text_features, device,
                                desc=f"[{model_key}] Clean Test")
    print(f"[{model_key}] Clean Test Acc: {clean_acc:.4f}")

    # Evaluate ScrewSet-S
    model_tag = f"{model_key}_screwset_s"
    print(f"\n[{model_tag}] Evaluating ScrewSet-S (19 corruptions × 5 severities)...")
    ss_results = evaluate_screwset_s_clip(
        model, tokenizer, preprocess, text_features, device, bs, model_tag)

    all_means = [v["mean"] for v in ss_results.values()]
    overall_mean = sum(all_means) / len(all_means) if all_means else 0.0

    result = {
        "model": model_key,
        "phase": 3,
        "model_family": cfg["family"],
        "dataset": "screwset_s",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "clean_test_acc": round(clean_acc, 6),
        "text_templates": "simple_1",
        "screwset_s_results": ss_results,
        "screwset_s_overall_mean": round(overall_mean, 6),
        "num_corruptions": len(ss_results),
        "num_severities": 5,
        "seed": SEED,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    print(f"[DONE] ScrewSet-S overall mean: {overall_mean:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


def run_phase3_generative(model_key, device):
    """Zero-shot generative VLM evaluation on ScrewSet-S (BLIP2/LLaVA).

    NOTE: Generative VLMs are VERY slow on 19×5×20480 = ~2M images.
    Only evaluates severity 3 (the 'typical' severity) as a representative sample,
    consistent with corruption eval being skipped in phase3_vlm_baselines.py.
    """
    print(f"\n{'='*70}")
    print(f"  Phase 3: {model_key} — ScrewSet-S Zero-Shot (Generative)")
    print(f"  NOTE: Generative VLMs are too slow for full corruption eval.")
    print(f"  Skipping ScrewSet-S (same as ScrewSet-C in phase3_vlm_baselines.py)")
    print(f"{'='*70}")

    out_path = RESULTS_DIR / f"{model_key}_screwset_s.json"
    result = {
        "model": model_key,
        "phase": 3,
        "model_family": PHASE3_MODEL_CFG[model_key]["family"],
        "dataset": "screwset_s",
        "evaluation_mode": "zero_shot_generative",
        "note": "Skipped — generative VLMs too slow for corruption eval (19×5×20480 images)",
        "screwset_s_results": None,
        "screwset_s_overall_mean": None,
        "seed": SEED,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] (Skipped) Results saved to {out_path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Reporter
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    """Print a summary table of all ScrewSet-S results."""
    print(f"\n{'='*70}")
    print(f"  ScrewSet-S Results Summary")
    print(f"{'='*70}")

    results = []
    for f in sorted(RESULTS_DIR.glob("*_screwset_s.json")):
        with open(f) as fh:
            r = json.load(fh)
        results.append(r)

    if not results:
        print("  No results found.")
        return

    print(f"\n{'Model':<35s} {'Phase':>5s} {'Overall Mean':>14s}")
    print("-" * 58)
    for r in results:
        mean = r.get("screwset_s_overall_mean")
        mean_str = f"{mean:.4f}" if mean is not None else "N/A"
        print(f"  {r['model']:<33s} {r['phase']:>5d} {mean_str:>14s}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ScrewSet-S Evaluation: All Models, All Phases")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which phase(s) to run")
    parser.add_argument("--model", type=str, default="all",
                        help="Specific model name, or 'all'")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; require existing checkpoints")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--summary", action="store_true",
                        help="Just print summary of existing results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_summary()
        return

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] ScrewSet-S dir: {SS_DIR}")
    print(f"[INFO] Results dir: {RESULTS_DIR}")

    # Verify ScrewSet-S data exists
    if not SS_DIR.exists():
        print(f"[FATAL] ScrewSet-S directory not found: {SS_DIR}", file=sys.stderr)
        sys.exit(1)

    phases = ["1", "2", "3"] if args.phase == "all" else [args.phase]
    failed = []

    for phase in phases:
        if phase == "1":
            model_list = [args.model] if args.model != "all" else PHASE1_MODELS
            for m in model_list:
                if m not in PHASE1_MODELS:
                    print(f"[WARN] {m} not in Phase 1 models, skipping")
                    continue
                set_seed()
                try:
                    run_phase1(m, device, eval_only=args.eval_only,
                               batch_size=args.batch_size)
                except Exception as e:
                    print(f"[ERROR] Phase 1 {m} failed: {e}")
                    import traceback; traceback.print_exc()
                    failed.append(f"P1:{m}")

        elif phase == "2":
            model_list = [args.model] if args.model != "all" else PHASE2_MODELS
            for m in model_list:
                if m not in PHASE2_MODELS:
                    print(f"[WARN] {m} not in Phase 2 models, skipping")
                    continue
                set_seed()
                try:
                    run_phase2(m, device, eval_only=args.eval_only,
                               batch_size=args.batch_size)
                except Exception as e:
                    print(f"[ERROR] Phase 2 {m} failed: {e}")
                    import traceback; traceback.print_exc()
                    failed.append(f"P2:{m}")

        elif phase == "3":
            model_list = [args.model] if args.model != "all" else PHASE3_MODELS
            for m in model_list:
                if m not in PHASE3_MODELS:
                    print(f"[WARN] {m} not in Phase 3 models, skipping")
                    continue
                set_seed()
                try:
                    if m in PHASE3_MODELS_CLIP:
                        run_phase3_clip(m, device, batch_size=args.batch_size)
                    else:
                        run_phase3_generative(m, device)
                except Exception as e:
                    print(f"[ERROR] Phase 3 {m} failed: {e}")
                    import traceback; traceback.print_exc()
                    failed.append(f"P3:{m}")

    # Print summary
    print_summary()

    if failed:
        print(f"\n[WARN] Failed runs: {', '.join(failed)}")
    else:
        print(f"\n[SUCCESS] All runs completed!")


if __name__ == "__main__":
    main()
