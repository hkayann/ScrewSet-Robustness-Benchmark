#!/usr/bin/env python3
"""
Phase 1: FP32 Baseline Training & Evaluation
=============================================
Trains (or evaluates pretrained) CNN models on all available datasets
and evaluates robustness against corruptions.

Models:
    resnet18, squeezenet1_1, mobilenet_v3_large, mobilenetv4_conv_small,
    shufflenet_v2_x1_0, efficientnetv2_rw_s, ghostnetv2_100, convnextv2_atto

Datasets:
    cifar10      — Train from scratch, eval CIFAR-10-C
    screwset     — Train from scratch, eval ScrewSet-C
    imagenet_a   — Pretrained eval only (200-class subset)
    imagenet_val — Pretrained eval only (1000-class clean)
    imagenet_c   — Pretrained eval only (19 corruptions × 5 severities)
    lens         — Fine-tune from pretrained, eval corrupted

Usage:
    python3 phase1_baselines.py --model resnet18 --dataset cifar10
    python3 phase1_baselines.py --model all --dataset all
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED, ALL_DATASETS,
    CNN_TIMM_PRETRAINED_IDS as TIMM_PRETRAINED_IDS,
    IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA,
)
from src.utils import patch_ipv4, set_seed, make_generator, seed_worker
from src.datasets import is_valid_image, NumpyDataset, SamplesDataset
from src.imagenet_utils import get_imagenet_class_index, build_imagenet_a_mapping
from src.corruption import discover_imagenet_c_corruptions, find_corruption_leaf_dirs
from src.evaluation import evaluate, evaluate_imagenet_a

patch_ipv4()

# ═══════════════════════════════════════════════════════════════════════════════
# Phase-specific Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "phase1"
MODELS_SAVE_DIR = RESULTS_DIR / "models"

ALL_MODELS = [
    "resnet18",
    "squeezenet1_1",
    "mobilenet_v3_large",
    "mobilenetv4_conv_small",
    "shufflenet_v2_x1_0",
    "efficientnetv2_rw_s",
    "ghostnetv2_100",
    "convnextv2_atto",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(name, num_classes, pretrained=False):
    """Create a model and adapt its classifier head to `num_classes`.

    When pretrained=True and num_classes != 1000, the classifier head is
    replaced AFTER loading pretrained weights (i.e., body is pretrained,
    head is freshly initialised).
    """
    is_timm = name in TIMM_PRETRAINED_IDS

    if is_timm:
        model_id = TIMM_PRETRAINED_IDS[name] if pretrained else name
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        return model

    # ── torchvision models ────────────────────────────────────────────────
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
        raise ValueError(f"Unknown model: {name}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Training (Phase 1: Adam + early stopping, no AMP)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, device, args, model_tag):
    """Train a model with early stopping. Returns best val acc."""
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
                                      desc=f"[{model_tag}] Val {epoch}/{args.num_epochs}")

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
                print(f"[{model_tag}] Early stopping after {epoch} epochs "
                      f"(no improvement for {args.patience})")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"[{model_tag}] Training complete. Best val acc: {best_val_acc:.4f}")
    return best_val_acc, str(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_cifar10(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  CIFAR-10 — {model_name}")
    print(f"{'='*70}")

    CIFAR_ROOT = DATA_DIR / "cifar10"
    CIFAR_C_DIR = DATA_DIR / "CIFAR-10-C"

    NORMALIZATION = {"mean": (0.4914, 0.4822, 0.4465), "std": (0.247, 0.243, 0.261)}

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])

    # Load CIFAR-10 directly (no ImageFolder split needed)
    full_train = datasets.CIFAR10(str(CIFAR_ROOT), train=True, download=False,
                                   transform=train_transform)
    test_ds = datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False,
                                transform=eval_transform)

    # Stratified train/val split (45000 / 5000)
    set_seed()
    indices = list(range(len(full_train)))
    labels = np.array(full_train.targets)
    train_idx, val_idx = [], []
    for cls in range(10):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = 500  # 5000 val total / 10 classes
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())

    train_ds = Subset(full_train, train_idx)

    # Validation needs eval transform, not train transform
    full_val = datasets.CIFAR10(str(CIFAR_ROOT), train=True, download=False,
                                 transform=eval_transform)
    val_ds = Subset(full_val, val_idx)

    gen = make_generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    NUM_CLASSES = 10
    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Create model
    set_seed()
    model = create_model(model_name, NUM_CLASSES, pretrained=False)
    model = model.to(device)
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
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
            _, acc = evaluate(model, loader, criterion, device,
                               desc=f"[{model_tag}] C-{fname}")
            corrupt_results[fname.replace(".npy", "")] = acc
            print(f"  CIFAR-10-C {fname.replace('.npy', '')}: {acc:.4f}")
    else:
        print("[WARN] CIFAR-10-C directory not found, skipping corruption eval")

    # Save results
    result = {
        "model": model_name,
        "dataset": "cifar10",
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "normalization_mean": NORMALIZATION["mean"],
        "normalization_std": NORMALIZATION["std"],
        "resize": [224, 224],
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
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
    print(f"  ScrewSet — {model_name}")
    print(f"{'='*70}")

    SPLIT_DIR = DATA_DIR / "screwset_split"
    CORRUPT_ROOT = DATA_DIR / "screwset_c"

    NORMALIZATION = {"mean": [0.7750, 0.7343, 0.6862], "std": [0.0802, 0.0838, 0.0871]}
    RESIZE_DIM = (240, 320)

    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
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

    gen = make_generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Create model
    set_seed()
    model = create_model(model_name, NUM_CLASSES, pretrained=False)
    model = model.to(device)
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
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)
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
        "dataset": "screwset",
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "normalization_mean": NORMALIZATION["mean"],
        "normalization_std": NORMALIZATION["std"],
        "resize": list(RESIZE_DIM),
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
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
    print(f"  ImageNet-A — {model_name}")
    print(f"{'='*70}")

    IMAGENET_A_DIR = DATA_DIR / "imagenet-a"
    if not IMAGENET_A_DIR.exists():
        print("[ERROR] ImageNet-A directory not found — skipping")
        return None

    NORMALIZATION = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])

    # Build class mapping
    class_mapping = build_imagenet_a_mapping(str(IMAGENET_A_DIR))

    # Load dataset
    ds = ImageFolder(str(IMAGENET_A_DIR), transform=eval_transform,
                      is_valid_file=is_valid_image)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    print(f"[INFO] ImageNet-A: {len(ds)} images, {len(ds.classes)} classes")

    # Load pretrained model (1000 classes — DO NOT replace head)
    set_seed()
    model = create_model(model_name, num_classes=1000, pretrained=True)
    model = model.to(device)
    model_tag = f"{model_name}_imagenet_a"

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    acc = evaluate_imagenet_a(model, loader, criterion, device, class_mapping,
                               desc=f"[{model_tag}] Evaluating")
    print(f"[{model_tag}] ImageNet-A Accuracy: {acc:.4f}")

    # Save results
    result = {
        "model": model_name,
        "dataset": "imagenet_a",
        "pretrained": True,
        "num_imagenet_a_classes": len(class_mapping),
        "imagenet_a_acc": acc,
        "normalization_mean": NORMALIZATION["mean"],
        "normalization_std": NORMALIZATION["std"],
        "resize": 256,
        "crop": 224,
        "batch_size": args.batch_size,
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
# ImageNet Validation (clean, 1000-class, eval-only with pretrained weights)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_val(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet Val — {model_name}")
    print(f"{'='*70}")

    IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
    if not IMAGENET_VAL_DIR.exists():
        print("[ERROR] ImageNet val directory not found — skipping")
        return None

    NORMALIZATION = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])

    ds = ImageFolder(str(IMAGENET_VAL_DIR), transform=eval_transform,
                      is_valid_file=is_valid_image)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    print(f"[INFO] ImageNet Val: {len(ds)} images, {len(ds.classes)} classes")

    # Load pretrained model (1000 classes — DO NOT replace head)
    set_seed()
    model = create_model(model_name, num_classes=1000, pretrained=True)
    model = model.to(device)
    model_tag = f"{model_name}_imagenet_val"

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, loader, criterion, device,
                                  desc=f"[{model_tag}] Evaluating")
    print(f"[{model_tag}] ImageNet Val: Loss={val_loss:.4f} Acc={val_acc:.4f}")

    # Per-class accuracy (top-1)
    model.eval()
    class_correct = {}
    class_total = {}
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"[{model_tag}] Per-class", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            for t, p in zip(targets, preds):
                t_item = t.item()
                class_total[t_item] = class_total.get(t_item, 0) + 1
                class_correct[t_item] = class_correct.get(t_item, 0) + (1 if p.item() == t_item else 0)

    per_class_acc = {k: class_correct.get(k, 0) / class_total[k]
                     for k in sorted(class_total.keys())}
    mean_per_class_acc = sum(per_class_acc.values()) / len(per_class_acc) if per_class_acc else 0.0

    # Save results
    result = {
        "model": model_name,
        "dataset": "imagenet_val",
        "pretrained": True,
        "num_classes": len(ds.classes),
        "num_images": len(ds),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "mean_per_class_acc": mean_per_class_acc,
        "normalization_mean": NORMALIZATION["mean"],
        "normalization_std": NORMALIZATION["std"],
        "resize": 256,
        "crop": 224,
        "batch_size": args.batch_size,
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
# ImageNet-C (19 corruptions × 5 severities, eval-only with pretrained weights)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_c(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet-C — {model_name}")
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

    NORMALIZATION = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])

    # Load pretrained model (1000 classes — DO NOT replace head)
    set_seed()
    model = create_model(model_name, num_classes=1000, pretrained=True)
    model = model.to(device)
    model_tag = f"{model_name}_imagenet_c"

    criterion = nn.CrossEntropyLoss()

    # Evaluate each corruption at each severity level
    corruption_results = {}  # {corruption: {severity: acc}}
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
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

            _, acc = evaluate(model, loader, criterion, device,
                               desc=f"[{model_tag}] {cname}/s{sev}")
            corruption_results[cname][str(sev)] = acc
            print(f"  {cname} sev-{sev}: {acc:.4f}")

        # Mean across severities for this corruption
        accs = [v for v in corruption_results[cname].values() if v is not None]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        corruption_results[cname]["mean"] = mean_acc
        print(f"  {cname} mean: {mean_acc:.4f}")

    # Compute aggregate metrics
    # mCE on standard 15 corruptions (unnormalized, = 1 - mean_acc)
    std15_accs = []
    for cname in IMAGENET_C_CORRUPTIONS_15:
        if cname in corruption_results and "mean" in corruption_results[cname]:
            std15_accs.append(corruption_results[cname]["mean"])
    mean_acc_15 = sum(std15_accs) / len(std15_accs) if std15_accs else 0.0
    mce_15 = 1.0 - mean_acc_15  # unnormalized mean corruption error

    # All corruptions (15 + extra)
    all_accs = [cr["mean"] for cr in corruption_results.values()
                if "mean" in cr and cr["mean"] is not None]
    mean_acc_all = sum(all_accs) / len(all_accs) if all_accs else 0.0
    mce_all = 1.0 - mean_acc_all

    print(f"\n[{model_tag}] Mean Acc (15 std): {mean_acc_15:.4f}  mCE: {mce_15:.4f}")
    print(f"[{model_tag}] Mean Acc (all {len(all_accs)}): {mean_acc_all:.4f}  mCE: {mce_all:.4f}")

    # Save results
    result = {
        "model": model_name,
        "dataset": "imagenet_c",
        "pretrained": True,
        "num_corruptions_evaluated": len(corruption_results),
        "corruption_results": corruption_results,
        "mean_acc_15_std": mean_acc_15,
        "mce_15_std": mce_15,
        "mean_acc_all": mean_acc_all,
        "mce_all": mce_all,
        "normalization_mean": NORMALIZATION["mean"],
        "normalization_std": NORMALIZATION["std"],
        "resize": 256,
        "crop": 224,
        "batch_size": args.batch_size,
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
# Lens / ImageNet-ES pipeline (fine-tune + eval)
# ═══════════════════════════════════════════════════════════════════════════════

def run_lens(model_name, args, device):
    print(f"\n{'='*70}")
    print(f"  Lens / ImageNet-ES — {model_name}")
    print(f"{'='*70}")

    LENS_DIR = DATA_DIR / "lens_split"
    if not LENS_DIR.exists():
        print("[ERROR] Lens split directory not found — skipping")
        return None

    NORMALIZATION = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
    ])

    train_ds = ImageFolder(str(LENS_DIR / "train"), transform=train_transform)
    val_ds = ImageFolder(str(LENS_DIR / "validation"), transform=eval_transform)
    test_ds = ImageFolder(str(LENS_DIR / "test"), transform=eval_transform)

    NUM_CLASSES = len(train_ds.classes)
    print(f"[INFO] Classes: {NUM_CLASSES}, Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Verify class consistency
    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        print("[ERROR] Class mismatch between train/val/test splits!")
        return None

    gen = make_generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Fine-tune from pretrained (replace head for NUM_CLASSES)
    set_seed()
    model = create_model(model_name, NUM_CLASSES, pretrained=True)
    model = model.to(device)
    model_tag = f"{model_name}_lens"

    # Train
    best_val_acc, ckpt_path = train_model(model, train_loader, val_loader,
                                           device, args, model_tag)

    # Evaluate clean test
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
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)
                _, acc = evaluate(model, loader, criterion, device,
                                   desc=f"[{model_tag}] {rel}")
                corrupt_results[rel] = acc
                print(f"  Lens corrupt {rel}: {acc:.4f}")
            except Exception as e:
                print(f"  [ERROR] {rel}: {e}")
                corrupt_results[rel] = None
    else:
        print("[WARN] Lens corrupted directory not found")

    # Save results
    result = {
        "model": model_name,
        "dataset": "lens",
        "pretrained_finetune": True,
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "model_path": ckpt_path,
        "normalization_mean": NORMALIZATION["mean"],
        "normalization_std": NORMALIZATION["std"],
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
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


def main():
    parser = argparse.ArgumentParser(description="Phase 1: FP32 Baseline Training & Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=ALL_MODELS + ["all"],
                        help="Model architecture to use")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=ALL_DATASETS + ["all"],
                        help="Dataset to train/evaluate on")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
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
    dataset_list = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    total_runs = len(model_list) * len(dataset_list)
    run_idx = 0

    for ds_name in dataset_list:
        runner = DATASET_RUNNERS[ds_name]
        for m_name in model_list:
            run_idx += 1
            print(f"\n{'#'*70}")
            print(f"  RUN {run_idx}/{total_runs}: {m_name} × {ds_name}")
            print(f"{'#'*70}")

            set_seed()
            try:
                runner(m_name, args, device)
            except Exception as e:
                print(f"[ERROR] {m_name} × {ds_name} failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  ALL RUNS COMPLETE ({run_idx}/{total_runs})")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
