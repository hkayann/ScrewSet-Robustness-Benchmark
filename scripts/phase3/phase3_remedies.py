#!/usr/bin/env python3
"""
Phase 3 Remedies: Fix Missing Corruption Data + Open-Ended VQA Ablation
========================================================================
Addresses two NeurIPS review concerns:

1. BLIP-2/LLaVA corruption data was skipped (too slow) — now run:
   - ScrewSet-C: 7,680 images (~40 min total for both)
   - Lens-C: 192,000 images (~16 hrs total for both)
   - CIFAR-10-C: subsampled 1K per corruption (19K per model, ~8 hrs)
   - ImageNet-C: remains skipped (4.75M images, 400+ hrs infeasible)

2. LLaVA 100% on ScrewSet reflects VQA format advantage — add:
   - Open-ended VQA: ask "What type of screw is this?" WITHOUT listing
     class names, then fuzzy-match free-form response.
   - Report both closed-form and open-ended accuracy for BLIP-2 + LLaVA
     on ScrewSet to quantify the prompt format effect.

Usage:
    python3 phase3_remedies.py                      # Run all remedies
    python3 phase3_remedies.py --smoke-test          # Quick sanity check
    python3 phase3_remedies.py --only-corruption     # Skip VQA ablation
    python3 phase3_remedies.py --only-vqa-ablation   # Skip corruption eval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import DATA_DIR, SEED, CIFAR10_CLASSES
from src.utils import patch_ipv4, set_seed
from src.datasets import is_valid_image, PILImageFolder, PILNumpyDataset
from src.imagenet_utils import get_imagenet_class_index
from src.class_names import screwset_folder_to_text, get_screwset_class_names, get_lens_class_names
from src.corruption import find_corruption_leaf_dirs

patch_ipv4()

RESULTS_DIR = REPO_ROOT / "results" / "phase3"
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# Subsample size per corruption for CIFAR-10-C (19 corruptions × 5 sev × 1K = 95K)
CIFAR10C_SUBSAMPLE_PER_SEVERITY = 1000

# Batch sizes for generative models
BLIP2_BATCH = 64
LLAVA_BATCH = 4


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_blip2_model(device):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    model_id = "Salesforce/blip2-opt-2.7b"
    print(f"[MODEL] Loading BLIP-2: {model_id} ...")
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] blip2: {n_params:,} params on {device}")
    return model, processor


def load_llava_model(device):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"[MODEL] Loading LLaVA: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] llava: {n_params:,} params on {device}")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Matching Logic (same as original phase3 script)
# ═══════════════════════════════════════════════════════════════════════════════

def _match_response_to_class(response, target_name, class_names_lower, target_idx):
    """Match a generative model's text response to the target class name.

    Uses best-match strategy: find which class the response best matches
    via fuzzy similarity. Avoids single-word overlap which is too permissive
    when a common word (like "screw") appears in every class name.

    Steps:
    1. Exact match → correct
    2. Best fuzzy match among all classes → correct iff best = target
    """
    from difflib import SequenceMatcher

    # 1. Exact match
    if response == target_name:
        return True

    # 2. Best fuzzy match among ALL class names
    best_ratio = 0.0
    best_idx = -1
    for idx, cn in enumerate(class_names_lower):
        ratio = SequenceMatcher(None, response, cn).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    return best_idx == target_idx and best_ratio > 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Closed-Form VQA Evaluation (list all class names in prompt)
# ═══════════════════════════════════════════════════════════════════════════════

def blip2_evaluate_closed(model, processor, class_names, dataset, device,
                          desc="BLIP-2", batch_size=64):
    """BLIP-2 generative zero-shot with class names listed (closed-form)."""
    class_names_lower = [c.lower() for c in class_names]
    class_list = ", ".join(class_names)
    prompt = (f"Question: Classify this image. The categories are: "
              f"{class_list}. Answer with only the category name. Answer:")

    correct, total = 0, 0
    _debug = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, len(dataset))
        imgs, labels = [], []
        for j in range(i, batch_end):
            img, label = dataset[j]
            imgs.append(img)
            labels.append(label)
        try:
            inputs = processor(images=imgs, text=[prompt]*len(imgs),
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                input_len = inputs["input_ids"].shape[-1]
                outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            generated = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            for text, label in zip(generated, labels):
                text_l = text.lower().strip().rstrip(".")
                if _debug < 3:
                    print(f"  [DBG BLIP2-closed] resp={text_l!r} tgt={class_names_lower[label]!r}")
                    _debug += 1
                if _match_response_to_class(text_l, class_names_lower[label], class_names_lower, label):
                    correct += 1
                total += 1
        except Exception as e:
            print(f"  [WARN] BLIP-2 batch {i} error: {e}")
            total += len(labels)
    return correct / total if total else 0.0


def llava_evaluate_closed(model, processor, class_names, dataset, device,
                          desc="LLaVA", batch_size=4):
    """LLaVA generative zero-shot with class names listed (closed-form)."""
    class_names_lower = [c.lower() for c in class_names]

    if len(class_names) <= 40:
        class_list = ", ".join(class_names)
        user_text = (f"Classify this image into exactly one of these categories: "
                     f"{class_list}. Answer with only the category name, nothing else.")
    else:
        user_text = ("What is the main subject of this image? "
                     "Answer with a single word or very short phrase.")

    _use_chat_template = hasattr(processor, "apply_chat_template")
    if _use_chat_template:
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text}]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        prompt_text = f"USER: <image>\n{user_text}\nASSISTANT:"

    correct, total = 0, 0
    _debug = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, len(dataset))
        imgs, labels = [], []
        for j in range(i, batch_end):
            img, label = dataset[j]
            imgs.append(img)
            labels.append(label)
        try:
            inputs = processor(text=[prompt_text]*len(imgs), images=imgs,
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                input_len = inputs["input_ids"].shape[-1]
                outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            generated = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            for text, label in zip(generated, labels):
                text_l = text.lower().strip().rstrip(".")
                if _debug < 3:
                    print(f"  [DBG LLaVA-closed] resp={text_l!r} tgt={class_names_lower[label]!r}")
                    _debug += 1
                if _match_response_to_class(text_l, class_names_lower[label], class_names_lower, label):
                    correct += 1
                total += 1
        except Exception as e:
            print(f"  [WARN] LLaVA batch {i} error: {e}")
            total += len(labels)
    return correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Open-Ended VQA Evaluation (NO class names in prompt)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_screw_keywords(text):
    """Extract discriminative screw keywords from a text response.

    Returns a normalized keyword string for matching, stripping
    common filler words but preserving size, length, and attribute info.
    """
    import re
    text = text.lower().strip()

    # Extract size designations
    sizes = re.findall(r'\bm[3-6](?:\.2)?\b', text)  # M3, M4, M4.2, M5, M6
    sizes += re.findall(r'\bnumber\s*(?:7|8|10)\b', text)
    sizes += re.findall(r'\b#\s*(?:7|8|10)\b', text)  # #7, #8, #10

    # Extract lengths (number followed by mm or millimeter)
    lengths = re.findall(r'(\d+)\s*(?:mm|millimeter)', text)

    # Extract attributes
    attrs = []
    if 'flat' in text:
        attrs.append('flat')
    if 'round' in text:
        attrs.append('round')
    if 'black' in text:
        attrs.append('black')
    if 'yellow' in text:
        attrs.append('yellow')
    if '4.6' in text or 'grade' in text:
        attrs.append('grade 4.6')

    # Build normalized string
    parts = sizes + [f"{l}mm" for l in lengths] + attrs + ['screw']
    return " ".join(parts)


def _match_response_open(response, target_name, class_names_lower, target_idx):
    """Stricter matching for open-ended VQA (no class names in prompt).

    Strategy: extract structured keywords from the response, then find
    the best-matching class among all classes. If the best match is the
    target class, count as correct.
    """
    from difflib import SequenceMatcher

    # Reject empty/trivial responses
    if len(response.strip()) < 3:
        return False

    # Extract keywords from the response
    kw_response = _extract_screw_keywords(response)

    # If keyword extraction yields nothing useful, fall back to raw response
    if len(kw_response.strip()) <= 6:  # Only "screw"
        query = response
    else:
        query = kw_response

    # Find best matching class by fuzzy similarity
    best_ratio = 0.0
    best_idx = -1
    for idx, cn in enumerate(class_names_lower):
        ratio = SequenceMatcher(None, query, cn).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    # Must match target class — no threshold (pure nearest-neighbor)
    return best_idx == target_idx


def blip2_evaluate_open(model, processor, class_names, dataset, device,
                        desc="BLIP-2 Open", batch_size=64):
    """BLIP-2 open-ended: ask about screw without listing categories."""
    class_names_lower = [c.lower() for c in class_names]
    # BLIP-2 (OPT-based) needs a direct, simple question format
    prompt = "Question: What type of screw is this? Describe the size, length, and head type. Answer:"

    correct, total = 0, 0
    _debug = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, len(dataset))
        imgs, labels = [], []
        for j in range(i, batch_end):
            img, label = dataset[j]
            imgs.append(img)
            labels.append(label)
        try:
            inputs = processor(images=imgs, text=[prompt]*len(imgs),
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                input_len = inputs["input_ids"].shape[-1]
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generated = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            for text, label in zip(generated, labels):
                text_l = text.lower().strip().rstrip(".")
                if _debug < 8:
                    kw = _extract_screw_keywords(text_l)
                    print(f"  [DBG BLIP2-open] resp={text_l!r} kw={kw!r} tgt={class_names_lower[label]!r}")
                    _debug += 1
                if _match_response_open(text_l, class_names_lower[label], class_names_lower, label):
                    correct += 1
                total += 1
        except Exception as e:
            print(f"  [WARN] BLIP-2 open batch {i} error: {e}")
            total += len(labels)
    return correct / total if total else 0.0


def llava_evaluate_open(model, processor, class_names, dataset, device,
                        desc="LLaVA Open", batch_size=4):
    """LLaVA open-ended: ask about screw without listing categories."""
    class_names_lower = [c.lower() for c in class_names]

    user_text = ("Describe this screw precisely. "
                 "What is its size designation (e.g., M3, M4, M5, M6, number 7, 8, 10), "
                 "length in millimeters, and head type (flat or round) or color (black or yellow)? "
                 "Answer with a short description.")

    _use_chat_template = hasattr(processor, "apply_chat_template")
    if _use_chat_template:
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text}]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        prompt_text = f"USER: <image>\n{user_text}\nASSISTANT:"

    correct, total = 0, 0
    _debug = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, len(dataset))
        imgs, labels = [], []
        for j in range(i, batch_end):
            img, label = dataset[j]
            imgs.append(img)
            labels.append(label)
        try:
            inputs = processor(text=[prompt_text]*len(imgs), images=imgs,
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                input_len = inputs["input_ids"].shape[-1]
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generated = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            for text, label in zip(generated, labels):
                text_l = text.lower().strip().rstrip(".")
                if _debug < 8:
                    kw = _extract_screw_keywords(text_l)
                    print(f"  [DBG LLaVA-open] resp={text_l!r}")
                    print(f"                   kw={kw!r} tgt={class_names_lower[label]!r}")
                    _debug += 1
                if _match_response_open(text_l, class_names_lower[label], class_names_lower, label):
                    correct += 1
                total += 1
        except Exception as e:
            print(f"  [WARN] LLaVA open batch {i} error: {e}")
            total += len(labels)
    return correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Remedy 1: Corruption Evaluation for Generative Models
# ═══════════════════════════════════════════════════════════════════════════════

def run_cifar10c_generative(model_key, model, processor, class_names, device,
                             batch_size, subsample=CIFAR10C_SUBSAMPLE_PER_SEVERITY):
    """Evaluate BLIP-2 or LLaVA on CIFAR-10-C (subsampled)."""
    print(f"\n{'='*70}")
    print(f"  REMEDY: CIFAR-10-C — {model_key} (subsampled {subsample}/severity)")
    print(f"{'='*70}")

    CIFAR_C_DIR = DATA_DIR / "CIFAR-10-C"
    if not CIFAR_C_DIR.exists():
        # Try alternative path
        CIFAR_C_DIR = DATA_DIR / "cifar10_c" / "CIFAR-10-C"
    if not CIFAR_C_DIR.exists():
        print("[ERROR] CIFAR-10-C directory not found")
        return None

    labels_all = np.load(str(CIFAR_C_DIR / "labels.npy"))  # 50000 labels
    class_names_lower = [c.lower() for c in class_names]
    eval_fn = blip2_evaluate_closed if model_key == "blip2" else llava_evaluate_closed

    corrupt_results = {}
    for fname in sorted(os.listdir(CIFAR_C_DIR)):
        if not fname.endswith(".npy") or fname == "labels.npy":
            continue
        cname = fname.replace(".npy", "")
        print(f"\n  Processing {cname}...")

        images_all = np.load(str(CIFAR_C_DIR / fname))  # (50000, 32, 32, 3)

        severity_accs = []
        for sev in range(5):
            # Each severity is 10000 images (indices: sev*10000 to (sev+1)*10000)
            start_idx = sev * 10000
            end_idx = (sev + 1) * 10000

            # Subsample
            rng = np.random.RandomState(SEED + sev)
            if subsample < 10000:
                indices = rng.choice(10000, subsample, replace=False)
            else:
                indices = np.arange(10000)

            sub_images = images_all[start_idx + indices]
            sub_labels = labels_all[start_idx + indices]

            ds = PILNumpyDataset(sub_images, sub_labels)
            acc = eval_fn(model, processor, class_names, ds, device,
                          desc=f"{cname} sev{sev+1}", batch_size=batch_size)
            severity_accs.append(acc)
            print(f"    sev{sev+1}: {acc:.4f} ({len(ds)} images)")

        mean_acc = sum(severity_accs) / len(severity_accs)
        corrupt_results[cname] = {
            "severity_accs": severity_accs,
            "mean": mean_acc,
        }
        print(f"    mean: {mean_acc:.4f}")

    mean_all = np.mean([v["mean"] for v in corrupt_results.values()])
    return corrupt_results, float(mean_all)


def run_screwsetc_generative(model_key, model, processor, class_names, device,
                              batch_size):
    """Evaluate BLIP-2 or LLaVA on ScrewSet-C (full, only 7680 images)."""
    print(f"\n{'='*70}")
    print(f"  REMEDY: ScrewSet-C — {model_key} (full)")
    print(f"{'='*70}")

    CORRUPT_ROOT = DATA_DIR / "screwset_c"
    if not CORRUPT_ROOT.exists():
        print("[ERROR] ScrewSet-C directory not found")
        return None

    eval_fn = blip2_evaluate_closed if model_key == "blip2" else llava_evaluate_closed
    corrupt_results = {}

    for corrupt_type in sorted(os.listdir(CORRUPT_ROOT)):
        corrupt_dir = CORRUPT_ROOT / corrupt_type
        if not corrupt_dir.is_dir():
            continue
        try:
            ds = PILImageFolder(str(corrupt_dir), is_valid_file=is_valid_image)
            acc = eval_fn(model, processor, class_names, ds, device,
                          desc=f"ScrewSet-C {corrupt_type}", batch_size=batch_size)
            corrupt_results[corrupt_type] = acc
            print(f"  {corrupt_type}: {acc:.4f} ({len(ds)} images)")
        except Exception as e:
            print(f"  [ERROR] {corrupt_type}: {e}")
            corrupt_results[corrupt_type] = None

    valid = [v for v in corrupt_results.values() if v is not None]
    mean_all = sum(valid) / len(valid) if valid else None
    return corrupt_results, mean_all


def run_lensc_generative(model_key, model, processor, class_names, device,
                          batch_size):
    """Evaluate BLIP-2 or LLaVA on Lens-C (full, 192K images via symlinks)."""
    print(f"\n{'='*70}")
    print(f"  REMEDY: Lens-C — {model_key} (full)")
    print(f"{'='*70}")

    corrupt_root = DATA_DIR / "lens_split" / "corrupted"
    if not corrupt_root.exists():
        print("[ERROR] Lens corrupted directory not found")
        return None

    eval_fn = blip2_evaluate_closed if model_key == "blip2" else llava_evaluate_closed
    corrupt_results = {}

    leaf_dirs = find_corruption_leaf_dirs(corrupt_root)
    print(f"[INFO] Found {len(leaf_dirs)} corrupted subsets")

    for leaf in leaf_dirs:
        rel = str(leaf.relative_to(corrupt_root))
        try:
            ds = PILImageFolder(str(leaf))
            acc = eval_fn(model, processor, class_names, ds, device,
                          desc=f"Lens-C {rel}", batch_size=batch_size)
            corrupt_results[rel] = acc
            print(f"  {rel}: {acc:.4f} ({len(ds)} images)")
        except Exception as e:
            print(f"  [ERROR] {rel}: {e}")
            corrupt_results[rel] = None

    valid = [v for v in corrupt_results.values() if v is not None]
    mean_all = sum(valid) / len(valid) if valid else None
    return corrupt_results, mean_all


# ═══════════════════════════════════════════════════════════════════════════════
# Remedy 2: Open-Ended VQA Ablation on ScrewSet
# ═══════════════════════════════════════════════════════════════════════════════

def run_vqa_ablation(model_key, model, processor, class_names, device, batch_size):
    """Run both closed-form and open-ended VQA on ScrewSet test set."""
    print(f"\n{'='*70}")
    print(f"  VQA ABLATION: ScrewSet — {model_key}")
    print(f"{'='*70}")

    SPLIT_DIR = DATA_DIR / "screwset_split"
    test_ds = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
    print(f"[INFO] ScrewSet test: {len(test_ds)} images, {len(class_names)} classes")

    # Closed-form (same as original)
    print(f"\n[EVAL] Closed-form VQA (class names listed in prompt)...")
    if model_key == "blip2":
        closed_acc = blip2_evaluate_closed(model, processor, class_names, test_ds, device,
                                           desc=f"[{model_key}] Closed", batch_size=batch_size)
    else:
        closed_acc = llava_evaluate_closed(model, processor, class_names, test_ds, device,
                                           desc=f"[{model_key}] Closed", batch_size=batch_size)
    print(f"[{model_key}] Closed-form acc: {closed_acc:.4f}")

    # Open-ended (no class names in prompt)
    print(f"\n[EVAL] Open-ended VQA (no class names in prompt)...")
    if model_key == "blip2":
        open_acc = blip2_evaluate_open(model, processor, class_names, test_ds, device,
                                       desc=f"[{model_key}] Open", batch_size=batch_size)
    else:
        open_acc = llava_evaluate_open(model, processor, class_names, test_ds, device,
                                       desc=f"[{model_key}] Open", batch_size=batch_size)
    print(f"[{model_key}] Open-ended acc: {open_acc:.4f}")

    gap = closed_acc - open_acc
    print(f"[{model_key}] Gap (closed - open): {gap:.4f} ({gap*100:.1f}pp)")

    return closed_acc, open_acc


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test(device):
    """Quick smoke test: 32 images from CIFAR-10-C + 32 from ScrewSet open VQA."""
    print("\n" + "="*70)
    print("  SMOKE TEST — Phase 3 Remedies")
    print("="*70)

    set_seed()

    # Test BLIP-2 on a tiny CIFAR-10-C subset
    print("\n[1/4] Loading BLIP-2...")
    model, processor = load_blip2_model(device)

    CIFAR_C_DIR = DATA_DIR / "CIFAR-10-C"
    if not CIFAR_C_DIR.exists():
        CIFAR_C_DIR = DATA_DIR / "cifar10_c" / "CIFAR-10-C"
    labels_all = np.load(str(CIFAR_C_DIR / "labels.npy"))
    images = np.load(str(CIFAR_C_DIR / "brightness.npy"))[:32]
    labels = labels_all[:32]
    ds = PILNumpyDataset(images, labels)

    print("\n[2/4] BLIP-2 CIFAR-10-C smoke (32 images, brightness)...")
    acc = blip2_evaluate_closed(model, processor, CIFAR10_CLASSES, ds, device,
                                desc="smoke", batch_size=16)
    print(f"  BLIP-2 CIFAR-10-C brightness smoke: {acc:.4f}")
    assert 0 <= acc <= 1, f"Invalid accuracy: {acc}"

    # Test BLIP-2 open VQA on ScrewSet
    print("\n[3/4] BLIP-2 ScrewSet open VQA (64 images, diverse classes)...")
    SPLIT_DIR = DATA_DIR / "screwset_split"
    class_names, _ = get_screwset_class_names(SPLIT_DIR)
    ss_ds = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
    # Sample from diverse classes (every 320th image → ~64 images across all classes)
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(ss_ds), 64, replace=False)
    tiny_ss = [(ss_ds[i][0], ss_ds[i][1]) for i in indices]
    open_acc = blip2_evaluate_open(model, processor, class_names, tiny_ss, device,
                                   desc="smoke-open", batch_size=16)
    print(f"  BLIP-2 ScrewSet open VQA smoke: {open_acc:.4f}")
    assert 0 <= open_acc <= 1

    closed_acc = blip2_evaluate_closed(model, processor, class_names, tiny_ss, device,
                                       desc="smoke-closed", batch_size=16)
    print(f"  BLIP-2 ScrewSet closed VQA smoke: {closed_acc:.4f}")

    del model
    torch.cuda.empty_cache()

    # Quick LLaVA test
    print("\n[4/4] LLaVA ScrewSet open VQA smoke (32 images)...")
    model, processor = load_llava_model(device)
    open_acc = llava_evaluate_open(model, processor, class_names, tiny_ss, device,
                                   desc="smoke-open", batch_size=4)
    print(f"  LLaVA ScrewSet open VQA smoke: {open_acc:.4f}")

    closed_acc = llava_evaluate_closed(model, processor, class_names, tiny_ss, device,
                                       desc="smoke-closed", batch_size=4)
    print(f"  LLaVA ScrewSet closed VQA smoke: {closed_acc:.4f}")

    del model
    torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("  SMOKE TEST PASSED ✓")
    print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_remedies(device, skip_corruption=False, skip_vqa=False):
    """Run all remedy evaluations."""
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = []
    failed = []

    for model_key in ["blip2", "llava"]:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_key}")
        print(f"{'#'*70}")

        batch_size = BLIP2_BATCH if model_key == "blip2" else LLAVA_BATCH

        if model_key == "blip2":
            model, processor = load_blip2_model(device)
        else:
            model, processor = load_llava_model(device)

        # ── Remedy 1: Corruption Evaluation ──
        if not skip_corruption:
            # ScrewSet-C
            try:
                class_names_ss, _ = get_screwset_class_names(DATA_DIR / "screwset_split")
                result = run_screwsetc_generative(model_key, model, processor,
                                                  class_names_ss, device, batch_size)
                if result is not None:
                    corrupt_results, mean_acc = result
                    # Update existing JSON
                    existing_path = RESULTS_DIR / f"{model_key}_screwset_baselines.json"
                    if existing_path.exists():
                        with open(existing_path) as f:
                            existing = json.load(f)
                    else:
                        existing = {}
                    existing["corrupt_results"] = corrupt_results
                    existing["mean_corrupt_acc"] = mean_acc
                    existing["corruption_eval_note"] = "Remedy run: full ScrewSet-C evaluation"
                    with open(existing_path, "w") as f:
                        json.dump(existing, f, indent=4)
                    print(f"[SAVED] Updated {existing_path}")
                    completed.append(f"{model_key}_screwset_c")
            except Exception as e:
                print(f"[FAILED] {model_key} ScrewSet-C: {e}")
                import traceback; traceback.print_exc()
                failed.append(f"{model_key}_screwset_c")

            # CIFAR-10-C (subsampled)
            try:
                result = run_cifar10c_generative(model_key, model, processor,
                                                  CIFAR10_CLASSES, device, batch_size)
                if result is not None:
                    corrupt_results, mean_acc = result
                    existing_path = RESULTS_DIR / f"{model_key}_cifar10_baselines.json"
                    if existing_path.exists():
                        with open(existing_path) as f:
                            existing = json.load(f)
                    else:
                        existing = {}
                    existing["corrupt_results"] = corrupt_results
                    existing["mean_corrupt_acc"] = float(mean_acc)
                    existing["corruption_eval_note"] = (
                        f"Remedy run: CIFAR-10-C subsampled "
                        f"{CIFAR10C_SUBSAMPLE_PER_SEVERITY} per severity"
                    )
                    with open(existing_path, "w") as f:
                        json.dump(existing, f, indent=4)
                    print(f"[SAVED] Updated {existing_path}")
                    completed.append(f"{model_key}_cifar10_c")
            except Exception as e:
                print(f"[FAILED] {model_key} CIFAR-10-C: {e}")
                import traceback; traceback.print_exc()
                failed.append(f"{model_key}_cifar10_c")

            # Lens-C
            try:
                class_names_lens = get_lens_class_names(DATA_DIR / "lens_split")
                result = run_lensc_generative(model_key, model, processor,
                                              class_names_lens, device, batch_size)
                if result is not None:
                    corrupt_results, mean_acc = result
                    existing_path = RESULTS_DIR / f"{model_key}_lens_baselines.json"
                    if existing_path.exists():
                        with open(existing_path) as f:
                            existing = json.load(f)
                    else:
                        existing = {}
                    existing["corrupt_results"] = corrupt_results
                    existing["mean_corrupt_acc"] = float(mean_acc)
                    existing["corruption_eval_note"] = "Remedy run: full Lens-C evaluation"
                    with open(existing_path, "w") as f:
                        json.dump(existing, f, indent=4)
                    print(f"[SAVED] Updated {existing_path}")
                    completed.append(f"{model_key}_lens_c")
            except Exception as e:
                print(f"[FAILED] {model_key} Lens-C: {e}")
                import traceback; traceback.print_exc()
                failed.append(f"{model_key}_lens_c")

        # ── Remedy 2: VQA Ablation (ScrewSet only) ──
        if not skip_vqa:
            try:
                class_names_ss, folder_names = get_screwset_class_names(
                    DATA_DIR / "screwset_split")
                closed_acc, open_acc = run_vqa_ablation(
                    model_key, model, processor, class_names_ss, device, batch_size)

                ablation_result = {
                    "model": model_key,
                    "dataset": "screwset",
                    "experiment": "vqa_prompt_ablation",
                    "closed_form_acc": closed_acc,
                    "open_ended_acc": open_acc,
                    "gap_pp": (closed_acc - open_acc) * 100,
                    "num_classes": len(class_names_ss),
                    "class_name_mapping": dict(zip(folder_names, class_names_ss)),
                    "closed_prompt": "Classify...categories: [list]. Answer with only the category name.",
                    "open_prompt": "Describe this screw precisely. Size, length, head type/color.",
                    "seed": SEED,
                }
                out_path = RESULTS_DIR / f"{model_key}_screwset_vqa_ablation.json"
                with open(out_path, "w") as f:
                    json.dump(ablation_result, f, indent=4)
                print(f"[SAVED] {out_path}")
                completed.append(f"{model_key}_vqa_ablation")
            except Exception as e:
                print(f"[FAILED] {model_key} VQA ablation: {e}")
                import traceback; traceback.print_exc()
                failed.append(f"{model_key}_vqa_ablation")

        # Free model
        del model
        torch.cuda.empty_cache()
        print(f"\n[{model_key}] Released GPU memory")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL REMEDIES COMPLETE")
    print(f"  Completed: {len(completed)}/{len(completed)+len(failed)}")
    print(f"  Elapsed: {elapsed/3600:.1f} hours")
    if completed:
        print(f"  ✓ {', '.join(completed)}")
    if failed:
        print(f"  ✗ FAILED: {', '.join(failed)}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Remedies")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test")
    parser.add_argument("--only-corruption", action="store_true",
                        help="Only run corruption evaluation (skip VQA ablation)")
    parser.add_argument("--only-vqa-ablation", action="store_true",
                        help="Only run VQA ablation (skip corruption eval)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: auto-detect)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] CUDA: {torch.version.cuda}")

    if args.smoke_test:
        smoke_test(device)
    else:
        run_all_remedies(
            device,
            skip_corruption=args.only_vqa_ablation,
            skip_vqa=args.only_corruption,
        )
