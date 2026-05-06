#!/usr/bin/env python3
"""
Phase 3: Qwen3-VL-8B-Instruct Zero-Shot Evaluation
===================================================
Evaluates Qwen3-VL-8B-Instruct on all datasets using generative
zero-shot classification, matching the exact same protocol as BLIP-2
and LLaVA in phase3_vlm_baselines.py.

Datasets:
    cifar10, screwset, imagenet_a, imagenet_val, imagenet_c (skipped), lens

Also runs ScrewSet-S (19 corruptions × 5 severities) evaluation.

Usage:
    python3 eval_qwen_vl.py                       # All datasets + ScrewSet-S
    python3 eval_qwen_vl.py --dataset screwset     # Single dataset
    python3 eval_qwen_vl.py --screwset-s-only      # Only ScrewSet-S
    python3 eval_qwen_vl.py --smoke-test           # Quick check (50 images)
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
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED, ALL_DATASETS, CIFAR10_CLASSES,
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
SCREWSET_S_DIR = REPO_ROOT / "results" / "screwset_s"

MODEL_KEY = "qwen3_vl_8b"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_FAMILY = "Qwen3-VL"
MODEL_PARAMS = "8B"
BACKEND = "qwen3_vl"

# Batch size tuned for RTX 5090 32 GB with Qwen3-VL-8B + capped pixels.
# Keeps high utilization with safe memory headroom.
EVAL_BATCH_SIZE = 12

ALL_CORRUPTIONS = IMAGENET_C_CORRUPTIONS_15 + IMAGENET_C_CORRUPTIONS_EXTRA
SS_DIR = DATA_DIR / "screwset_s"
SPLIT_DIR = DATA_DIR / "screwset_split"


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_qwen_model(device):
    """Load Qwen3-VL-8B-Instruct. Returns (model, processor)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[MODEL] Loading Qwen3-VL: {MODEL_ID} ...")
    # max_pixels caps dynamic resolution to 448×448 (= 256 patches of 28×28).
    # Without this, Qwen2.5-VL encodes raw full-res images as thousands of
    # visual tokens, causing OOM and 800s+/batch on ImageNet.
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=28 * 28 * 4, max_pixels=448 * 448
    )
    # Decoder-only model requires left-padding for correct generation
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {MODEL_KEY}: {n_params:,} params loaded on {device}")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Response Matching (same logic as phase3_vlm_baselines.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _match_response_to_class(response, target_name, class_names_lower, target_idx):
    """Match a generative model's text response to the target class name.

    Uses multiple matching strategies (identical to phase3_vlm_baselines.py):
    1. Exact match
    2. Substring containment (bidirectional)
    3. Word overlap (words > 3 chars)
    4. Fuzzy match (always find best match, no cutoff)
    """
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
# Qwen2.5-VL Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def qwen_evaluate(model, processor, class_names, dataset, device,
                  desc="Qwen3-VL", batch_size=None, class_mapping=None):
    """Qwen3-VL generative zero-shot evaluation. Returns accuracy.

    Uses the same prompt strategy as BLIP-2/LLaVA:
    - ≤40 classes: enumerate all classes in prompt
    - >40 classes: open-ended "What is the main subject?"
    """
    if batch_size is None:
        batch_size = EVAL_BATCH_SIZE

    class_names_lower = [c.lower() for c in class_names]

    # Build prompt (same strategy as BLIP-2/LLaVA)
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

        # Build conversation for each image in the batch
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
            # Process each conversation through the chat template
            prompts = [
                processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]

            # Process images and text together
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

                if _debug_count < 5:
                    target_idx_dbg = class_mapping[label] if class_mapping else label
                    print(f"  [DEBUG Qwen] response={text_lower!r}  "
                          f"target={class_names_lower[target_idx_dbg]!r}")
                    _debug_count += 1

                if class_mapping is not None:
                    target_idx = class_mapping[label]
                else:
                    target_idx = label

                target_name = class_names_lower[target_idx]
                if _match_response_to_class(text_lower, target_name, class_names_lower, target_idx):
                    correct += 1
                total += 1

        except Exception as e:
            print(f"  [WARN] Qwen batch {i} error: {e}")
            total += len(batch_labels)

    return correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Runners (mirroring phase3_vlm_baselines.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cifar10(model, processor, device):
    """CIFAR-10 evaluation."""
    from torchvision import datasets as tv_datasets

    print(f"\n{'='*70}")
    print(f"  CIFAR-10 — {MODEL_KEY} ({MODEL_FAMILY})")
    print(f"{'='*70}")

    CIFAR_ROOT = DATA_DIR / "cifar10"
    class_names = CIFAR10_CLASSES

    test_ds = tv_datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False)
    test_ds_pil = [(test_ds[i][0], test_ds[i][1]) for i in range(len(test_ds))]
    print(f"[INFO] CIFAR-10 test: {len(test_ds_pil)} images, {len(class_names)} classes")

    test_acc = qwen_evaluate(model, processor, class_names, test_ds_pil, device,
                             desc=f"[{MODEL_KEY}] CIFAR-10 Clean")
    print(f"[{MODEL_KEY}] CIFAR-10 Clean Test Acc: {test_acc:.4f}")

    # Skip CIFAR-10-C for generative (same as BLIP-2/LLaVA)
    print(f"[INFO] Skipping CIFAR-10-C for {MODEL_KEY} (generative, too slow)")

    result = {
        "model": MODEL_KEY,
        "model_family": MODEL_FAMILY,
        "backend": BACKEND,
        "dataset": "cifar10",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "test_acc": test_acc,
        "corrupt_results": {},
        "mean_corrupt_acc": None,
        "text_templates": "generative_prompt",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{MODEL_KEY}_cifar10_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    return result


def run_screwset(model, processor, device):
    """ScrewSet evaluation."""
    print(f"\n{'='*70}")
    print(f"  ScrewSet — {MODEL_KEY} ({MODEL_FAMILY})")
    print(f"{'='*70}")

    class_names, folder_names = get_screwset_class_names(SPLIT_DIR)
    print(f"[INFO] ScrewSet classes ({len(class_names)}):")
    for i, (fn, cn) in enumerate(zip(folder_names, class_names)):
        print(f"  {i:2d}. {fn:25s} → \"{cn}\"")

    test_ds_pil = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
    print(f"[INFO] ScrewSet test: {len(test_ds_pil)} images, {len(class_names)} classes")

    test_acc = qwen_evaluate(model, processor, class_names, test_ds_pil, device,
                             desc=f"[{MODEL_KEY}] ScrewSet Clean")
    print(f"[{MODEL_KEY}] ScrewSet Clean Test Acc: {test_acc:.4f}")

    # Skip ScrewSet-C for generative (same as BLIP-2/LLaVA)
    print(f"[INFO] Skipping ScrewSet-C for {MODEL_KEY} (generative, too slow)")

    result = {
        "model": MODEL_KEY,
        "model_family": MODEL_FAMILY,
        "backend": BACKEND,
        "dataset": "screwset",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "class_name_mapping": dict(zip(folder_names, class_names)),
        "test_acc": test_acc,
        "corrupt_results": {},
        "mean_corrupt_acc": None,
        "text_templates": "generative_prompt",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{MODEL_KEY}_screwset_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    return result


def run_imagenet_a(model, processor, device):
    """ImageNet-A evaluation."""
    print(f"\n{'='*70}")
    print(f"  ImageNet-A — {MODEL_KEY} ({MODEL_FAMILY})")
    print(f"{'='*70}")

    IMAGENET_A_DIR = DATA_DIR / "imagenet-a"
    if not IMAGENET_A_DIR.exists():
        print("[ERROR] ImageNet-A directory not found — skipping")
        return None

    class_names = get_imagenet_class_names()
    class_mapping = build_imagenet_a_mapping(str(IMAGENET_A_DIR))
    ds_pil = PILImageFolder(str(IMAGENET_A_DIR), is_valid_file=is_valid_image)
    print(f"[INFO] ImageNet-A: {len(ds_pil)} images, {len(class_mapping)} classes → 1000 class text")

    acc = qwen_evaluate(model, processor, class_names, ds_pil, device,
                        desc=f"[{MODEL_KEY}] ImageNet-A", class_mapping=class_mapping)
    print(f"[{MODEL_KEY}] ImageNet-A Accuracy: {acc:.4f}")

    result = {
        "model": MODEL_KEY,
        "model_family": MODEL_FAMILY,
        "backend": BACKEND,
        "dataset": "imagenet_a",
        "evaluation_mode": "zero_shot",
        "num_imagenet_a_classes": len(class_mapping),
        "imagenet_a_acc": acc,
        "text_templates": "generative_prompt",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{MODEL_KEY}_imagenet_a_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    return result


def run_imagenet_val(model, processor, device):
    """ImageNet Validation evaluation."""
    print(f"\n{'='*70}")
    print(f"  ImageNet Val — {MODEL_KEY} ({MODEL_FAMILY})")
    print(f"{'='*70}")

    IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
    if not IMAGENET_VAL_DIR.exists():
        print("[ERROR] ImageNet val directory not found — skipping")
        return None

    class_names = get_imagenet_class_names()
    ds_pil = PILImageFolder(str(IMAGENET_VAL_DIR), is_valid_file=is_valid_image)
    print(f"[INFO] ImageNet Val: {len(ds_pil)} images, {len(class_names)} classes")

    val_acc = qwen_evaluate(model, processor, class_names, ds_pil, device,
                            desc=f"[{MODEL_KEY}] ImageNet Val")
    print(f"[{MODEL_KEY}] ImageNet Val Accuracy: {val_acc:.4f}")

    result = {
        "model": MODEL_KEY,
        "model_family": MODEL_FAMILY,
        "backend": BACKEND,
        "dataset": "imagenet_val",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "num_images": len(ds_pil),
        "val_acc": val_acc,
        "text_templates": "generative_prompt",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{MODEL_KEY}_imagenet_val_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    return result


def run_imagenet_c(model, processor, device):
    """ImageNet-C — skipped for generative VLMs (same as BLIP-2/LLaVA)."""
    print(f"\n{'='*70}")
    print(f"  ImageNet-C — {MODEL_KEY} (SKIPPED — generative)")
    print(f"{'='*70}")
    print(f"[INFO] Skipping ImageNet-C for {MODEL_KEY} "
          f"(generative zero-shot too slow for 4.75M images)")

    result = {
        "model": MODEL_KEY,
        "model_family": MODEL_FAMILY,
        "backend": BACKEND,
        "dataset": "imagenet_c",
        "evaluation_mode": "zero_shot",
        "skipped": True,
        "skip_reason": f"Generative {MODEL_KEY} too slow for 4.75M images (estimated 130+ hours)",
    }

    out_path = RESULTS_DIR / f"{MODEL_KEY}_imagenet_c_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] (Skipped) Results saved to {out_path}")
    return result


def run_lens(model, processor, device):
    """Lens / ImageNet-ES evaluation."""
    print(f"\n{'='*70}")
    print(f"  Lens / ImageNet-ES — {MODEL_KEY} ({MODEL_FAMILY})")
    print(f"{'='*70}")

    LENS_DIR = DATA_DIR / "lens_split"
    if not LENS_DIR.exists():
        print("[ERROR] Lens split directory not found — skipping")
        return None

    class_names = get_lens_class_names(LENS_DIR)
    test_ds_pil = PILImageFolder(str(LENS_DIR / "test"))
    print(f"[INFO] Lens test: {len(test_ds_pil)} images, {len(class_names)} classes")

    test_acc = qwen_evaluate(model, processor, class_names, test_ds_pil, device,
                             desc=f"[{MODEL_KEY}] Lens Clean")
    print(f"[{MODEL_KEY}] Lens Clean Test Acc: {test_acc:.4f}")

    # Skip Lens corrupted for generative (same as BLIP-2/LLaVA)
    print(f"[INFO] Skipping Lens corrupted for {MODEL_KEY} (generative, too slow)")

    result = {
        "model": MODEL_KEY,
        "model_family": MODEL_FAMILY,
        "backend": BACKEND,
        "dataset": "lens",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "test_acc": test_acc,
        "corrupt_results": {},
        "mean_corrupt_acc": None,
        "text_templates": "generative_prompt",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{MODEL_KEY}_lens_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ScrewSet-S Evaluation (19 corruptions × 5 severities)
# ═══════════════════════════════════════════════════════════════════════════════

def run_screwset_s(model, processor, device):
    """Evaluate Qwen2.5-VL on ScrewSet-S (19 corruptions × 5 severities).

    Same protocol as other generative VLMs: skip (too slow for ~2M images).
    Save stub result consistent with BLIP-2/LLaVA in eval_screwset_s.py.
    """
    print(f"\n{'='*70}")
    print(f"  ScrewSet-S — {MODEL_KEY} ({MODEL_FAMILY})")
    print(f"  NOTE: Generative VLMs are too slow for full corruption eval.")
    print(f"  Skipping ScrewSet-S (same as BLIP-2/LLaVA)")
    print(f"{'='*70}")

    SCREWSET_S_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCREWSET_S_DIR / f"{MODEL_KEY}_screwset_s.json"

    result = {
        "model": MODEL_KEY,
        "phase": 3,
        "model_family": MODEL_FAMILY,
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
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test(device):
    """Quick test: 50 ImageNet-Val images to check the model loads and generates."""
    print(f"\n{'='*70}")
    print(f"  SMOKE TEST — {MODEL_KEY}")
    print(f"{'='*70}")

    IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
    if not IMAGENET_VAL_DIR.exists():
        print("[FATAL] ImageNet-val not found")
        sys.exit(1)

    model, processor = load_qwen_model(device)
    class_names = get_imagenet_class_names()

    full_ds = PILImageFolder(str(IMAGENET_VAL_DIR), is_valid_file=is_valid_image)
    import random
    random.seed(SEED)
    n_smoke = min(50, len(full_ds))
    indices = random.sample(range(len(full_ds)), n_smoke)
    subset = [(full_ds[i][0], full_ds[i][1]) for i in indices]

    start = time.time()
    acc = qwen_evaluate(model, processor, class_names, subset, device,
                        desc="Smoke Test", batch_size=4)
    elapsed = time.time() - start

    print(f"\n  Accuracy: {acc:.4f} ({n_smoke} images)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_smoke:.2f}s/image)")
    print(f"  Status: {'PASS' if acc > 0 else 'FAIL (zero accuracy)'}")

    del model
    torch.cuda.empty_cache()
    return acc > 0


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
    parser = argparse.ArgumentParser(
        description=f"Phase 3: {MODEL_FAMILY} Zero-Shot Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default=None,
                        choices=ALL_DATASETS + ["all"],
                        help="Dataset to evaluate on (default: all)")
    parser.add_argument("--screwset-s-only", action="store_true",
                        help="Only run ScrewSet-S evaluation")
    parser.add_argument("--skip-screwset-s", action="store_true",
                        help="Skip ScrewSet-S evaluation")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 50-image integrity check")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size for evaluation (default: {EVAL_BATCH_SIZE})")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] Model: {MODEL_ID} ({MODEL_PARAMS})")
    print(f"[INFO] Batch size: {EVAL_BATCH_SIZE}")
    print(f"[INFO] Seed: {SEED}")

    set_seed()

    # ── Smoke test ──
    if args.smoke_test:
        success = run_smoke_test(device)
        sys.exit(0 if success else 1)

    # ── ScrewSet-S only ──
    if args.screwset_s_only:
        model, processor = load_qwen_model(device)
        run_screwset_s(model, processor, device)
        return

    # ── Load model once ──
    model, processor = load_qwen_model(device)

    # ── Run datasets ──
    dataset_list = ALL_DATASETS if args.dataset in (None, "all") else [args.dataset]
    failed = []

    for ds_name in dataset_list:
        runner = DATASET_RUNNERS[ds_name]

        # Auto-resume: skip if JSON exists
        json_path = RESULTS_DIR / f"{MODEL_KEY}_{ds_name}_baselines.json"
        if json_path.exists():
            print(f"\n[SKIP] {MODEL_KEY} × {ds_name} — {json_path.name} exists")
            continue

        print(f"\n{'#'*70}")
        print(f"  {MODEL_KEY} × {ds_name}")
        print(f"{'#'*70}")

        set_seed()
        try:
            runner(model, processor, device)
        except Exception as e:
            print(f"[ERROR] {MODEL_KEY} × {ds_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(ds_name)

    # ── ScrewSet-S ──
    if not args.skip_screwset_s:
        screwset_s_path = SCREWSET_S_DIR / f"{MODEL_KEY}_screwset_s.json"
        if screwset_s_path.exists():
            print(f"\n[SKIP] ScrewSet-S — {screwset_s_path.name} exists")
        else:
            try:
                run_screwset_s(model, processor, device)
            except Exception as e:
                print(f"[ERROR] ScrewSet-S failed: {e}")
                failed.append("screwset_s")

    # ── Summary ──
    n_total = len(dataset_list) + (0 if args.skip_screwset_s else 1)
    print(f"\n{'='*70}")
    print(f"  ALL RUNS COMPLETE ({n_total - len(failed)}/{n_total} succeeded)")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  ScrewSet-S: {SCREWSET_S_DIR}")
    print(f"{'='*70}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
