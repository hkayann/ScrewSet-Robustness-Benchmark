#!/usr/bin/env python3
"""
Phase 3: VLM Zero-Shot Baseline Evaluation
============================================
Evaluates Vision-Language Models on all datasets using zero-shot
classification. No fine-tuning — tests out-of-the-box robustness.

Models (CLIP-family via open_clip):
    clip_vit_b32, clip_vit_b16, clip_vit_l14,
    openclip_vit_b16, siglip_vit_b16, eva02_clip_vit_b16

Models (Generative VLMs via transformers):
    blip2        — Salesforce/blip2-opt-2.7b (feature-based zero-shot)
    llava        — llava-hf/llava-1.5-7b-hf  (generative zero-shot, clean only)

Datasets (same 6 as Phase 1 & 2):
    cifar10, screwset, imagenet_a, imagenet_val, imagenet_c, lens

Usage:
    python3 phase3_vlm_baselines.py --model clip_vit_b16 --dataset cifar10
    python3 phase3_vlm_baselines.py --model all --dataset all
    python3 phase3_vlm_baselines.py --smoke-test   # Quick integrity check
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import (
    DATA_DIR, SEED, ALL_DATASETS, CIFAR10_CLASSES,
    IMAGENET_C_CORRUPTIONS_15, IMAGENET_C_CORRUPTIONS_EXTRA,
)
from src.utils import patch_ipv4, set_seed
from src.datasets import is_valid_image, NumpyDataset, PILImageFolder, PILNumpyDataset
from src.imagenet_utils import (
    get_imagenet_class_index, get_imagenet_class_names, build_imagenet_a_mapping,
)
from src.class_names import screwset_folder_to_text, get_screwset_class_names, get_lens_class_names
from src.corruption import discover_imagenet_c_corruptions, find_corruption_leaf_dirs

patch_ipv4()

# Lazy imports (only when needed)
# import open_clip            — loaded in load_clip_model()
# from transformers import ...  — loaded in load_blip2/llava_model()


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = REPO_ROOT / "results" / "phase3"

ALL_MODELS = {
    # ── CLIP-family (via open_clip) ──────────────────────────────
    "clip_vit_b32": {
        "backend": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "family": "CLIP (OpenAI)",
        "params": "151M",
        "expected_imagenet_zs": 0.634,  # published zero-shot IN-1K with 80 templates
    },
    "clip_vit_b16": {
        "backend": "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "family": "CLIP (OpenAI)",
        "params": "150M",
        "expected_imagenet_zs": 0.683,
    },
    "clip_vit_l14": {
        "backend": "open_clip",
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "family": "CLIP (OpenAI)",
        "params": "428M",
        "expected_imagenet_zs": 0.755,
    },
    "openclip_vit_b16": {
        "backend": "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "family": "OpenCLIP (LAION-2B)",
        "params": "150M",
        "expected_imagenet_zs": 0.702,
    },
    "siglip_vit_b16": {
        "backend": "open_clip",
        "model_name": "ViT-B-16-SigLIP",
        "pretrained": "webli",
        "family": "SigLIP (Google)",
        "params": "150M",
        "expected_imagenet_zs": 0.731,
    },
    "eva02_clip_vit_b16": {
        "backend": "open_clip",
        "model_name": "EVA02-B-16",
        "pretrained": "merged2b_s8b_b131k",
        "family": "EVA-02-CLIP",
        "params": "150M",
        "expected_imagenet_zs": 0.747,
    },
    # ── Generative VLMs (via transformers) ───────────────────────
    "blip2": {
        "backend": "blip2",
        "model_id": "Salesforce/blip2-opt-2.7b",
        "family": "BLIP-2",
        "params": "3.7B",
        "expected_imagenet_zs": None,  # generative, not directly comparable
    },
    "llava": {
        "backend": "llava",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "family": "LLaVA-1.5",
        "params": "7B",
        "expected_imagenet_zs": None,  # generative, not directly comparable
        # NOTE: LLaVA is generative — evaluated only on clean test sets,
        # corruption evaluation is skipped (too slow for millions of images).
    },
}

# Eval batch sizes (no gradients → can use large batches)
EVAL_BATCH_SIZES = {
    "clip_vit_b32": 1024,
    "clip_vit_b16": 512,
    "clip_vit_l14": 256,
    "openclip_vit_b16": 512,
    "siglip_vit_b16": 512,
    "eva02_clip_vit_b16": 512,
    "blip2": 64,
    "llava": 4,
}

NUM_WORKERS = 12
PREFETCH_FACTOR = 4
PIN_MEMORY = True


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP 80-Template Ensemble (from original CLIP paper)
# ═══════════════════════════════════════════════════════════════════════════════
OPENAI_IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

# Simplified templates for domain-specific datasets
SIMPLE_TEMPLATES = [
    "a photo of a {}.",
]


def make_eval_loader(dataset, batch_size):
    """Create evaluation DataLoader for tensor datasets."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading & Text Feature Building
# ═══════════════════════════════════════════════════════════════════════════════

def load_clip_model(model_key, device):
    """Load an open_clip model. Returns (model, preprocess, tokenizer)."""
    import open_clip

    cfg = ALL_MODELS[model_key]
    print(f"[MODEL] Loading {cfg['model_name']} ({cfg['pretrained']}) via open_clip...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=cfg["pretrained"]
    )
    tokenizer = open_clip.get_tokenizer(cfg["model_name"])
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {model_key}: {n_params:,} params loaded on {device}")
    return model, preprocess, tokenizer


def _tokenize_for_clip(tokenizer, texts, device):
    """Tokenize texts for open_clip, handling SigLIP's T5-based tokenizer.

    SigLIP uses a T5 tokenizer via HFTokenizer wrapper. In transformers >=5.x,
    the slow T5Tokenizer removed batch_encode_plus, breaking open_clip's
    HFTokenizer.__call__. We work around by calling the underlying HF tokenizer
    directly and extracting input_ids.
    """
    try:
        tokens = tokenizer(texts)
        if isinstance(tokens, dict):
            return tokens["input_ids"].to(device)
        return tokens.to(device)
    except AttributeError:
        # Fallback for SigLIP / T5Tokenizer missing batch_encode_plus
        hf_tok = getattr(tokenizer, "tokenizer", tokenizer)
        tok_out = hf_tok(
            texts, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )
        return tok_out["input_ids"].to(device)


def build_clip_text_features(model, tokenizer, class_names, templates, device):
    """Build text feature matrix using template ensembling.

    Returns: text_features (num_classes, embed_dim), normalized.
    """
    print(f"[TEXT] Building text features: {len(class_names)} classes × {len(templates)} templates")
    all_features = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for template in tqdm(templates, desc="Templates", leave=False):
            texts = [template.format(c) for c in class_names]
            tokens = _tokenize_for_clip(tokenizer, texts, device)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
            all_features.append(feats)

    # Average over templates, re-normalize
    text_features = torch.stack(all_features).mean(dim=0)
    text_features = F.normalize(text_features, dim=-1)
    print(f"[TEXT] Text features shape: {text_features.shape}")
    return text_features


def load_blip2_model(device):
    """Load BLIP-2 model for generative zero-shot. Returns (model, processor)."""
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    model_id = ALL_MODELS["blip2"]["model_id"]
    print(f"[MODEL] Loading BLIP-2: {model_id} ...")
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] blip2: {n_params:,} params loaded on {device}")
    return model, processor


def _extract_tensor(output):
    """Extract tensor from model output (handles transformers 5.x return types)."""
    if isinstance(output, torch.Tensor):
        return output
    # transformers 5.x may return BaseModelOutputWithPooling or similar
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None and isinstance(val, torch.Tensor):
            return val
    # Fallback: first positional element
    return output[0]


def build_blip2_text_features(model, processor, class_names, device):
    """Build text features for BLIP-2 using ITC projection.

    Returns: text_features (num_classes, embed_dim), normalized.
    """
    print(f"[TEXT] Building BLIP-2 text features: {len(class_names)} classes")
    all_feats = []

    with torch.no_grad():
        # Process in batches of 64 to avoid OOM on text side
        batch_size = 64
        for i in tqdm(range(0, len(class_names), batch_size), desc="BLIP-2 text", leave=False):
            batch_names = class_names[i:i + batch_size]
            texts = [f"a photo of a {c}" for c in batch_names]
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            feats = model.get_text_features(**inputs)
            feats = _extract_tensor(feats)
            # feats may be (B, D) or (B, seq_len, D)
            if feats.dim() == 3:
                feats = feats[:, 0, :]  # CLS token
            all_feats.append(feats.float())

    text_features = torch.cat(all_feats, dim=0)
    text_features = F.normalize(text_features, dim=-1)
    print(f"[TEXT] BLIP-2 text features shape: {text_features.shape}")
    return text_features


class BLIP2Transform:
    """Wraps BLIP-2 image processor as a torchvision-compatible transform."""

    def __init__(self, processor):
        self.image_processor = processor.image_processor

    def __call__(self, pil_image):
        result = self.image_processor(images=pil_image, return_tensors="pt")
        return result["pixel_values"].squeeze(0)


def load_llava_model(device):
    """Load LLaVA-1.5-7B. Returns (model, processor)."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model_id = ALL_MODELS["llava"]["model_id"]
    print(f"[MODEL] Loading LLaVA: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] llava: {n_params:,} params loaded on {device}")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Zero-Shot Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def clip_evaluate(model, loader, text_features, device, desc="Evaluating"):
    """CLIP zero-shot evaluation. Returns accuracy."""
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


def clip_evaluate_imagenet_a(model, loader, text_features, class_mapping, device,
                              desc="ImageNet-A"):
    """CLIP zero-shot on ImageNet-A with class mapping to 1000-class text features."""
    correct, total = 0, 0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            # Full 1000-class logits
            logits = image_features @ text_features.T  # (B, 1000)
            preds = logits.argmax(dim=-1)  # predicted 1K index

            # Map folder targets (0-199) to ImageNet-1K indices
            mapped = torch.tensor(
                [class_mapping[t.item()] for t in targets],
                device=device, dtype=torch.long,
            )
            correct += preds.eq(mapped).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def blip2_evaluate(model, loader, text_features, device, desc="Evaluating"):
    """BLIP-2 feature-based zero-shot evaluation. Returns accuracy."""
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = _extract_tensor(
                model.get_image_features(pixel_values=images.half())
            )
            # image_features: (B, num_query_tokens, D)
            image_features = image_features.float()

            # Compute similarity: max over query tokens
            # image_features: (B, Q, D), text_features: (C, D)
            if image_features.dim() == 3:
                sim = torch.einsum("bqd,cd->bqc", F.normalize(image_features, dim=-1),
                                   text_features)
                logits = sim.max(dim=1).values
            else:
                sim = F.normalize(image_features, dim=-1) @ text_features.T
                logits = sim

            preds = logits.argmax(dim=-1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def blip2_evaluate_imagenet_a(model, loader, text_features, class_mapping, device,
                               desc="ImageNet-A"):
    """BLIP-2 zero-shot on ImageNet-A with class mapping."""
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = _extract_tensor(
                model.get_image_features(pixel_values=images.half())
            )
            image_features = image_features.float()

            if image_features.dim() == 3:
                sim = torch.einsum("bqd,cd->bqc", F.normalize(image_features, dim=-1),
                                   text_features)
                logits = sim.max(dim=1).values
            else:
                logits = F.normalize(image_features, dim=-1) @ text_features.T
            preds = logits.argmax(dim=-1)

            mapped = torch.tensor(
                [class_mapping[t.item()] for t in targets],
                device=device, dtype=torch.long,
            )
            correct += preds.eq(mapped).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def _match_response_to_class(response, target_name, class_names_lower, target_idx):
    """Match a generative model's text response to the target class name.

    Uses multiple matching strategies:
    1. Exact match
    2. Substring containment (bidirectional)
    3. Word overlap (words > 3 chars)
    4. Fuzzy match (always find best match, no cutoff)
    """
    from difflib import SequenceMatcher

    # 1. Exact match
    if response == target_name:
        return True
    # 2. Target contained in response or vice versa
    if target_name in response or response in target_name:
        return True
    # 3. Word overlap (meaningful words > 3 chars)
    response_words = set(response.split())
    target_words = set(target_name.split())
    if any(w in response_words for w in target_words if len(w) > 3):
        return True
    # 4. Best fuzzy match among ALL class names (no cutoff)
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


def llava_evaluate(model, processor, class_names, dataset, device, desc="LLaVA",
                   batch_size=4, class_mapping=None):
    """LLaVA generative zero-shot evaluation. Returns accuracy.

    For each image, generates a text description and matches to closest class name.
    class_mapping: if provided, maps dataset class indices to class_names indices.
    """
    from difflib import get_close_matches

    # Build class name lookup (lowercase)
    class_names_lower = [c.lower() for c in class_names]

    # Build prompt using processor's chat template (transformers 5.x compatible)
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

    # Use apply_chat_template if available (transformers >=5.x)
    _use_chat_template = hasattr(processor, "apply_chat_template")
    if _use_chat_template:
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]},
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(f"  [LLaVA] Using chat template: {prompt_text[:80]}...")
    else:
        # Legacy format (transformers <5.x)
        prompt_text = f"USER: <image>\n{user_text}\nASSISTANT:"
        print(f"  [LLaVA] Using legacy prompt format")

    correct, total = 0, 0
    n = len(dataset)
    _debug_count = 0  # Print first few responses for debugging

    for i in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, n)
        batch_images = []
        batch_labels = []
        for j in range(i, batch_end):
            img, label = dataset[j]
            batch_images.append(img)
            batch_labels.append(label)

        prompts = [prompt_text] * len(batch_images)

        try:
            inputs = processor(
                text=prompts, images=batch_images,
                return_tensors="pt", padding=True
            ).to(device)

            with torch.no_grad():
                # Only decode NEW tokens (exclude input)
                input_len = inputs["input_ids"].shape[-1]
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )

            # Decode only the generated tokens
            generated_ids = outputs[:, input_len:]
            generated = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for text, label in zip(generated, batch_labels):
                text_lower = text.lower().strip().rstrip(".")

                # Debug: print first few responses
                if _debug_count < 5:
                    target_idx_dbg = class_mapping[label] if class_mapping else label
                    print(f"  [DEBUG] response={text_lower!r}  target={class_names_lower[target_idx_dbg]!r}")
                    _debug_count += 1

                # Determine target class index
                if class_mapping is not None:
                    target_idx = class_mapping[label]
                else:
                    target_idx = label

                target_name = class_names_lower[target_idx]

                # Match using the shared matching logic
                if _match_response_to_class(text_lower, target_name, class_names_lower, target_idx):
                    correct += 1
                total += 1

        except Exception as e:
            print(f"  [WARN] LLaVA batch {i} error: {e}")
            total += len(batch_labels)

    return correct / total if total else 0.0


def blip2_generative_evaluate(model, processor, class_names, dataset, device,
                               desc="BLIP-2", batch_size=16, class_mapping=None):
    """BLIP-2 generative zero-shot evaluation using VQA-style prompting.

    Uses the same matching logic as LLaVA but with BLIP-2's prompt format.
    dataset: list of (PIL_image, label) tuples.
    """
    from difflib import get_close_matches, SequenceMatcher

    class_names_lower = [c.lower() for c in class_names]

    if len(class_names) <= 40:
        class_list = ", ".join(class_names)
        prompt = f"Question: Classify this image. The categories are: {class_list}. Answer with only the category name. Answer:"
    else:
        prompt = "Question: What is the specific type or species of the main subject in this image? Answer with a precise name. Answer:"

    correct, total = 0, 0
    _debug_count = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
        batch_end = min(i + batch_size, len(dataset))
        batch_images = []
        batch_labels = []
        for j in range(i, batch_end):
            img, label = dataset[j]
            batch_images.append(img)
            batch_labels.append(label)

        try:
            inputs = processor(
                images=batch_images,
                text=[prompt] * len(batch_images),
                return_tensors="pt",
                padding=True,
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
                    print(f"  [DEBUG BLIP2] response={text_lower!r}  target={class_names_lower[target_idx_dbg]!r}")
                    _debug_count += 1

                if class_mapping is not None:
                    target_idx = class_mapping[label]
                else:
                    target_idx = label

                target_name = class_names_lower[target_idx]

                # Match using the shared matching logic
                if _match_response_to_class(text_lower, target_name, class_names_lower, target_idx):
                    correct += 1
                total += 1

        except Exception as e:
            print(f"  [WARN] BLIP-2 batch {i} error: {e}")
            total += len(batch_labels)

    return correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Evaluation Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_loader(model_key, model, loader, text_features, device,
                       desc="Eval", class_mapping=None,
                       dataset_pil=None, processor=None, class_names=None):
    """Dispatch evaluation to the right backend function."""
    backend = ALL_MODELS[model_key]["backend"]

    if backend == "open_clip":
        if class_mapping is not None:
            return clip_evaluate_imagenet_a(model, loader, text_features, class_mapping,
                                           device, desc=desc)
        return clip_evaluate(model, loader, text_features, device, desc=desc)

    elif backend == "blip2":
        if dataset_pil is None:
            raise ValueError("BLIP-2 (generative) requires dataset_pil (PIL image dataset)")
        return blip2_generative_evaluate(model, processor, class_names, dataset_pil, device,
                                         desc=desc, batch_size=EVAL_BATCH_SIZES.get("blip2", 16),
                                         class_mapping=class_mapping)

    elif backend == "llava":
        if dataset_pil is None:
            raise ValueError("LLaVA requires dataset_pil (PIL image dataset)")
        return llava_evaluate(model, processor, class_names, dataset_pil, device,
                              desc=desc, batch_size=EVAL_BATCH_SIZES.get(model_key, 4),
                              class_mapping=class_mapping)

    raise ValueError(f"Unknown backend: {backend}")


# ═══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_cifar10(model_key, device):
    print(f"\n{'='*70}")
    print(f"  CIFAR-10 — {model_key} ({ALL_MODELS[model_key]['family']})")
    print(f"{'='*70}")

    cfg = ALL_MODELS[model_key]
    backend = cfg["backend"]
    CIFAR_ROOT = DATA_DIR / "cifar10"
    CIFAR_C_DIR = DATA_DIR / "CIFAR-10-C"
    batch_size = EVAL_BATCH_SIZES.get(model_key, 256)
    class_names = CIFAR10_CLASSES
    templates = OPENAI_IMAGENET_TEMPLATES

    # ── Load model & build classifier ──
    if backend == "open_clip":
        model, preprocess, tokenizer = load_clip_model(model_key, device)
        text_features = build_clip_text_features(model, tokenizer, class_names, templates, device)

        test_ds = datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False, transform=preprocess)
        test_loader = make_eval_loader(test_ds, batch_size)

    elif backend == "blip2":
        model, processor = load_blip2_model(device)
        text_features = None
        preprocess = None

        # BLIP-2 needs PIL images — use raw CIFAR-10
        test_ds_tensor = datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False)
        test_ds_pil = []
        for i in range(len(test_ds_tensor)):
            img, label = test_ds_tensor[i]
            test_ds_pil.append((img, label))
        test_loader = None

    elif backend == "llava":
        model, processor = load_llava_model(device)
        text_features = None
        preprocess = None

        # LLaVA needs PIL images — use raw CIFAR-10
        test_ds_tensor = datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False)
        test_ds_pil = []
        for i in range(len(test_ds_tensor)):
            img, label = test_ds_tensor[i]
            test_ds_pil.append((img, label))  # CIFAR-10 returns PIL by default when no transform
        test_loader = None

    print(f"[INFO] CIFAR-10 test: {len(test_ds) if backend not in ('llava', 'blip2') else len(test_ds_pil)} images, "
          f"{len(class_names)} classes")

    # ── Evaluate clean test ──
    model_tag = f"{model_key}_cifar10"
    if backend == "llava":
        test_acc = llava_evaluate(model, processor, class_names, test_ds_pil, device,
                                  desc=f"[{model_tag}] Clean Test", batch_size=batch_size)
    elif backend == "blip2":
        test_acc = blip2_generative_evaluate(model, processor, class_names, test_ds_pil, device,
                                              desc=f"[{model_tag}] Clean Test", batch_size=batch_size)
    else:
        test_acc = evaluate_on_loader(model_key, model, test_loader, text_features, device,
                                      desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test Acc: {test_acc:.4f}")

    # ── Evaluate CIFAR-10-C ──
    corrupt_results = {}
    if backend in ("llava", "blip2"):
        print(f"[INFO] Skipping CIFAR-10-C for {model_key} (generative, too slow)")
    elif CIFAR_C_DIR.exists():
        corr_labels = np.load(str(CIFAR_C_DIR / "labels.npy"))
        for fname in sorted(os.listdir(CIFAR_C_DIR)):
            if not fname.endswith(".npy") or fname == "labels.npy":
                continue
            images = np.load(str(CIFAR_C_DIR / fname))
            ds = NumpyDataset(images, corr_labels,
                              preprocess if backend == "open_clip" else BLIP2Transform(processor))
            loader = make_eval_loader(ds, batch_size)
            acc = evaluate_on_loader(model_key, model, loader, text_features, device,
                                     desc=f"[{model_tag}] C-{fname}")
            cname = fname.replace(".npy", "")
            corrupt_results[cname] = acc
            print(f"  CIFAR-10-C {cname}: {acc:.4f}")
    else:
        print("[WARN] CIFAR-10-C directory not found")

    # ── Save results ──
    mean_corrupt = (sum(corrupt_results.values()) / len(corrupt_results)
                    if corrupt_results else None)
    result = {
        "model": model_key,
        "model_family": cfg["family"],
        "backend": backend,
        "dataset": "cifar10",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "mean_corrupt_acc": mean_corrupt,
        "text_templates": "ensemble_80",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_key}_cifar10_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ScrewSet Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_screwset(model_key, device):
    print(f"\n{'='*70}")
    print(f"  ScrewSet — {model_key} ({ALL_MODELS[model_key]['family']})")
    print(f"{'='*70}")

    cfg = ALL_MODELS[model_key]
    backend = cfg["backend"]
    SPLIT_DIR = DATA_DIR / "screwset_split"
    CORRUPT_ROOT = DATA_DIR / "screwset_c"
    batch_size = EVAL_BATCH_SIZES.get(model_key, 256)

    class_names, folder_names = get_screwset_class_names(SPLIT_DIR)
    templates = SIMPLE_TEMPLATES  # Domain-specific, use simple prompts

    print(f"[INFO] ScrewSet classes ({len(class_names)}):")
    for i, (fn, cn) in enumerate(zip(folder_names, class_names)):
        print(f"  {i:2d}. {fn:25s} → \"{cn}\"")

    # ── Load model & build classifier ──
    if backend == "open_clip":
        model, preprocess, tokenizer = load_clip_model(model_key, device)
        text_features = build_clip_text_features(model, tokenizer, class_names, templates, device)

        test_ds = ImageFolder(str(SPLIT_DIR / "test"), transform=preprocess,
                              is_valid_file=is_valid_image)
        test_loader = make_eval_loader(test_ds, batch_size)

    elif backend == "blip2":
        model, processor = load_blip2_model(device)
        text_features = None
        preprocess = None

        test_ds_pil = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
        test_loader = None

    elif backend == "llava":
        model, processor = load_llava_model(device)
        text_features = None
        preprocess = None

        test_ds_pil = PILImageFolder(str(SPLIT_DIR / "test"), is_valid_file=is_valid_image)
        test_loader = None

    n_test = len(test_ds) if backend not in ('llava', 'blip2') else len(test_ds_pil)
    print(f"[INFO] ScrewSet test: {n_test} images, {len(class_names)} classes")

    # ── Evaluate clean test ──
    model_tag = f"{model_key}_screwset"
    if backend == "llava":
        test_acc = llava_evaluate(model, processor, class_names, test_ds_pil, device,
                                  desc=f"[{model_tag}] Clean Test", batch_size=batch_size)
    elif backend == "blip2":
        test_acc = blip2_generative_evaluate(model, processor, class_names, test_ds_pil, device,
                                              desc=f"[{model_tag}] Clean Test", batch_size=batch_size)
    else:
        test_acc = evaluate_on_loader(model_key, model, test_loader, text_features, device,
                                      desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test Acc: {test_acc:.4f}")

    # ── Evaluate ScrewSet-C ──
    corrupt_results = {}
    if backend in ("llava", "blip2"):
        print(f"[INFO] Skipping ScrewSet-C for {model_key} (generative, too slow)")
    elif CORRUPT_ROOT.exists():
        for corrupt_type in sorted(os.listdir(CORRUPT_ROOT)):
            corrupt_dir = CORRUPT_ROOT / corrupt_type
            if not corrupt_dir.is_dir():
                continue
            try:
                ds = ImageFolder(str(corrupt_dir),
                                 transform=preprocess if backend == "open_clip" else BLIP2Transform(processor),
                                 is_valid_file=is_valid_image)
                loader = make_eval_loader(ds, batch_size)
                acc = evaluate_on_loader(model_key, model, loader, text_features, device,
                                         desc=f"[{model_tag}] {corrupt_type}")
                corrupt_results[corrupt_type] = acc
                print(f"  ScrewSet-C {corrupt_type}: {acc:.4f}")
            except Exception as e:
                print(f"  [ERROR] {corrupt_type}: {e}")
                corrupt_results[corrupt_type] = None
    else:
        print("[WARN] ScrewSet-C directory not found")

    # ── Save results ──
    valid_accs = [v for v in corrupt_results.values() if v is not None]
    mean_corrupt = sum(valid_accs) / len(valid_accs) if valid_accs else None
    result = {
        "model": model_key,
        "model_family": cfg["family"],
        "backend": backend,
        "dataset": "screwset",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "class_name_mapping": dict(zip(folder_names, class_names)),
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "mean_corrupt_acc": mean_corrupt,
        "text_templates": "simple_1",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_key}_screwset_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet-A Pipeline (eval-only, 200 classes mapped to 1000-class text features)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_a(model_key, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet-A — {model_key} ({ALL_MODELS[model_key]['family']})")
    print(f"{'='*70}")

    cfg = ALL_MODELS[model_key]
    backend = cfg["backend"]
    IMAGENET_A_DIR = DATA_DIR / "imagenet-a"
    if not IMAGENET_A_DIR.exists():
        print("[ERROR] ImageNet-A directory not found — skipping")
        return None

    batch_size = EVAL_BATCH_SIZES.get(model_key, 256)
    class_names = get_imagenet_class_names()  # All 1000 ImageNet classes
    templates = OPENAI_IMAGENET_TEMPLATES
    class_mapping = build_imagenet_a_mapping(str(IMAGENET_A_DIR))

    # ── Load model ──
    if backend == "open_clip":
        model, preprocess, tokenizer = load_clip_model(model_key, device)
        text_features = build_clip_text_features(model, tokenizer, class_names, templates, device)

        ds = ImageFolder(str(IMAGENET_A_DIR), transform=preprocess, is_valid_file=is_valid_image)
        loader = make_eval_loader(ds, batch_size)

    elif backend == "blip2":
        model, processor = load_blip2_model(device)
        text_features = None

        ds_pil = PILImageFolder(str(IMAGENET_A_DIR), is_valid_file=is_valid_image)
        loader = None

    elif backend == "llava":
        model, processor = load_llava_model(device)
        text_features = None

        ds_pil = PILImageFolder(str(IMAGENET_A_DIR), is_valid_file=is_valid_image)
        loader = None

    n = len(ds) if backend not in ('llava', 'blip2') else len(ds_pil)
    print(f"[INFO] ImageNet-A: {n} images, {len(class_mapping)} classes → 1000 class text")

    # ── Evaluate ──
    model_tag = f"{model_key}_imagenet_a"
    if backend == "llava":
        acc = llava_evaluate(model, processor, class_names, ds_pil, device,
                             desc=f"[{model_tag}] Evaluating", batch_size=batch_size,
                             class_mapping=class_mapping)
    elif backend == "blip2":
        acc = blip2_generative_evaluate(model, processor, class_names, ds_pil, device,
                                        desc=f"[{model_tag}] Evaluating", batch_size=batch_size,
                                        class_mapping=class_mapping)
    else:
        acc = evaluate_on_loader(model_key, model, loader, text_features, device,
                                 desc=f"[{model_tag}] Evaluating", class_mapping=class_mapping)
    print(f"[{model_tag}] ImageNet-A Accuracy: {acc:.4f}")

    result = {
        "model": model_key,
        "model_family": cfg["family"],
        "backend": backend,
        "dataset": "imagenet_a",
        "evaluation_mode": "zero_shot",
        "num_imagenet_a_classes": len(class_mapping),
        "imagenet_a_acc": acc,
        "text_templates": "ensemble_80",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_key}_imagenet_a_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet Validation Pipeline (eval-only, 1000-class clean)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_val(model_key, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet Val — {model_key} ({ALL_MODELS[model_key]['family']})")
    print(f"{'='*70}")

    cfg = ALL_MODELS[model_key]
    backend = cfg["backend"]
    IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
    if not IMAGENET_VAL_DIR.exists():
        print("[ERROR] ImageNet val directory not found — skipping")
        return None

    batch_size = EVAL_BATCH_SIZES.get(model_key, 256)
    class_names = get_imagenet_class_names()
    templates = OPENAI_IMAGENET_TEMPLATES

    # ── Load model ──
    if backend == "open_clip":
        model, preprocess, tokenizer = load_clip_model(model_key, device)
        text_features = build_clip_text_features(model, tokenizer, class_names, templates, device)

        ds = ImageFolder(str(IMAGENET_VAL_DIR), transform=preprocess, is_valid_file=is_valid_image)
        loader = make_eval_loader(ds, batch_size)

    elif backend == "blip2":
        model, processor = load_blip2_model(device)
        text_features = None

        ds_pil = PILImageFolder(str(IMAGENET_VAL_DIR), is_valid_file=is_valid_image)
        loader = None

    elif backend == "llava":
        model, processor = load_llava_model(device)
        text_features = None

        ds_pil = PILImageFolder(str(IMAGENET_VAL_DIR), is_valid_file=is_valid_image)
        loader = None

    n = len(ds) if backend not in ('llava', 'blip2') else len(ds_pil)
    print(f"[INFO] ImageNet Val: {n} images, {len(class_names)} classes")

    # ── Evaluate ──
    model_tag = f"{model_key}_imagenet_val"
    if backend == "llava":
        val_acc = llava_evaluate(model, processor, class_names, ds_pil, device,
                                 desc=f"[{model_tag}] Evaluating", batch_size=batch_size)
    elif backend == "blip2":
        val_acc = blip2_generative_evaluate(model, processor, class_names, ds_pil, device,
                                             desc=f"[{model_tag}] Evaluating", batch_size=batch_size)
    else:
        val_acc = evaluate_on_loader(model_key, model, loader, text_features, device,
                                     desc=f"[{model_tag}] Evaluating")
    print(f"[{model_tag}] ImageNet Val Accuracy: {val_acc:.4f}")

    result = {
        "model": model_key,
        "model_family": cfg["family"],
        "backend": backend,
        "dataset": "imagenet_val",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "num_images": n,
        "val_acc": val_acc,
        "text_templates": "ensemble_80",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_key}_imagenet_val_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet-C Pipeline (eval-only, 19 corruptions × 5 severities)
# ═══════════════════════════════════════════════════════════════════════════════

def run_imagenet_c(model_key, device):
    print(f"\n{'='*70}")
    print(f"  ImageNet-C — {model_key} ({ALL_MODELS[model_key]['family']})")
    print(f"{'='*70}")

    cfg = ALL_MODELS[model_key]
    backend = cfg["backend"]

    if backend in ("llava", "blip2"):
        print(f"[INFO] Skipping ImageNet-C for {model_key} (generative zero-shot too slow for 4.75M images)")
        # Save a stub result
        result = {
            "model": model_key,
            "model_family": cfg["family"],
            "backend": backend,
            "dataset": "imagenet_c",
            "evaluation_mode": "zero_shot",
            "skipped": True,
            "skip_reason": f"Generative {model_key} too slow for 4.75M images (estimated 130+ hours)",
        }
        out_path = RESULTS_DIR / f"{model_key}_imagenet_c_baselines.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)
        return result

    IMAGENET_C_DIR = DATA_DIR / "imagenet-c"
    if not IMAGENET_C_DIR.exists():
        print("[ERROR] ImageNet-C directory not found — skipping")
        return None

    corruptions = discover_imagenet_c_corruptions(IMAGENET_C_DIR)
    if not corruptions:
        print("[ERROR] No corrupt dirs found in ImageNet-C — skipping")
        return None

    print(f"[INFO] Found {len(corruptions)} corruptions: {list(corruptions.keys())}")

    batch_size = EVAL_BATCH_SIZES.get(model_key, 256)
    class_names = get_imagenet_class_names()
    templates = OPENAI_IMAGENET_TEMPLATES

    # ── Load model ──
    if backend == "open_clip":
        model, preprocess, tokenizer = load_clip_model(model_key, device)
        text_features = build_clip_text_features(model, tokenizer, class_names, templates, device)
    elif backend == "blip2":
        model, processor = load_blip2_model(device)
        preprocess = BLIP2Transform(processor)
        text_features = build_blip2_text_features(model, processor, class_names, device)

    model_tag = f"{model_key}_imagenet_c"
    corruption_results = {}
    severity_levels = [1, 2, 3, 4, 5]

    for cname, cpath in sorted(corruptions.items()):
        corruption_results[cname] = {}
        for sev in severity_levels:
            sev_dir = cpath / str(sev)
            if not sev_dir.exists():
                print(f"  [WARN] {cname}/severity-{sev} not found, skipping")
                continue

            ds = ImageFolder(str(sev_dir),
                             transform=preprocess if backend == "open_clip" else BLIP2Transform(processor),
                             is_valid_file=is_valid_image)
            loader = make_eval_loader(ds, batch_size)

            acc = evaluate_on_loader(model_key, model, loader, text_features, device,
                                     desc=f"[{model_tag}] {cname}/s{sev}")
            corruption_results[cname][str(sev)] = acc
            print(f"  {cname} sev-{sev}: {acc:.4f}")

        accs = [v for v in corruption_results[cname].values() if v is not None]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        corruption_results[cname]["mean"] = mean_acc
        print(f"  {cname} mean: {mean_acc:.4f}")

    # ── Aggregate metrics ──
    std15_accs = [corruption_results[c]["mean"] for c in IMAGENET_C_CORRUPTIONS_15
                  if c in corruption_results and "mean" in corruption_results[c]]
    mean_acc_15 = sum(std15_accs) / len(std15_accs) if std15_accs else 0.0
    mce_15 = 1.0 - mean_acc_15

    all_accs = [cr["mean"] for cr in corruption_results.values()
                if "mean" in cr and cr["mean"] is not None]
    mean_acc_all = sum(all_accs) / len(all_accs) if all_accs else 0.0
    mce_all = 1.0 - mean_acc_all

    print(f"\n[{model_tag}] Mean Acc (15 std): {mean_acc_15:.4f}  mCE: {mce_15:.4f}")
    print(f"[{model_tag}] Mean Acc (all {len(all_accs)}): {mean_acc_all:.4f}  mCE: {mce_all:.4f}")

    result = {
        "model": model_key,
        "model_family": cfg["family"],
        "backend": backend,
        "dataset": "imagenet_c",
        "evaluation_mode": "zero_shot",
        "num_corruptions_evaluated": len(corruption_results),
        "corruption_results": corruption_results,
        "mean_acc_15_std": mean_acc_15,
        "mce_15_std": mce_15,
        "mean_acc_all": mean_acc_all,
        "mce_all": mce_all,
        "text_templates": "ensemble_80",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_key}_imagenet_c_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Lens / ImageNet-ES Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_lens(model_key, device):
    print(f"\n{'='*70}")
    print(f"  Lens / ImageNet-ES — {model_key} ({ALL_MODELS[model_key]['family']})")
    print(f"{'='*70}")

    cfg = ALL_MODELS[model_key]
    backend = cfg["backend"]
    LENS_DIR = DATA_DIR / "lens_split"
    if not LENS_DIR.exists():
        print("[ERROR] Lens split directory not found — skipping")
        return None

    batch_size = EVAL_BATCH_SIZES.get(model_key, 256)
    class_names = get_lens_class_names(LENS_DIR)
    templates = OPENAI_IMAGENET_TEMPLATES  # Lens classes are ImageNet classes

    # ── Load model ──
    if backend == "open_clip":
        model, preprocess, tokenizer = load_clip_model(model_key, device)
        text_features = build_clip_text_features(model, tokenizer, class_names, templates, device)

        test_ds = ImageFolder(str(LENS_DIR / "test"), transform=preprocess)
        test_loader = make_eval_loader(test_ds, batch_size)

    elif backend == "blip2":
        model, processor = load_blip2_model(device)
        text_features = None
        preprocess = None

        test_ds_pil = PILImageFolder(str(LENS_DIR / "test"))
        test_loader = None

    elif backend == "llava":
        model, processor = load_llava_model(device)
        text_features = None
        preprocess = None

        test_ds_pil = PILImageFolder(str(LENS_DIR / "test"))
        test_loader = None

    n = len(test_ds) if backend not in ('llava', 'blip2') else len(test_ds_pil)
    print(f"[INFO] Lens test: {n} images, {len(class_names)} classes")

    # ── Evaluate clean test ──
    model_tag = f"{model_key}_lens"
    if backend == "llava":
        test_acc = llava_evaluate(model, processor, class_names, test_ds_pil, device,
                                  desc=f"[{model_tag}] Clean Test", batch_size=batch_size)
    elif backend == "blip2":
        test_acc = blip2_generative_evaluate(model, processor, class_names, test_ds_pil, device,
                                              desc=f"[{model_tag}] Clean Test", batch_size=batch_size)
    else:
        test_acc = evaluate_on_loader(model_key, model, test_loader, text_features, device,
                                      desc=f"[{model_tag}] Clean Test")
    print(f"[{model_tag}] Clean Test Acc: {test_acc:.4f}")

    # ── Evaluate corrupted subsets ──
    corrupt_results = {}
    corrupt_root = LENS_DIR / "corrupted"
    if backend in ("llava", "blip2"):
        print(f"[INFO] Skipping Lens corrupted for {model_key} (generative, too slow)")
    elif corrupt_root.exists():
        leaf_dirs = find_corruption_leaf_dirs(corrupt_root)
        print(f"[INFO] Found {len(leaf_dirs)} corrupted subsets")
        for leaf in leaf_dirs:
            rel = str(leaf.relative_to(corrupt_root))
            try:
                if backend == "open_clip":
                    ds = ImageFolder(str(leaf), transform=preprocess)
                else:
                    ds = ImageFolder(str(leaf), transform=BLIP2Transform(processor))

                loader = make_eval_loader(ds, batch_size)
                acc = evaluate_on_loader(model_key, model, loader, text_features, device,
                                         desc=f"[{model_tag}] {rel}")
                corrupt_results[rel] = acc
                print(f"  Lens corrupt {rel}: {acc:.4f}")
            except Exception as e:
                print(f"  [ERROR] {rel}: {e}")
                corrupt_results[rel] = None
    else:
        print("[WARN] Lens corrupted directory not found")

    # ── Save results ──
    valid_accs = [v for v in corrupt_results.values() if v is not None]
    mean_corrupt = sum(valid_accs) / len(valid_accs) if valid_accs else None
    result = {
        "model": model_key,
        "model_family": cfg["family"],
        "backend": backend,
        "dataset": "lens",
        "evaluation_mode": "zero_shot",
        "num_classes": len(class_names),
        "test_acc": test_acc,
        "corrupt_results": corrupt_results,
        "mean_corrupt_acc": mean_corrupt,
        "text_templates": "ensemble_80",
        "seed": SEED,
    }

    out_path = RESULTS_DIR / f"{model_key}_lens_baselines.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[DONE] Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke Test — Integrity Check for All Models
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test(device):
    """Quick integrity check: load each model, run on small ImageNet-Val subset,
    verify accuracy is in expected range."""

    print("\n" + "=" * 70)
    print("  SMOKE TEST — Phase 3 VLM Integrity Check")
    print("=" * 70)

    IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
    if not IMAGENET_VAL_DIR.exists():
        print("[FATAL] ImageNet-val not found, required for smoke test")
        sys.exit(1)

    class_names = get_imagenet_class_names()
    templates = OPENAI_IMAGENET_TEMPLATES
    N_SMOKE = 200  # Number of images for quick check

    results = {}

    for model_key, cfg in ALL_MODELS.items():
        print(f"\n{'─'*50}")
        print(f"  Smoke: {model_key} ({cfg['family']})")
        print(f"{'─'*50}")
        backend = cfg["backend"]

        try:
            start = time.time()

            if backend == "open_clip":
                model, preprocess, tokenizer = load_clip_model(model_key, device)
                text_features = build_clip_text_features(model, tokenizer, class_names,
                                                         templates, device)

                full_ds = ImageFolder(str(IMAGENET_VAL_DIR), transform=preprocess,
                                      is_valid_file=is_valid_image)
                # Take first N_SMOKE images
                subset = torch.utils.data.Subset(full_ds, list(range(min(N_SMOKE, len(full_ds)))))
                loader = make_eval_loader(subset, EVAL_BATCH_SIZES[model_key])

                acc = clip_evaluate(model, loader, text_features, device,
                                    desc=f"Smoke {model_key}")

            elif backend == "blip2":
                model, processor = load_blip2_model(device)
                full_ds = PILImageFolder(str(IMAGENET_VAL_DIR), is_valid_file=is_valid_image)
                import random
                random.seed(SEED)
                indices = random.sample(range(len(full_ds)), min(N_SMOKE, len(full_ds)))
                subset_pil = [(full_ds[i][0], full_ds[i][1]) for i in indices]
                acc = blip2_generative_evaluate(model, processor, class_names, subset_pil, device,
                                                desc=f"Smoke {model_key}", batch_size=16)

            elif backend == "llava":
                model, processor = load_llava_model(device)

                full_ds = PILImageFolder(str(IMAGENET_VAL_DIR), is_valid_file=is_valid_image)
                n_llava = min(50, len(full_ds))  # Fewer for LLaVA (slow)
                import random
                random.seed(SEED)
                indices = random.sample(range(len(full_ds)), n_llava)
                subset_pil = [(full_ds[i][0], full_ds[i][1]) for i in indices]

                acc = llava_evaluate(model, processor, class_names, subset_pil, device,
                                     desc=f"Smoke {model_key}", batch_size=4)

            elapsed = time.time() - start
            expected = cfg.get("expected_imagenet_zs")

            # Validation
            status = "✓"
            if expected is not None:
                # Allow wider margin for small subset (sampling variance)
                # First 200 images are biased toward a few classes (fish/animals)
                margin = 0.25  # ±25% for 200-image non-stratified subset
                if abs(acc - expected) > margin:
                    status = "⚠ DEVIATION"
            elif backend in ("llava", "blip2"):
                # Generative VLM — just check it's > 0
                if acc > 0:
                    status = "✓ (generative)"
                else:
                    status = "⚠ ZERO ACC"

            results[model_key] = {
                "accuracy": acc,
                "expected": expected,
                "status": status,
                "time": elapsed,
            }

            expected_str = f"{expected:.3f}" if expected else "N/A"
            print(f"  Accuracy: {acc:.4f} (expected ~{expected_str}) — {status}")
            print(f"  Time: {elapsed:.1f}s")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            results[model_key] = {
                "accuracy": None,
                "expected": cfg.get("expected_imagenet_zs"),
                "status": f"FAILED: {e}",
                "time": 0,
            }
            print(f"  FAILED: {e}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    all_ok = True
    for mk, res in results.items():
        exp_str = f"{res['expected']:.3f}" if res['expected'] else "N/A"
        acc_str = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "FAIL"
        print(f"  {mk:25s}  acc={acc_str}  expected={exp_str}  {res['status']}  ({res['time']:.0f}s)")
        if "FAILED" in res["status"] or "ZERO" in res["status"]:
            all_ok = False

    if all_ok:
        print(f"\n  ✓ ALL {len(results)} MODELS PASSED SMOKE TEST")
    else:
        print(f"\n  ⚠ SOME MODELS FAILED — CHECK ABOVE")
    print(f"{'='*70}")

    return all_ok


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
        description="Phase 3: VLM Zero-Shot Baseline Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=list(ALL_MODELS.keys()) + ["all"],
                        help="VLM to evaluate")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=ALL_DATASETS + ["all"],
                        help="Dataset to evaluate on")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick integrity check on all models (200 ImageNet-Val images)")
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
    print(f"[INFO] Seed: {SEED}")

    set_seed()

    # ── Smoke test mode ──
    if args.smoke_test:
        success = run_smoke_test(device)
        sys.exit(0 if success else 1)

    # ── Regular evaluation mode ──
    if args.model is None or args.dataset is None:
        parser.error("--model and --dataset are required (or use --smoke-test)")

    model_list = list(ALL_MODELS.keys()) if args.model == "all" else [args.model]
    dataset_list = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    total_runs = len(model_list) * len(dataset_list)
    run_idx = 0
    failed_runs = []

    for ds_name in dataset_list:
        runner = DATASET_RUNNERS[ds_name]
        for m_key in model_list:
            run_idx += 1

            # ── Auto-resume: skip if JSON already exists ──
            json_path = RESULTS_DIR / f"{m_key}_{ds_name}_baselines.json"
            if json_path.exists():
                print(f"\n[SKIP] RUN {run_idx}/{total_runs}: "
                      f"{m_key} × {ds_name} — {json_path.name} exists")
                continue

            print(f"\n{'#'*70}")
            print(f"  RUN {run_idx}/{total_runs}: {m_key} × {ds_name}")
            print(f"  Family: {ALL_MODELS[m_key]['family']} | "
                  f"Params: {ALL_MODELS[m_key]['params']}")
            print(f"{'#'*70}")

            set_seed()
            try:
                runner(m_key, device)
            except Exception as e:
                print(f"[ERROR] {m_key} × {ds_name} failed: {e}")
                import traceback
                traceback.print_exc()
                failed_runs.append(f"{m_key} × {ds_name}")

    print(f"\n{'='*70}")
    print(f"  ALL RUNS COMPLETE ({run_idx - len(failed_runs)}/{total_runs} succeeded)")
    if failed_runs:
        print(f"  FAILED: {', '.join(failed_runs)}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
