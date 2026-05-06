#!/usr/bin/env python3
"""
Qwen3-VL Few-Shot + Linear Probe — ScrewSet
============================================
Evaluates Qwen3-VL-8B on ScrewSet with two additional protocols
on top of zero-shot (result reused from the existing JSON):

  • K-shot in-context learning   (K = 1, 2, 4 by default)
      One diverse set of K demo image-label pairs is prepended to every
      query as a multi-turn conversation. Same K demos for all queries.

  • Linear probe
      Freeze the Qwen3-VL visual encoder completely.
      Extract mean-pooled patch embeddings for every train/test image.
      Train a single nn.Linear on those features with Adam (200 epochs).

Usage:
    # Smoke test — verifies the whole pipeline in ~5 min
    python3 scripts/phase3/eval_qwen3_vl_fewshot_lp.py --smoke-test

    # Full evaluation (nohup-friendly)
    nohup python3 scripts/phase3/eval_qwen3_vl_fewshot_lp.py \
        > logs/qwen3_fewshot_lp.log 2>&1 &
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.config import DATA_DIR, SEED
from src.utils import patch_ipv4, set_seed
from src.datasets import is_valid_image, PILImageFolder
from src.class_names import get_screwset_class_names

patch_ipv4()

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR    = REPO_ROOT / "results" / "phase3"
SPLIT_DIR      = DATA_DIR / "screwset_split"

MODEL_ID       = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_KEY      = "qwen3_vl_8b"
ZS_JSON        = RESULTS_DIR / f"{MODEL_KEY}_screwset_baselines.json"

K_SHOTS        = [1, 2, 4]
FS_BATCH_SIZE  = 4     # queries per few-shot batch
LP_BATCH_SIZE  = 32    # images per LP feature-extraction batch
LP_EPOCHS      = 200
LP_LR          = 1e-2
LP_WD          = 1e-4
LP_MINIBATCH   = 512

FINETUNE_BEST  = 0.9979   # ViT-Small Phase-1 reference point

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(device):
    """Load Qwen3-VL-8B (fp16 eval mode). Returns (model, processor)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[MODEL] Loading {MODEL_ID} …")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=28 * 28 * 4, max_pixels=448 * 448
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {MODEL_KEY}: {n_params:,} params on {device}")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Shared response matcher  (identical to ZS eval — both bugs fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def _match(response, target_name, class_names_lower, target_idx):
    """Match generated text to the target class name using layered strategies."""
    from difflib import SequenceMatcher

    if not response:
        return False
    if response == target_name:
        return True
    if target_name in response:
        return True

    response_words = set(response.split())
    target_words   = set(target_name.split())
    generic = {"screw", "screws", "head", "number", "grade", "mm", "flat", "round"}
    meaningful = [w for w in target_words if len(w) > 3 and w not in generic]
    if meaningful and any(w in response_words for w in meaningful):
        return True

    best_r, best_i = 0.0, -1
    for i, cn in enumerate(class_names_lower):
        r = SequenceMatcher(None, response, cn).ratio()
        if r > best_r:
            best_r, best_i = r, i
    return best_i == target_idx and best_r > 0.4


# ═══════════════════════════════════════════════════════════════════════════════
# Few-shot evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def select_demonstrations(ds, class_names, k, rng):
    """Pick K (PIL_image, class_name) demo pairs covering K distinct classes.

    Works with PILImageFolder or a plain list of (PIL, label) tuples.
    """
    n_classes = len(class_names)

    # Build class → [index] map (fast path for PILImageFolder)
    class_to_indices = {c: [] for c in range(n_classes)}
    if hasattr(ds, "folder"):
        for idx, (_, lbl) in enumerate(ds.folder.samples):
            class_to_indices[lbl].append(idx)
    else:
        for idx, (_, lbl) in enumerate(ds):
            class_to_indices[lbl].append(idx)

    classes = list(range(n_classes))
    rng.shuffle(classes)

    demos = []
    for c in classes[:k]:
        indices = class_to_indices.get(c, [])
        if not indices:
            continue
        img, _ = ds[rng.choice(indices)]
        demos.append((img, class_names[c]))
    return demos


def fewshot_evaluate(model, processor, class_names, test_dataset,
                     demos, device, desc="few-shot", batch_size=FS_BATCH_SIZE):
    """K-shot in-context generative evaluation.

    Conversation structure per query:
      user:      [demo_img_1]  What type of screw is this?
      assistant: <class_1>
      …  (repeated K times)
      user:      [query_img]   Classify this image into exactly one of: …
      (model generates)
    """
    from qwen_vl_utils import process_vision_info

    class_names_lower = [c.lower() for c in class_names]
    class_list  = ", ".join(class_names)
    query_text  = (
        f"Classify this image into exactly one of these categories: "
        f"{class_list}. Answer with only the category name, nothing else."
    )

    # Pre-build the demo turns once; they are reused for every query.
    demo_turns = []
    for demo_img, demo_class in demos:
        demo_turns += [
            {"role": "user", "content": [
                {"type": "image", "image": demo_img},
                {"type": "text",  "text": "What type of screw is this?"},
            ]},
            {"role": "assistant", "content": demo_class},
        ]

    correct, total = 0, 0
    n   = len(test_dataset)
    dbg = 0

    for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        end   = min(start + batch_size, n)
        batch = [test_dataset[j] for j in range(start, end)]
        batch_imgs   = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]

        # Each conversation = K demo turns + 1 query turn
        conversations = []
        for q_img in batch_imgs:
            conv = demo_turns + [{
                "role": "user",
                "content": [
                    {"type": "image", "image": q_img},
                    {"type": "text",  "text": query_text},
                ],
            }]
            conversations.append(conv)

        try:
            prompts = [
                processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in conversations
            ]
            all_images = []
            for conv in conversations:
                imgs, _ = process_vision_info(conv)
                all_images.extend(imgs if imgs else [])

            inputs = processor(
                text=prompts,
                images=all_images if all_images else None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                input_len = inputs["input_ids"].shape[-1]
                outputs   = model.generate(**inputs, max_new_tokens=30, do_sample=False)

            generated = processor.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

            for text, label in zip(generated, batch_labels):
                t_low = text.lower().strip().rstrip(".")
                if dbg < 3:
                    print(f"  [DBG FS] response={t_low!r}  "
                          f"target={class_names_lower[label]!r}")
                    dbg += 1
                if _match(t_low, class_names_lower[label], class_names_lower, label):
                    correct += 1
                total += 1

        except Exception as exc:
            print(f"  [WARN] few-shot batch {start}: {exc}")
            total += len(batch)

    return correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Linear probe — feature extraction + linear head training
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(model, processor, dataset, device,
                     desc="features", batch_size=LP_BATCH_SIZE):
    """Extract mean-pooled visual encoder features for all images.

    Uses model.model.visual (Qwen3VLVisionModel inside Qwen3VLModel) directly.
    We use .pooler_output (the post-merger token sequence — same features the
    LLM receives) and mean-pool across the merged patches.
    Qwen3-VL merges 2×2 spatial patches, so:
        n_output_tokens_per_image = (h // 2) * (w // 2)
    where (t, h, w) = image_grid_thw for that image.

    Returns:
        features : FloatTensor (N, hidden_dim) on CPU
        labels   : LongTensor (N,)
    """
    from qwen_vl_utils import process_vision_info

    all_feats  = []
    all_labels = []
    n = len(dataset)

    for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        end   = min(start + batch_size, n)
        batch = [dataset[j] for j in range(start, end)]
        batch_imgs   = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]

        # Minimal conversation: image only, no text generation
        conversations = [
            [{"role": "user", "content": [{"type": "image", "image": img}]}]
            for img in batch_imgs
        ]
        prompts = [
            processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in conversations
        ]
        all_imgs_flat = []
        for conv in conversations:
            imgs, _ = process_vision_info(conv)
            all_imgs_flat.extend(imgs if imgs else [])

        try:
            inputs = processor(
                text=prompts,
                images=all_imgs_flat if all_imgs_flat else None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            pixel_values   = inputs["pixel_values"]    # (total_raw_patches, …)
            image_grid_thw = inputs["image_grid_thw"]  # (B, 3)  [t, h, w]

            with torch.no_grad():
                # model.model.visual returns BaseModelOutputWithDeepstackFeatures.
                # .pooler_output = merged hidden states (what LLM actually sees),
                # shape: (total_merged_patches_across_batch, hidden_dim)
                vis_out = model.model.visual(
                    hidden_states=pixel_values, grid_thw=image_grid_thw
                )
                patch_embs = vis_out.pooler_output  # (total_merged, hidden_dim)

            # Split by image using merged patch count formula, then mean-pool
            ptr = 0
            for grid in image_grid_thw:
                t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
                n_merged = t * (h // 2) * (w // 2)
                img_embs = patch_embs[ptr: ptr + n_merged]
                all_feats.append(img_embs.mean(0).cpu().float())
                ptr += n_merged

            all_labels.extend(batch_labels)

        except Exception as exc:
            print(f"  [WARN] feature extraction batch {start}: {exc}")
            raise  # re-raise so smoke test catches bugs early

    features = torch.stack(all_feats)
    labels   = torch.tensor(all_labels, dtype=torch.long)
    return features, labels


def train_linear_probe(X_tr, y_tr, X_te, y_te, n_classes, device):
    """Train a linear classifier on L2-normalised features using Adam.

    Returns final test accuracy (float in [0, 1]).
    """
    X_tr_n = F.normalize(X_tr.to(device), dim=1).float()
    X_te_n = F.normalize(X_te.to(device), dim=1).float()
    y_tr   = y_tr.to(device)
    y_te   = y_te.to(device)

    feat_dim = X_tr_n.shape[1]
    n_train  = X_tr_n.shape[0]
    print(f"[LP] feat_dim={feat_dim}, train_n={n_train}, "
          f"test_n={len(X_te_n)}, n_classes={n_classes}")

    linear = nn.Linear(feat_dim, n_classes).to(device)
    opt    = torch.optim.Adam(linear.parameters(), lr=LP_LR, weight_decay=LP_WD)
    crit   = nn.CrossEntropyLoss()

    perm   = torch.randperm(n_train, device=device)
    X_tr_n = X_tr_n[perm]
    y_tr   = y_tr[perm]

    for epoch in range(1, LP_EPOCHS + 1):
        linear.train()
        total_loss = 0.0
        for s in range(0, n_train, LP_MINIBATCH):
            xb = X_tr_n[s: s + LP_MINIBATCH]
            yb = y_tr  [s: s + LP_MINIBATCH]
            opt.zero_grad()
            loss = crit(linear(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 50 == 0 or epoch == LP_EPOCHS:
            linear.eval()
            with torch.no_grad():
                test_acc = (linear(X_te_n).argmax(1) == y_te).float().mean().item()
            print(f"  [LP] epoch {epoch:3d}/{LP_EPOCHS}  "
                  f"loss={total_loss:.4f}  test_acc={test_acc:.4f}")

    linear.eval()
    with torch.no_grad():
        final_acc = (linear(X_te_n).argmax(1) == y_te).float().mean().item()
    return final_acc


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(results, smoke_test):
    suffix = "_smoke" if smoke_test else ""
    out = RESULTS_DIR / f"{MODEL_KEY}_screwset_fewshot_lp{suffix}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[SAVED] {out}")


def _print_summary(results, k_shots):
    zs  = results.get("zero_shot_acc")
    lp  = results.get("linear_probe_acc")
    fsa = results.get("few_shot_accs", {})

    print(f"\n{'='*70}")
    print(f"  RESULTS  —  Qwen3-VL ScrewSet: ZS / Few-shot / LP comparison")
    print(f"{'='*70}")
    print(f"  {'Method':<26} {'Accuracy':>10}")
    print(f"  {'-'*38}")
    if zs is not None:
        print(f"  {'Zero-shot (K=0)':<26} {zs*100:>9.2f}%")
    for k in k_shots:
        v = fsa.get(f"{k}_shot")
        if v is not None:
            print(f"  {f'{k}-shot in-context':<26} {v*100:>9.2f}%")
    if lp is not None:
        print(f"  {'Linear Probe (frozen)':<26} {lp*100:>9.2f}%")
    print(f"  {'Fine-tuned best':<26} {FINETUNE_BEST*100:>9.2f}%  ← ViT-Small Phase-1")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL few-shot + linear probe on ScrewSet"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Tiny run (20 test images, ≥40 train) for pipeline sanity check"
    )
    parser.add_argument("--smoke-n",      type=int, default=20,
                        help="Number of test images in smoke test (default 20)")
    parser.add_argument("--skip-fewshot", action="store_true")
    parser.add_argument("--skip-lp",      action="store_true")
    parser.add_argument("--k-shots",      type=int, nargs="+", default=K_SHOTS,
                        help="K values for few-shot (default: 1 2 4)")
    parser.add_argument("--fs-batch",     type=int, default=FS_BATCH_SIZE,
                        help="Query batch size for few-shot inference")
    parser.add_argument("--lp-batch",     type=int, default=LP_BATCH_SIZE,
                        help="Image batch size for LP feature extraction")
    args = parser.parse_args()

    set_seed(SEED)
    rng    = random.Random(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Datasets ───────────────────────────────────────────────────────────────
    class_names, _ = get_screwset_class_names(SPLIT_DIR)
    n_classes = len(class_names)
    print(f"[INFO] ScrewSet: {n_classes} classes")

    full_train = PILImageFolder(str(SPLIT_DIR / "train"), is_valid_file=is_valid_image)
    full_test  = PILImageFolder(str(SPLIT_DIR / "test"),  is_valid_file=is_valid_image)
    print(f"[INFO] Full train: {len(full_train)},  Full test: {len(full_test)}")

    if args.smoke_test:
        # Need ≥ n_classes images in train so select_demonstrations finds each class
        n_train = max(n_classes * 2, args.smoke_n * 2)   # ≥ 80 images
        n_test  = args.smoke_n
        train_ds = [full_train[i] for i in range(min(n_train, len(full_train)))]
        test_ds  = [full_test[i]  for i in range(min(n_test,  len(full_test)))]
        print(f"[SMOKE] Using {len(test_ds)} test images, {len(train_ds)} train images")
    else:
        train_ds = full_train
        test_ds  = full_test

    # ── Zero-shot result (reused) ──────────────────────────────────────────────
    zs_acc = None
    if ZS_JSON.exists():
        with open(ZS_JSON) as f:
            zs_acc = json.load(f).get("test_acc")
        print(f"[INFO] Zero-shot acc (from {ZS_JSON.name}): {zs_acc:.4f}")
    else:
        print(f"[WARN] ZS result not found: {ZS_JSON}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, processor = load_model(device)

    results = {
        "model":            MODEL_KEY,
        "dataset":          "screwset",
        "n_classes":        n_classes,
        "smoke_test":       args.smoke_test,
        "seed":             SEED,
        "zero_shot_acc":    zs_acc,
        "few_shot_accs":    {},
        "linear_probe_acc": None,
    }

    # ══ Few-shot ══════════════════════════════════════════════════════════════
    if not args.skip_fewshot:
        for k in args.k_shots:
            print(f"\n{'='*70}")
            print(f"  {k}-shot in-context  —  {MODEL_KEY}")
            print(f"{'='*70}")

            demos = select_demonstrations(train_ds, class_names, k, rng)
            print(f"[INFO] {k} demonstration(s) selected:")
            for _, d_cls in demos:
                print(f"  • {d_cls}")

            t0  = time.time()
            acc = fewshot_evaluate(
                model, processor, class_names, test_ds, demos, device,
                desc=f"[{MODEL_KEY}] {k}-shot",
                batch_size=args.fs_batch,
            )
            elapsed = time.time() - t0
            print(f"\n[{MODEL_KEY}] {k}-shot ScrewSet Acc: {acc:.4f}  "
                  f"({elapsed / 60:.1f} min)")
            results["few_shot_accs"][f"{k}_shot"] = acc
            _save(results, args.smoke_test)   # checkpoint after each K

    # ══ Linear probe ══════════════════════════════════════════════════════════
    if not args.skip_lp:
        print(f"\n{'='*70}")
        print(f"  Linear Probe  —  {MODEL_KEY}")
        print(f"{'='*70}")

        t0 = time.time()
        print("[LP] Extracting train features …")
        X_tr, y_tr = extract_features(
            model, processor, train_ds, device,
            desc="[LP] train", batch_size=args.lp_batch,
        )
        print(f"[LP] Train features: {X_tr.shape}  ({(time.time() - t0) / 60:.1f} min)")

        t0 = time.time()
        print("[LP] Extracting test features …")
        X_te, y_te = extract_features(
            model, processor, test_ds, device,
            desc="[LP] test", batch_size=args.lp_batch,
        )
        print(f"[LP] Test features: {X_te.shape}  ({(time.time() - t0) / 60:.1f} min)")

        print("[LP] Training linear head …")
        lp_acc = train_linear_probe(X_tr, y_tr, X_te, y_te, n_classes, device)
        print(f"\n[{MODEL_KEY}] Linear Probe ScrewSet Acc: {lp_acc:.4f}")
        results["linear_probe_acc"] = lp_acc
        _save(results, args.smoke_test)

    # ── Final summary ──────────────────────────────────────────────────────────
    _print_summary(results, args.k_shots)
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
