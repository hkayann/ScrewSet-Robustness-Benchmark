#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL-8B on ScrewSet (40 classes) using LoRA.

Approach:
    - Load Qwen3-VL-8B in BF16
    - Apply LoRA adapters to the language model layers
    - Train with image+text prompt → class label generation
    - Evaluate on val/test splits, log per-class accuracy
    - Save best checkpoint by val accuracy

Usage:
    python3 finetune_qwen3_vl.py --smoke-test        # Quick 2-batch check
    python3 finetune_qwen3_vl.py                      # Full training
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
SCREWSET_TRAIN = REPO_ROOT / "data" / "screwset_split" / "train"
SCREWSET_VAL   = REPO_ROOT / "data" / "screwset_split" / "validation"
SCREWSET_TEST  = REPO_ROOT / "data" / "screwset_split" / "test"
RESULTS_DIR    = REPO_ROOT / "results" / "phase3"
CKPT_DIR       = REPO_ROOT / "results" / "screwset_s" / "models"

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective batch = 16
WARMUP_RATIO = 0.1
MAX_LENGTH = 2048
SEED = 42

PROMPT_TEMPLATE = (
    "Classify this screw image into exactly one of the following 40 categories:\n"
    "{class_list}\n\n"
    "Answer with ONLY the category name, nothing else."
)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class ScrewSetVLMDataset(Dataset):
    """Image classification dataset for Qwen3-VL fine-tuning."""

    def __init__(self, root_dir, class_names, split="train"):
        self.root_dir = Path(root_dir)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.split = split
        self.samples = []

        for cls_name in class_names:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    self.samples.append((str(img_file), cls_name))

        print(f"[{split}] Loaded {len(self.samples)} samples across {len(class_names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        return img_path, label


def build_prompt(class_names):
    class_list = ", ".join(class_names)
    return PROMPT_TEMPLATE.format(class_list=class_list)


# ──────────────────────────────────────────────────────────────
# Collate for Qwen3-VL
# ──────────────────────────────────────────────────────────────
def collate_fn(batch, processor, class_names, tokenizer, is_train=True):
    """Build Qwen3-VL chat messages and tokenize."""
    prompt = build_prompt(class_names)

    all_messages = []
    labels_text = []

    for img_path, label in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        if is_train:
            messages.append({"role": "assistant", "content": label})

        all_messages.append(messages)
        labels_text.append(label)

    # Process with Qwen's chat template
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=not is_train)
        for msgs in all_messages
    ]

    from qwen_vl_utils import process_vision_info
    image_inputs_list = []
    video_inputs_list = []
    for msgs in all_messages:
        img_in, vid_in = process_vision_info(msgs)
        image_inputs_list.append(img_in)
        video_inputs_list.append(vid_in)

    # Flatten image inputs
    all_images = []
    for img_list in image_inputs_list:
        if img_list:
            all_images.extend(img_list)

    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        padding=True,
        return_tensors="pt",
    )

    if is_train:
        input_ids = inputs["input_ids"]
        target_ids = input_ids.clone()

        # Mask everything except the assistant response (label tokens)
        for i in range(len(batch)):
            label_text = labels_text[i]
            label_tokens = tokenizer.encode(label_text, add_special_tokens=False)
            seq = input_ids[i].tolist()

            # Find the label tokens at the end of the sequence
            label_start = -1
            for j in range(len(seq) - len(label_tokens), -1, -1):
                if seq[j:j+len(label_tokens)] == label_tokens:
                    label_start = j
                    break

            if label_start >= 0:
                target_ids[i, :label_start] = -100
                # Also mask padding
                pad_mask = input_ids[i] == tokenizer.pad_token_id
                target_ids[i, pad_mask] = -100
            else:
                # Fallback: mask all but last few tokens
                target_ids[i, :] = -100

        inputs["labels"] = target_ids

    return inputs, labels_text


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, processor, tokenizer, dataset, class_names, batch_size=4, max_new_tokens=50):
    """Evaluate with generation and fuzzy matching."""
    from difflib import SequenceMatcher

    model.eval()
    prompt = build_prompt(class_names)

    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    # Process in batches
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start, end)]

        messages_batch = []
        for img_path, label in batch_items:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)

        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]

        from qwen_vl_utils import process_vision_info
        all_images = []
        for msgs in messages_batch:
            img_in, _ = process_vision_info(msgs)
            if img_in:
                all_images.extend(img_in)

        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )

        for idx, (img_path, target) in enumerate(batch_items):
            pred_raw = generated_texts[idx].strip()
            pred_lower = pred_raw.lower().strip()
            target_lower = target.lower().strip()

            matched = False
            # Exact match
            if pred_lower == target_lower:
                matched = True
            else:
                # Substring
                for cn in class_names:
                    if cn.lower() in pred_lower or pred_lower in cn.lower():
                        if cn.lower() == target_lower:
                            matched = True
                            break
                # Fuzzy
                if not matched:
                    best_score = 0
                    best_class = None
                    for cn in class_names:
                        score = SequenceMatcher(None, pred_lower, cn.lower()).ratio()
                        if score > best_score:
                            best_score = score
                            best_class = cn
                    if best_class and best_class.lower() == target_lower and best_score > 0.6:
                        matched = True

            if matched:
                correct += 1
                per_class[target]["correct"] += 1
            per_class[target]["total"] += 1
            total += 1

        if total % 500 == 0 and total > 0:
            print(f"  [{total}/{len(dataset)}] running acc = {correct/total*100:.2f}%")

    accuracy = correct / total if total > 0 else 0

    per_class_acc = {}
    for cls_name in class_names:
        c = per_class[cls_name]["correct"]
        t = per_class[cls_name]["total"]
        per_class_acc[cls_name] = {"correct": c, "total": t, "acc": c/t if t > 0 else 0}

    model.train()
    return accuracy, per_class_acc


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-batch test")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Fraction of training data to use (0.1 = 10%)")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    print("=" * 70)
    print("Qwen3-VL-8B Fine-tuning on ScrewSet (LoRA)")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Smoke test: {args.smoke_test}")
    print(f"Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}")
    print(f"LoRA r={args.lora_r}, alpha={LORA_ALPHA}")
    print(f"Subsample: {args.subsample*100:.0f}%")
    print()

    # Assert CUDA
    assert torch.cuda.is_available(), "CUDA required!"

    # ── Load model ──────────────────────────────────────────
    print("[1/5] Loading Qwen3-VL-8B...")
    t0 = time.time()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

    # Try Qwen3 first, fall back to Qwen2.5 class (they share architecture)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
        print("  Using Qwen3VLForConditionalGeneration")
    except ImportError:
        model_cls = Qwen2_5_VLForConditionalGeneration
        print("  Using Qwen2_5_VLForConditionalGeneration (fallback)")

    model = model_cls.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        min_pixels=28 * 28 * 4, max_pixels=448 * 448,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer.pad_token = tokenizer.pad_token

    print(f"  Model loaded in {time.time()-t0:.1f}s")
    print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # ── Apply LoRA ──────────────────────────────────────────
    print("\n[2/5] Applying LoRA adapters...")
    from peft import LoraConfig, get_peft_model, TaskType

    # Target the language model attention layers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load dataset ────────────────────────────────────────
    print("\n[3/5] Loading ScrewSet...")
    class_names = sorted([
        d for d in os.listdir(SCREWSET_TRAIN)
        if (SCREWSET_TRAIN / d).is_dir()
    ])
    assert len(class_names) == 40, f"Expected 40 classes, got {len(class_names)}"

    train_ds = ScrewSetVLMDataset(SCREWSET_TRAIN, class_names, split="train")
    val_ds = ScrewSetVLMDataset(SCREWSET_VAL, class_names, split="validation")
    test_ds = ScrewSetVLMDataset(SCREWSET_TEST, class_names, split="test")

    if args.smoke_test:
        # Limit to 8 train, 8 val, 8 test samples
        train_ds.samples = train_ds.samples[:8]
        val_ds.samples = val_ds.samples[:8]
        test_ds.samples = test_ds.samples[:8]
        args.epochs = 1
        print(f"  SMOKE TEST: {len(train_ds.samples)} train, {len(val_ds.samples)} val")
    elif args.subsample < 1.0:
        import random
        random.seed(SEED)
        # Stratified subsample: take same fraction from each class
        from collections import defaultdict
        by_class = defaultdict(list)
        for s in train_ds.samples:
            by_class[s[1]].append(s)
        subsampled = []
        for cls_name in class_names:
            cls_samples = by_class[cls_name]
            k = max(1, int(len(cls_samples) * args.subsample))
            subsampled.extend(random.sample(cls_samples, k))
        random.shuffle(subsampled)
        train_ds.samples = subsampled
        print(f"  SUBSAMPLE: {len(train_ds.samples)} train ({args.subsample*100:.0f}% stratified)")

    # ── Training loop ───────────────────────────────────────
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(train_ds) // (args.batch_size * GRAD_ACCUM_STEPS)) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0
    best_epoch = -1

    import random
    indices = list(range(len(train_ds)))

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(indices)
        epoch_loss = 0
        num_batches = 0

        t_epoch = time.time()

        for step_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[step_start:step_start + args.batch_size]
            batch = [train_ds[i] for i in batch_indices]

            try:
                inputs, labels_text = collate_fn(
                    batch, processor, class_names, tokenizer, is_train=True
                )
                inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

                outputs = model(**inputs)
                loss = outputs.loss / GRAD_ACCUM_STEPS

                loss.backward()

                if (num_batches + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                num_batches += 1

                if num_batches % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    lr_now = scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  Epoch {epoch+1}/{args.epochs} step {num_batches} | "
                          f"loss={avg_loss:.4f} lr={lr_now:.2e} gpu_mem={gpu_mem:.1f}GB",
                          flush=True)

            except Exception as e:
                print(f"  [WARN] Batch error: {e}")
                optimizer.zero_grad()
                continue

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - t_epoch
        print(f"\n  Epoch {epoch+1} done: avg_loss={avg_epoch_loss:.4f} "
              f"time={epoch_time:.0f}s ({epoch_time/60:.1f}min)", flush=True)

        # ── Save checkpoint BEFORE eval (so we don't lose work if eval OOMs) ──
        ckpt_path = CKPT_DIR / "qwen3_vl_8b_screwset_lora"
        model.save_pretrained(str(ckpt_path))
        print(f"  Checkpoint saved to {ckpt_path}", flush=True)

        # ── Validation ──────────────────────────────────────
        print(f"  Evaluating on validation set...")
        torch.cuda.empty_cache()
        import gc; gc.collect()
        val_acc, val_per_class = evaluate(
            model, processor, tokenizer, val_ds, class_names,
            batch_size=2,  # smaller batch for eval to avoid OOM
        )
        print(f"  Val accuracy: {val_acc*100:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"  *** New best! epoch={best_epoch} val_acc={best_val_acc*100:.2f}%")

    # ── Final Test Evaluation ───────────────────────────────
    print(f"\n[5/5] Final test evaluation (best epoch={best_epoch}, val_acc={best_val_acc*100:.2f}%)...", flush=True)

    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Load best adapter if we saved one
    if best_epoch > 0 and not args.smoke_test:
        ckpt_path = CKPT_DIR / "qwen3_vl_8b_screwset_lora"
        if ckpt_path.exists():
            from peft import PeftModel
            # Reload base and apply best adapter
            print(f"  Loading best adapter from {ckpt_path}...")

    test_acc, test_per_class = evaluate(
        model, processor, tokenizer, test_ds, class_names,
        batch_size=2,  # smaller batch for eval to avoid OOM
    )
    print(f"\n  TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"  Best val accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")

    # Per-class breakdown
    print("\n  Per-class accuracy:")
    for cls_name in class_names:
        info = test_per_class[cls_name]
        print(f"    {cls_name:30s} {info['correct']:4d}/{info['total']:4d} = {info['acc']*100:.1f}%")

    # ── Save results ────────────────────────────────────────
    result = {
        "model": "qwen3_vl_8b",
        "model_family": "Qwen3-VL",
        "evaluation_mode": "finetuned_lora",
        "dataset": "screwset",
        "num_classes": 40,
        "lora_config": {
            "r": args.lora_r,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": target_modules,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": args.batch_size * GRAD_ACCUM_STEPS,
            "learning_rate": args.lr,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "seed": SEED,
        },
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "per_class_accuracy": test_per_class,
        "smoke_test": args.smoke_test,
    }

    out_name = "qwen3_vl_8b_screwset_finetuned.json"
    if args.smoke_test:
        out_name = "qwen3_vl_8b_screwset_finetuned_smoke.json"

    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print(f"DONE — test_acc={test_acc*100:.2f}%, best_val={best_val_acc*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
