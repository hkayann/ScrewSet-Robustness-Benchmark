#!/usr/bin/env python3
"""
Evaluate a saved LoRA-finetuned Qwen3-VL-8B checkpoint on ScrewSet.
Loads the base model + LoRA adapter, runs val and test evaluation.
"""

import json
import os
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCREWSET_VAL  = REPO_ROOT / "data" / "screwset_split" / "validation"
SCREWSET_TEST = REPO_ROOT / "data" / "screwset_split" / "test"
RESULTS_DIR   = REPO_ROOT / "results" / "phase3"
CKPT_DIR      = REPO_ROOT / "results" / "screwset_s" / "models" / "qwen3_vl_8b_screwset_lora"
MODEL_NAME    = "Qwen/Qwen3-VL-8B-Instruct"
SEED = 42

PROMPT_TEMPLATE = (
    "Classify this screw image into exactly one of the following 40 categories:\n"
    "{class_list}\n\n"
    "Answer with ONLY the category name, nothing else."
)


class ScrewSetVLMDataset:
    def __init__(self, root_dir, class_names, split="test"):
        self.root_dir = Path(root_dir)
        self.class_names = class_names
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
        return self.samples[idx]


@torch.no_grad()
def evaluate(model, processor, tokenizer, dataset, class_names, batch_size=2):
    model.eval()
    prompt = PROMPT_TEMPLATE.format(class_list=", ".join(class_names))

    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start, end)]

        messages_batch = []
        for img_path, label in batch_items:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": prompt},
            ]}]
            messages_batch.append(messages)

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_batch]

        from qwen_vl_utils import process_vision_info
        all_images = []
        for msgs in messages_batch:
            img_in, _ = process_vision_info(msgs)
            if img_in:
                all_images.extend(img_in)

        inputs = processor(
            text=texts, images=all_images if all_images else None,
            padding=True, return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            temperature=None, top_p=None,
        )

        input_len = inputs["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )

        for idx, (img_path, target) in enumerate(batch_items):
            pred_raw = generated_texts[idx].strip()
            pred_lower = pred_raw.lower().strip()
            target_lower = target.lower().strip()

            matched = False
            if pred_lower == target_lower:
                matched = True
            else:
                for cn in class_names:
                    if cn.lower() in pred_lower or pred_lower in cn.lower():
                        if cn.lower() == target_lower:
                            matched = True
                            break
                if not matched:
                    best_score, best_class = 0, None
                    for cn in class_names:
                        score = SequenceMatcher(None, pred_lower, cn.lower()).ratio()
                        if score > best_score:
                            best_score, best_class = score, cn
                    if best_class and best_class.lower() == target_lower and best_score > 0.6:
                        matched = True

            if matched:
                correct += 1
                per_class[target]["correct"] += 1
            per_class[target]["total"] += 1
            total += 1

        if total % 200 == 0 and total > 0:
            print(f"  [{total}/{len(dataset)}] running acc = {correct/total*100:.2f}%", flush=True)

    accuracy = correct / total if total > 0 else 0
    per_class_acc = {}
    for cls_name in class_names:
        c = per_class[cls_name]["correct"]
        t = per_class[cls_name]["total"]
        per_class_acc[cls_name] = {"correct": c, "total": t, "acc": c/t if t > 0 else 0}

    return accuracy, per_class_acc


def main():
    torch.manual_seed(SEED)
    assert torch.cuda.is_available(), "CUDA required!"

    print("=" * 70)
    print("Qwen3-VL-8B LoRA Checkpoint Evaluation on ScrewSet")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CKPT_DIR}")
    print()

    # Load base model
    print("[1/4] Loading base model...")
    t0 = time.time()
    from transformers import AutoProcessor, AutoTokenizer
    try:
        from transformers import Qwen3VLForConditionalGeneration as model_cls
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as model_cls

    model = model_cls.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        min_pixels=28*28*4, max_pixels=448*448,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer.pad_token = tokenizer.pad_token
    print(f"  Base model loaded in {time.time()-t0:.1f}s")

    # Apply LoRA adapter
    print("\n[2/4] Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(CKPT_DIR))
    model.eval()
    print(f"  LoRA adapter loaded from {CKPT_DIR}")

    # Load datasets
    print("\n[3/4] Loading datasets...")
    class_names = sorted([
        d for d in os.listdir(SCREWSET_VAL) if (SCREWSET_VAL / d).is_dir()
    ])
    val_ds = ScrewSetVLMDataset(SCREWSET_VAL, class_names, split="validation")
    test_ds = ScrewSetVLMDataset(SCREWSET_TEST, class_names, split="test")

    # Evaluate
    print("\n[4/4] Running evaluation...")
    torch.cuda.empty_cache()

    print("\n  --- Validation ---")
    val_acc, val_per_class = evaluate(model, processor, tokenizer, val_ds, class_names)
    print(f"\n  VAL ACCURACY: {val_acc*100:.2f}%")

    torch.cuda.empty_cache()

    print("\n  --- Test ---")
    test_acc, test_per_class = evaluate(model, processor, tokenizer, test_ds, class_names)
    print(f"\n  TEST ACCURACY: {test_acc*100:.2f}%")

    print("\n  Per-class accuracy:")
    for cls_name in class_names:
        info = test_per_class[cls_name]
        print(f"    {cls_name:30s} {info['correct']:4d}/{info['total']:4d} = {info['acc']*100:.1f}%")

    # Save results
    result = {
        "model": "qwen3_vl_8b",
        "model_family": "Qwen3-VL",
        "evaluation_mode": "finetuned_lora",
        "dataset": "screwset",
        "num_classes": 40,
        "lora_checkpoint": str(CKPT_DIR),
        "val_acc": val_acc,
        "test_acc": test_acc,
        "per_class_accuracy": test_per_class,
        "val_per_class_accuracy": val_per_class,
        "seed": SEED,
    }

    out_path = RESULTS_DIR / "qwen3_vl_8b_screwset_finetuned.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"\n{'='*70}")
    print(f"DONE — test_acc={test_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
