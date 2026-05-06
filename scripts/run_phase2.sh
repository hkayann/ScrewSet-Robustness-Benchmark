#!/bin/bash
# Run all Phase 2 ViT models for ScrewSet-S evaluation
set -e

# Activate your environment before running, e.g.:
#   conda activate arcade
#   source .venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Starting Phase 2 ViT models (bs=128)"
echo "$(date)"
echo "=========================================="

# Phase 2 models: vit_tiny_patch16_224, vit_small_patch16_224,
# deit_tiny_patch16_224, deit_small_patch16_224,
# swin_tiny_patch4_window7_224, mobilevit_s,
# efficientformer_l1, convnext_tiny

python scripts/eval_screwset_s.py --phase 2 --model all --batch-size 128 2>&1

echo ""
echo "=========================================="
echo "All Phase 2 models complete!"
echo "$(date)"
echo "=========================================="
