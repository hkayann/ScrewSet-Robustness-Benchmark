#!/bin/bash
# Run the 3 remaining Phase 1 models for ScrewSet-S evaluation
set -e

# Activate your environment before running, e.g.:
#   conda activate arcade
#   source .venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Starting remaining Phase 1 models"
echo "$(date)"
echo "=========================================="

# 1. efficientnetv2_rw_s (OOM at bs=256, use bs=128)
echo ""
echo "[$(date)] Starting efficientnetv2_rw_s (bs=128)..."
python scripts/eval_screwset_s.py --phase 1 --model efficientnetv2_rw_s --batch-size 128 2>&1
echo "[$(date)] efficientnetv2_rw_s DONE"

# 2. ghostnetv2_100 (slow model, use bs=256 to match original)
echo ""
echo "[$(date)] Starting ghostnetv2_100 (bs=256)..."
python scripts/eval_screwset_s.py --phase 1 --model ghostnetv2_100 --batch-size 256 2>&1
echo "[$(date)] ghostnetv2_100 DONE"

# 3. convnextv2_atto (standard, use bs=256)
echo ""
echo "[$(date)] Starting convnextv2_atto (bs=256)..."
python scripts/eval_screwset_s.py --phase 1 --model convnextv2_atto --batch-size 256 2>&1
echo "[$(date)] convnextv2_atto DONE"

echo ""
echo "=========================================="
echo "All Phase 1 remaining models complete!"
echo "$(date)"
echo "=========================================="
