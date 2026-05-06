#!/bin/bash
# Run ALL remaining ScrewSet-S evaluations: Phase 1 remaining → Phase 2 → Phase 3
# Then print summary
set -e

# Activate your environment before running, e.g.:
#   conda activate arcade
#   source .venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ScrewSet-S Full Evaluation Pipeline                           ║"
echo "║  Phase 1 (3 remaining CNNs) → Phase 2 (8 ViTs) → Phase 3 (8)  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Start: $(date)"
echo ""

# ── Phase 1: 3 remaining CNNs ──
echo "═══ PHASE 1: Remaining CNNs ═══"

for model in efficientnetv2_rw_s ghostnetv2_100 convnextv2_atto; do
    BS=256
    if [ "$model" = "efficientnetv2_rw_s" ]; then
        BS=128
    fi
    echo ""
    echo "[$(date)] Phase 1: $model (bs=$BS)"
    python scripts/eval_screwset_s.py --phase 1 --model $model --batch-size $BS 2>&1 || {
        echo "[ERROR] Phase 1 $model failed, continuing..."
    }
done

echo ""
echo "═══ PHASE 2: All 8 ViTs (bs=128) ═══"
python scripts/eval_screwset_s.py --phase 2 --model all --batch-size 128 2>&1 || {
    echo "[ERROR] Phase 2 had failures, continuing..."
}

echo ""
echo "═══ PHASE 3: All VLMs ═══"
python scripts/eval_screwset_s.py --phase 3 --model all 2>&1 || {
    echo "[ERROR] Phase 3 had failures, continuing..."
}

echo ""
echo "═══ FINAL SUMMARY ═══"
python scripts/eval_screwset_s.py --summary

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ALL DONE!                                                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "End: $(date)"
