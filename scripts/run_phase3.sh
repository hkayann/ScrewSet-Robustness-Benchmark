#!/bin/bash
# Run all Phase 3 VLM models for ScrewSet-S evaluation
set -e

# Activate your environment before running, e.g.:
#   conda activate arcade
#   source .venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Starting Phase 3 VLM models"
echo "$(date)"
echo "=========================================="

python scripts/eval_screwset_s.py --phase 3 --model all 2>&1

echo ""
echo "=========================================="
echo "All Phase 3 models complete!"
echo "$(date)"
echo "=========================================="
