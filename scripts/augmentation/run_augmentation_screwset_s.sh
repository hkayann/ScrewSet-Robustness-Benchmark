#!/usr/bin/env bash
#
# Run all 24 augmentation × ScrewSet-S experiments with subprocess isolation.
# Each (method × model) pair runs as a separate Python process so CUDA memory
# is fully released between runs — prevents OOM cascading.
#
# Usage:
#   nohup bash scripts/augmentation/run_augmentation_screwset_s.sh > /dev/null 2>&1 &
#
# Monitor:
#   tail -f logs/augmentation_screwset_s_v3.log
#   ls -la results/screwset_s/augmentation/
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs

LOG="logs/augmentation_screwset_s_v3.log"
SCRIPT="scripts/augmentation/train_and_eval_screwset_s.py"
RESULTS_DIR="results/screwset_s/augmentation"

METHODS=(cutmix_mixup randaugment trivialaugment 3augment tta)
MODELS=(resnet18 efficientnetv2_rw_s vit_tiny_patch16_224 convnext_tiny)

TOTAL=$(( ${#METHODS[@]} * ${#MODELS[@]} ))
RUN=0
COMPLETED=0
SKIPPED=0
FAILED=0

echo "═══════════════════════════════════════════════════════════════" | tee "$LOG"
echo "  Augmentation → ScrewSet-S Pipeline (subprocess-isolated)"    | tee -a "$LOG"
echo "  Started: $(date)"                                            | tee -a "$LOG"
echo "  Methods: ${METHODS[*]}"                                      | tee -a "$LOG"
echo "  Models:  ${MODELS[*]}"                                       | tee -a "$LOG"
echo "  Total runs: $TOTAL (skipping existing results)"              | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG"

for METHOD in "${METHODS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        RUN=$((RUN + 1))

        # Skip if result JSON already exists
        RESULT_JSON="${RESULTS_DIR}/${MODEL}_screwset_${METHOD}_ss_s.json"
        if [[ -f "$RESULT_JSON" ]]; then
            echo "[SKIP $RUN/$TOTAL] ${METHOD} × ${MODEL} — result exists" | tee -a "$LOG"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "" | tee -a "$LOG"
        echo "══════════════════════════════════════════════════════════" | tee -a "$LOG"
        echo "  RUN $RUN/$TOTAL: ${METHOD} × ${MODEL}" | tee -a "$LOG"
        echo "  $(date)" | tee -a "$LOG"

        # Show GPU before run
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | tee -a "$LOG"

        # Run as isolated subprocess — fresh CUDA context each time
        RUN_LOG="logs/aug_ss_s_v3_${METHOD}_${MODEL}.log"
        python3 "$SCRIPT" --method "$METHOD" --model "$MODEL" > "$RUN_LOG" 2>&1
        EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 0 ]]; then
            # Verify JSON was actually created
            if [[ -f "$RESULT_JSON" ]]; then
                COMPLETED=$((COMPLETED + 1))
                echo "  ✓ DONE ($COMPLETED completed)" | tee -a "$LOG"
            else
                FAILED=$((FAILED + 1))
                echo "  ✗ FAILED (exit 0 but no JSON) — see $RUN_LOG" | tee -a "$LOG"
            fi
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ FAILED (exit $EXIT_CODE) — see $RUN_LOG" | tee -a "$LOG"
            # Print last 15 lines of the failed run log for diagnosis
            tail -15 "$RUN_LOG" 2>/dev/null | tee -a "$LOG"
        fi

        # Brief pause to let GPU fully release memory
        sleep 3
    done
done

echo "" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  PIPELINE COMPLETE — $(date)" | tee -a "$LOG"
echo "  Total: $TOTAL | Completed: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$LOG"
echo "  Results: $RESULTS_DIR" | tee -a "$LOG"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG"
