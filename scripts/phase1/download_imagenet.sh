#!/bin/bash
# Download ImageNet validation set (from HuggingFace) and ImageNet-C (from Zenodo)
# Requires: HuggingFace token at ~/.cache/huggingface/token
# Uses curl -4 to force IPv4 (IPv6 is broken on this workstation)

set -e

DATA_DIR="/home/hakan/ARCADE--Screwset/data"
HF_TOKEN=$(cat ~/.cache/huggingface/token)
IMAGENET_VAL_DIR="$DATA_DIR/imagenet-val-parquet"
IMAGENET_C_DIR="$DATA_DIR/imagenet-c"

echo "============================================================"
echo "  STEP 1: Download ImageNet Validation (14 parquet files)"
echo "============================================================"

mkdir -p "$IMAGENET_VAL_DIR"

for i in $(seq -w 0 13); do
    FILE="validation-000${i}-of-00014.parquet"
    DEST="$IMAGENET_VAL_DIR/$FILE"
    if [[ -f "$DEST" ]]; then
        SIZE=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
        if [[ $SIZE -gt 100000000 ]]; then
            echo "[SKIP] $FILE already exists ($(numfmt --to=iec $SIZE))"
            continue
        fi
    fi
    echo "[DOWN] $FILE ..."
    curl -4 -L --connect-timeout 30 --retry 3 --retry-delay 5 \
        -H "Authorization: Bearer $HF_TOKEN" \
        -o "$DEST" \
        "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/$FILE"
    SIZE=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
    echo "  -> $(numfmt --to=iec $SIZE)"
done

echo ""
echo "ImageNet val parquet download complete."
echo "Total size: $(du -sh $IMAGENET_VAL_DIR | cut -f1)"

echo ""
echo "============================================================"
echo "  STEP 2: Download ImageNet-C from Zenodo (5 tar files)"
echo "============================================================"

mkdir -p "$IMAGENET_C_DIR"

# Zenodo direct download links for ImageNet-C (DOI: 10.5281/zenodo.2235448)
declare -A TARS
TARS[blur]="https://zenodo.org/records/2235448/files/blur.tar"
TARS[digital]="https://zenodo.org/records/2235448/files/digital.tar"
TARS[noise]="https://zenodo.org/records/2235448/files/noise.tar"
TARS[weather]="https://zenodo.org/records/2235448/files/weather.tar"
TARS[extra]="https://zenodo.org/records/2235448/files/extra.tar"

for name in blur digital noise weather extra; do
    TAR_FILE="$IMAGENET_C_DIR/${name}.tar"
    URL="${TARS[$name]}"

    if [[ -f "$TAR_FILE.done" ]]; then
        echo "[SKIP] ${name}.tar already downloaded"
        continue
    fi

    echo "[DOWN] ${name}.tar ..."
    curl -4 -L --connect-timeout 30 --retry 3 --retry-delay 5 \
        -o "$TAR_FILE" \
        "$URL"
    SIZE=$(stat -c%s "$TAR_FILE" 2>/dev/null || echo 0)
    echo "  -> $(numfmt --to=iec $SIZE)"
    touch "$TAR_FILE.done"
done

echo ""
echo "ImageNet-C tar download complete."
echo "Total size: $(du -sh $IMAGENET_C_DIR | cut -f1)"

echo ""
echo "============================================================"
echo "  STEP 3: Extract ImageNet-C"
echo "============================================================"

cd "$IMAGENET_C_DIR"
for name in blur digital noise weather extra; do
    TAR_FILE="${name}.tar"
    if [[ ! -f "$TAR_FILE" ]]; then
        echo "[WARN] $TAR_FILE not found, skipping"
        continue
    fi
    if [[ -f "${name}.extracted" ]]; then
        echo "[SKIP] ${name}.tar already extracted"
        continue
    fi
    echo "[EXTRACT] ${name}.tar ..."
    tar xf "$TAR_FILE"
    touch "${name}.extracted"
    echo "  -> done"
done

echo ""
echo "============================================================"
echo "  ALL DOWNLOADS COMPLETE"
echo "  $(date)"
echo "============================================================"
echo "ImageNet val parquet: $IMAGENET_VAL_DIR"
echo "ImageNet-C:           $IMAGENET_C_DIR"
du -sh "$IMAGENET_VAL_DIR" "$IMAGENET_C_DIR"
