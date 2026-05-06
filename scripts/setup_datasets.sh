#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"
cd "${ROOT_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

download_file() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    log "Exists: $out"
  else
    log "Downloading: $url"
  fi
  wget -c -O "$out" "$url"
}

extract_targz_into_data_root() {
  local archive="$1"
  log "Extracting: $archive"
  tar xzf "$archive" -C "${DATA_DIR}"
}

extract_tar_into_dir() {
  local archive="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  log "Extracting: $archive -> $target_dir"
  tar xf "$archive" -C "$target_dir"
}

log "=== Dataset bootstrap started ==="
log "Root: ${ROOT_DIR}"
log "Data dir: ${DATA_DIR}"

# 1) ScrewSet (clean)
if [[ -d "${DATA_DIR}/screwset_split/train" && -d "${DATA_DIR}/screwset_split/validation" && -d "${DATA_DIR}/screwset_split/test" ]]; then
  log "ScrewSet split already prepared."
else
  download_file "https://zenodo.org/records/16744219/files/screwset_split.tar.gz" "${DATA_DIR}/screwset_split.tar.gz"
  extract_targz_into_data_root "${DATA_DIR}/screwset_split.tar.gz"
  rm -f "${DATA_DIR}/screwset_split.tar.gz"
  log "ScrewSet split ready."
fi

# 2) ScrewSet-C
if [[ -d "${DATA_DIR}/screwset_c" ]]; then
  log "ScrewSet-C already present."
else
  download_file "https://zenodo.org/records/16744219/files/screwset_c.tar.gz" "${DATA_DIR}/screwset_c.tar.gz"
  extract_targz_into_data_root "${DATA_DIR}/screwset_c.tar.gz"
  rm -f "${DATA_DIR}/screwset_c.tar.gz"
  log "ScrewSet-C ready."
fi

# 3) CIFAR-10
if [[ -d "${DATA_DIR}/cifar10/cifar-10-batches-py" ]]; then
  log "CIFAR-10 already present."
else
  log "Downloading CIFAR-10..."
  set +e
  python3 -c "from torchvision.datasets import CIFAR10; CIFAR10(root='${DATA_DIR}/cifar10', train=True, download=True); CIFAR10(root='${DATA_DIR}/cifar10', train=False, download=True); print('CIFAR-10 downloaded via torchvision.')"
  cifar_status=$?
  set -e
  if [[ $cifar_status -ne 0 ]]; then
    log "torchvision path failed; using direct CIFAR-10 tarball fallback"
    mkdir -p "${DATA_DIR}/cifar10"
    download_file "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" "${DATA_DIR}/cifar-10-python.tar.gz"
    tar xzf "${DATA_DIR}/cifar-10-python.tar.gz" -C "${DATA_DIR}/cifar10"
    rm -f "${DATA_DIR}/cifar-10-python.tar.gz"
  fi
fi

# 4) CIFAR-10-C
if [[ -d "${DATA_DIR}/cifar10_c/CIFAR-10-C" ]]; then
  log "CIFAR-10-C already present."
else
  mkdir -p "${DATA_DIR}/cifar10_c"
  download_file "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar" "${DATA_DIR}/CIFAR-10-C.tar"
  extract_tar_into_dir "${DATA_DIR}/CIFAR-10-C.tar" "${DATA_DIR}/cifar10_c"
  rm -f "${DATA_DIR}/CIFAR-10-C.tar"
  log "CIFAR-10-C ready."
fi
ln -sfn "cifar10_c/CIFAR-10-C" "${DATA_DIR}/CIFAR-10-C"

# 5) ImageNet-A
if [[ -d "${DATA_DIR}/imagenet-a" ]]; then
  log "ImageNet-A already present."
else
  download_file "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar" "${DATA_DIR}/imagenet-a.tar"
  extract_tar_into_dir "${DATA_DIR}/imagenet-a.tar" "${DATA_DIR}"
  rm -f "${DATA_DIR}/imagenet-a.tar"
  log "ImageNet-A ready."
fi

# 6) ImageNet validation (helper script, may require auth)
if [[ -d "${DATA_DIR}/imagenet-val" && "$(find "${DATA_DIR}/imagenet-val" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ge 1000 ]]; then
  log "ImageNet validation already present."
else
  log "Attempting ImageNet validation download via scripts/phase1/download_imagenet_val.py"
  set +e
  python3 "${ROOT_DIR}/scripts/phase1/download_imagenet_val.py"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    log "WARNING: ImageNet validation auto-download failed (likely access/auth issue)."
  fi
fi

# 7) ImageNet-C
mkdir -p "${DATA_DIR}/imagenet-c"
if [[ -d "${DATA_DIR}/imagenet-c/gaussian_noise/1" && -d "${DATA_DIR}/imagenet-c/zoom_blur/5" ]]; then
  log "ImageNet-C already extracted."
else
  cd "${DATA_DIR}/imagenet-c"
  download_file "https://zenodo.org/records/2235448/files/blur.tar" "blur.tar"
  download_file "https://zenodo.org/records/2235448/files/digital.tar" "digital.tar"
  download_file "https://zenodo.org/records/2235448/files/extra.tar" "extra.tar"
  download_file "https://zenodo.org/records/2235448/files/noise.tar" "noise.tar"
  download_file "https://zenodo.org/records/2235448/files/weather.tar" "weather.tar"

  for f in blur.tar digital.tar extra.tar noise.tar weather.tar; do
    log "Extracting ${f}"
    tar xf "$f"
    log "Done ${f}"
  done

  rm -f blur.tar digital.tar extra.tar noise.tar weather.tar
  cd "${ROOT_DIR}"
  log "ImageNet-C ready."
fi

# 8) LENS (manual source link usually required)
if [[ -d "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-diverse-test" ]]; then
  log "LENS dataset already present."
else
  mkdir -p "${DATA_DIR}/lens"
  if [[ ! -f "${DATA_DIR}/lens/ImageNet-ES.zip" ]]; then
    log "Attempting LENS download from HuggingFace mirror"
    set +e
    wget -c -O "${DATA_DIR}/lens/ImageNet-ES.zip" "https://huggingface.co/datasets/Edw2n/ImageNet-ES/resolve/main/ImageNet-ES.zip"
    lens_status=$?
    set -e
    if [[ $lens_status -ne 0 ]]; then
      log "WARNING: LENS download failed. Place dataset manually under ${DATA_DIR}/lens/ImageNet-ES-Diverse"
    fi
  fi

  if [[ -f "${DATA_DIR}/lens/ImageNet-ES.zip" && ! -d "${DATA_DIR}/lens/ImageNet-ES" && ! -d "${DATA_DIR}/lens/ImageNet-ES-Diverse" ]]; then
    log "Extracting LENS archive"
    set +e
    python3 -c "import zipfile; zipfile.ZipFile('${DATA_DIR}/lens/ImageNet-ES.zip').extractall('${DATA_DIR}/lens')"
    unzip_status=$?
    set -e
    if [[ $unzip_status -ne 0 ]]; then
      log "WARNING: Could not extract ImageNet-ES.zip automatically."
    fi
  fi

  if [[ -d "${DATA_DIR}/lens/ImageNet-ES" && ! -d "${DATA_DIR}/lens/ImageNet-ES-Diverse" ]]; then
    log "Normalizing LENS folder name: ImageNet-ES -> ImageNet-ES-Diverse"
    mv "${DATA_DIR}/lens/ImageNet-ES" "${DATA_DIR}/lens/ImageNet-ES-Diverse"
  fi

  if [[ -d "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-test" && ! -d "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-diverse-test" ]]; then
    log "Normalizing split name: es-test -> es-diverse-test"
    ln -sfn es-test "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-diverse-test"
  fi

  if [[ ! -d "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-diverse-test" ]]; then
    log "WARNING: LENS structure still missing expected es-diverse-test path."
  fi
fi

if [[ -d "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-diverse-test" ]]; then
  log "Creating lens split"
  python3 "${ROOT_DIR}/scripts/create_lens_split.py"
fi

# 9) ImageNet class index file
if [[ -f "${DATA_DIR}/imagenet_class_index.json" ]]; then
  log "imagenet_class_index.json already present."
else
  download_file "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json" "${DATA_DIR}/imagenet_class_index.json"
fi

# 10) Verification
log "=== Verification ==="
for d in \
  "${DATA_DIR}/cifar10/cifar-10-batches-py" \
  "${DATA_DIR}/cifar10_c/CIFAR-10-C" \
  "${DATA_DIR}/screwset_split/train" \
  "${DATA_DIR}/screwset_split/validation" \
  "${DATA_DIR}/screwset_split/test" \
  "${DATA_DIR}/screwset_c" \
  "${DATA_DIR}/imagenet-a" \
  "${DATA_DIR}/imagenet-val" \
  "${DATA_DIR}/imagenet-c" \
  "${DATA_DIR}/lens/ImageNet-ES-Diverse/es-diverse-test"; do
  if [[ -d "$d" ]]; then
    echo "  ✓ $d"
  else
    echo "  ✗ MISSING: $d"
  fi
done

echo ""
echo "CIFAR-10-C corruptions: $(ls "${DATA_DIR}"/cifar10_c/CIFAR-10-C/*.npy 2>/dev/null | wc -l) files"
echo "ScrewSet classes:       $(ls "${DATA_DIR}"/screwset_split/train/ 2>/dev/null | wc -l) classes"
echo "ScrewSet-C types:       $(ls "${DATA_DIR}"/screwset_c/ 2>/dev/null | wc -l) corruption types"
echo "ImageNet-A classes:     $(ls "${DATA_DIR}"/imagenet-a/ 2>/dev/null | wc -l) classes"
echo "ImageNet-val classes:   $(ls "${DATA_DIR}"/imagenet-val/ 2>/dev/null | wc -l) classes"
echo "ImageNet-C types:       $(find "${DATA_DIR}"/imagenet-c -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) corruption types"

log "=== Dataset bootstrap finished ==="
