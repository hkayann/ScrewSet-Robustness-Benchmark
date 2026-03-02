# Dataset Setup Guide for ARCADE--Screwset

> **Give this file to your AI assistant on the target machine.**
> All datasets go under `<repo_root>/data/`. Budget **at least 200 GB** free disk (117 GB final + 62 GB temp tarballs).

---

## 0. Clone the repo & create the data directory

```bash
git clone https://github.com/hkayann/ARCADE--Screwset.git
cd ARCADE--Screwset
mkdir -p data
```

---

## 1. ScrewSet (clean) — `data/screwset_split/`

**Source:** Zenodo — <https://zenodo.org/records/16744219>

Download `screwset_split.tar.gz` and extract:

```bash
cd data
wget https://zenodo.org/records/16744219/files/screwset_split.tar.gz
tar xzf screwset_split.tar.gz
rm screwset_split.tar.gz
cd ..
```

**Expected structure** (~19 GB, 40 classes, 1536 images/class):
```
data/screwset_split/
├── train/          # 61,440 images
│   ├── 10#_19_4.6/
│   ├── 10#_38_4.6/
│   ├── 7#_16_Yellow/
│   ├── ...         # 40 class folders total
│   └── M6_20_Round/
├── validation/     # 20,480 images
│   └── (same 40 class folders)
└── test/           # (same 40 class folders)
```

---

## 2. ScrewSet-C (physical corruptions) — `data/screwset_c/`

**Source:** Same Zenodo — <https://zenodo.org/records/16744219>

Download `screwset_c.tar.gz` and extract:

```bash
cd data
wget https://zenodo.org/records/16744219/files/screwset_c.tar.gz
tar xzf screwset_c.tar.gz
rm screwset_c.tar.gz
cd ..
```

**Expected structure** (~4 GB, 6 corruption types, 1280 images each):
```
data/screwset_c/
├── screwset_multi_object/
├── screwset_occlusion_bottom_right/
├── screwset_occlusion_top_left/
├── screwset_reflection/
├── screwset_scrap_paper/
└── screwset_shadow/
```

Each corruption folder contains the same 40 class subfolders as `screwset_split/`.

---

## 3. CIFAR-10 — `data/cifar10/`

**Source:** Auto-download via torchvision, or <https://www.cs.toronto.edu/~kriz/cifar.html>

```bash
python3 -c "
from torchvision.datasets import CIFAR10
CIFAR10(root='data/cifar10', train=True, download=True)
CIFAR10(root='data/cifar10', train=False, download=True)
print('CIFAR-10 downloaded.')
"
```

**Expected structure** (~341 MB):
```
data/cifar10/
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── ...
    └── test_batch
```

---

## 4. CIFAR-10-C — `data/cifar10_c/CIFAR-10-C/`

**Source:** <https://zenodo.org/records/2535967/files/CIFAR-10-C.tar>

```bash
cd data
mkdir -p cifar10_c
wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
tar xf CIFAR-10-C.tar -C cifar10_c/
rm CIFAR-10-C.tar
# Create a convenience symlink used by some scripts
ln -sf cifar10_c/CIFAR-10-C CIFAR-10-C
cd ..
```

**Expected structure** (~5.5 GB, 19 corruption .npy files + labels.npy):
```
data/cifar10_c/CIFAR-10-C/
├── brightness.npy
├── contrast.npy
├── defocus_blur.npy
├── elastic_transform.npy
├── fog.npy
├── frost.npy
├── gaussian_blur.npy
├── gaussian_noise.npy
├── glass_blur.npy
├── impulse_noise.npy
├── jpeg_compression.npy
├── labels.npy              ← shared labels for all corruptions
├── motion_blur.npy
├── pixelate.npy
├── saturate.npy
├── shot_noise.npy
├── snow.npy
├── spatter.npy
├── speckle_noise.npy
└── zoom_blur.npy
```

Each `.npy` contains 50,000 images (10,000 per severity × 5 severity levels).

---

## 5. ImageNet-A — `data/imagenet-a/`

**Source:** <https://github.com/hendrycks/natural-adv-examples>

```bash
cd data
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar xf imagenet-a.tar
rm imagenet-a.tar
cd ..
```

**Expected structure** (~666 MB, 200 classes, 7,501 images):
```
data/imagenet-a/
├── n01498041/
├── n01531178/
├── ...            # 200 WordNet ID folders (subset of ImageNet-1K)
└── README.txt
```

---

## 6. ImageNet Validation Set — `data/imagenet-val/`

**Source:** ILSVRC2012 validation set. Requires academic access.

**Option A — from HuggingFace (recommended):**
The repo includes a helper script:
```bash
python3 scripts/phase1/download_imagenet_val.py
```

**Option B — manual from ILSVRC2012:**
1. Get `ILSVRC2012_img_val.tar` from <https://image-net.org/download-images.php>
2. Extract and reorganize into class folders using the standard `valprep.sh` script:
   ```bash
   cd data && mkdir -p imagenet-val
   tar xf ILSVRC2012_img_val.tar -C imagenet-val/
   # Download and run the reorganization script:
   wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader/master/valprep.sh | bash
   cd ..
   ```

**Expected structure** (~6.4 GB, 1000 classes, 50,000 images):
```
data/imagenet-val/
├── n01440764/
├── n01443537/
├── ...         # 1000 WordNet ID folders
└── n15075141/
```

---

## 7. ImageNet-C — `data/imagenet-c/`

**Source:** Zenodo — <https://zenodo.org/records/2235448>

There are 5 tarballs to download. **Use `-4` flag if IPv6 causes timeouts.**

```bash
cd data
mkdir -p imagenet-c && cd imagenet-c

# Download all 5 tarballs (~72 GB total compressed)
wget -4 https://zenodo.org/records/2235448/files/blur.tar
wget -4 https://zenodo.org/records/2235448/files/digital.tar
wget -4 https://zenodo.org/records/2235448/files/extra.tar
wget -4 https://zenodo.org/records/2235448/files/noise.tar
wget -4 https://zenodo.org/records/2235448/files/weather.tar

# Extract each (they create corruption_type/ folders directly)
for f in blur.tar digital.tar extra.tar noise.tar weather.tar; do
    echo "Extracting $f ..."
    tar xf "$f"
    echo "$f done"
done

cd ../..
```

**Expected structure** (~72 GB extracted, 19 corruption types × 5 severities × 1000 classes):
```
data/imagenet-c/
├── brightness/
│   ├── 1/          # severity 1
│   │   ├── n01440764/   # 1000 class folders
│   │   └── ...
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
├── contrast/
│   └── (same 1/2/3/4/5 structure)
├── defocus_blur/
├── elastic_transform/
├── fog/
├── frost/
├── gaussian_blur/
├── gaussian_noise/
├── glass_blur/
├── impulse_noise/
├── jpeg_compression/
├── motion_blur/
├── pixelate/
├── saturate/
├── shot_noise/
├── snow/
├── spatter/
├── speckle_noise/
└── zoom_blur/
```

Each severity folder contains 1000 class folders with 50 images each (50,000 per severity).

---

## 8. ImageNet-ES / LENS — `data/lens/`

**Source:** <https://github.com/Arlene036/ImageNet-ES>

The dataset is called "ImageNet-ES Diverse". Download the full archive:

```bash
cd data
mkdir -p lens
# Download from the ImageNet-ES release (check their GitHub for the latest link)
# The archive should extract to: lens/ImageNet-ES-Diverse/
cd ..
```

**Expected structure** (~8.8 GB, 200 classes):
```
data/lens/
└── ImageNet-ES-Diverse/
    ├── es-diverse-test/
    │   ├── auto_exposure/
    │   │   ├── l1/
    │   │   │   ├── param_1/    # each param dir has ~200 class folders (wnids)
    │   │   │   ├── param_2/
    │   │   │   ├── ...
    │   │   │   └── param_5/
    │   │   ├── l2/
    │   │   ├── l3/
    │   │   ├── ...
    │   │   └── l6/
    │   └── param_control/
    │       ├── l1/ ... l6/     # same nesting as auto_exposure
    │       └── (each l*/param_*/ has class wnid folders)
    └── es-train/
        └── tin_no_resize_sample_removed/
            └── (class wnid folders with training images)
```

**Then create the LENS split** (used by phase1/phase2 scripts):
```bash
python3 scripts/create_lens_split.py
```

This creates `data/lens_split/` with train/validation/test/corrupted splits (symlinks, ~38 MB).

---

## 9. ImageNet class index file — `data/imagenet_class_index.json`

**Source:** PyTorch hub (auto-downloaded by torchvision, but useful to have on disk):

```bash
cd data
wget https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
cd ..
```

This maps integer indices 0–999 → (wnid, human-readable class name). Used by ImageNet-A evaluation for wnid→index mapping.

---

## 10. (Optional) ScrewSet-S — `data/screwset_s/`

> **Note:** Currently only severity 3 exists. This is NOT used by the active scripts.

If you have it, place the `.npy` corruption files here:
```
data/screwset_s/
├── brightness.npy
├── contrast.npy
├── ...               # 19 corruption .npy files
├── labels.npy
├── class_mapping.json
└── summary.json
```

Generate with: `python3 scripts/create_screwset_s.py`

---

## Quick verification

After downloading everything, verify your setup:

```bash
cd ARCADE--Screwset

echo "=== Checking all required datasets ==="

# Required directories
for d in data/cifar10/cifar-10-batches-py \
         data/cifar10_c/CIFAR-10-C \
         data/screwset_split/train \
         data/screwset_split/validation \
         data/screwset_split/test \
         data/screwset_c \
         data/imagenet-a \
         data/imagenet-val \
         data/imagenet-c \
         data/lens/ImageNet-ES-Diverse/es-diverse-test; do
    if [ -d "$d" ]; then
        echo "  ✓ $d"
    else
        echo "  ✗ MISSING: $d"
    fi
done

# Check counts
echo ""
echo "CIFAR-10-C corruptions: $(ls data/cifar10_c/CIFAR-10-C/*.npy 2>/dev/null | wc -l) files"
echo "ScrewSet classes:       $(ls data/screwset_split/train/ 2>/dev/null | wc -l) classes"
echo "ScrewSet-C types:       $(ls data/screwset_c/ 2>/dev/null | wc -l) corruption types"
echo "ImageNet-A classes:     $(ls data/imagenet-a/ 2>/dev/null | wc -l) classes"
echo "ImageNet-val classes:   $(ls data/imagenet-val/ 2>/dev/null | wc -l) classes"
echo "ImageNet-C types:       $(ls data/imagenet-c/ 2>/dev/null | grep -v tar | grep -v extracted | wc -l) corruption types"
```

Expected output:
```
  ✓ data/cifar10/cifar-10-batches-py
  ✓ data/cifar10_c/CIFAR-10-C
  ✓ data/screwset_split/train
  ✓ data/screwset_split/validation
  ✓ data/screwset_split/test
  ✓ data/screwset_c
  ✓ data/imagenet-a
  ✓ data/imagenet-val
  ✓ data/imagenet-c
  ✓ data/lens/ImageNet-ES-Diverse/es-diverse-test

CIFAR-10-C corruptions: 20 files
ScrewSet classes:       40 classes
ScrewSet-C types:       6 corruption types
ImageNet-A classes:     201 classes
ImageNet-val classes:   1000 classes
ImageNet-C types:       19 corruption types
```

---

## Environment setup

```bash
# Create conda environment
conda create -n arcade python=3.12 -y
conda activate arcade

# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install timm open_clip_torch tqdm numpy scipy pandas matplotlib seaborn

# For VLMs (Phase 3)
pip install transformers accelerate

# Brevitas (for quantization, if needed)
pip install git+https://github.com/Xilinx/brevitas.git
```

---

## Disk space summary

| Dataset | Directory | Extracted size | Download size |
|---------|-----------|---------------|---------------|
| CIFAR-10 | `data/cifar10/` | 341 MB | 170 MB |
| CIFAR-10-C | `data/cifar10_c/` | 5.5 GB | 2.7 GB |
| ScrewSet (clean) | `data/screwset_split/` | 19 GB | ~10 GB |
| ScrewSet-C | `data/screwset_c/` | 4 GB | ~2 GB |
| ImageNet-A | `data/imagenet-a/` | 666 MB | 650 MB |
| ImageNet val | `data/imagenet-val/` | 6.4 GB | 6.4 GB |
| ImageNet-C | `data/imagenet-c/` | 72 GB | 62 GB (5 tars) |
| ImageNet-ES (LENS) | `data/lens/` | 8.8 GB | ~5 GB |
| LENS split (symlinks) | `data/lens_split/` | 38 MB | — (generated) |
| **Total extracted** | | **~117 GB** | |
| **Peak during download** | | **~179 GB** | *(before deleting tarballs)* |

> **Important:** ImageNet-C alone needs ~134 GB during extraction (62 GB tarballs + 72 GB extracted).
> Delete the `.tar` files after extraction to reclaim 62 GB:
> ```bash
> rm data/imagenet-c/*.tar
> ```
> Budget **at least 200 GB free** before starting all downloads.
