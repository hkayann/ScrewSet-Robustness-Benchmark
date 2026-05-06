# Phase 1: FP32 Baseline Results

**Date:** February 16–18, 2026
**Precision:** FP32 (full precision, no quantization)
**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)
**Framework:** PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, timm 1.0.24

> Canonical cross-phase summary is maintained in `results/ALL_RESULTS.md`. This phase file is a detailed breakdown of Phase 1 only.

---

## 1. Experimental Setup

### 1.1 Models

Eight CNN architectures were evaluated, spanning a range of parameter counts from 0.7M to 22.3M:

| # | Model | Source | Parameters | Checkpoint Size |
|---|-------|--------|------------|-----------------|
| 1 | ResNet-18 | torchvision | 11,191,262 | 43 MB |
| 2 | SqueezeNet 1.1 | torchvision | 727,626 | 2.8 MB |
| 3 | MobileNetV3-Large | torchvision | 4,239,288 | 17 MB |
| 4 | ShuffleNetV2 x1.0 | torchvision | 1,280,090 | 5.0 MB |
| 5 | MobileNetV4-Conv-Small | timm | 2,530,968 | 9.8 MB |
| 6 | EfficientNetV2-RW-S | timm | 22,327,186 | 86 MB |
| 7 | GhostNetV2-100 | timm | 4,951,224 | 20 MB |
| 8 | ConvNeXtV2-Atto | timm | 3,390,610 | 13 MB |

### 1.2 Datasets

| Dataset | Classes | Train | Val | Test | Corruption Variants | Task |
|---------|---------|-------|-----|------|---------------------|------|
| CIFAR-10 | 10 | 45,000 | 5,000 | 10,000 | 19 (CIFAR-10-C) | Train from pretrained |
| ScrewSet | 40 | variable | variable | variable | 6 (ScrewSet-C) | Train from pretrained |
| ImageNet-A | 200 | -- | -- | 7,500 | 0 | Eval-only (pretrained) |
| ImageNet Val | 1,000 | -- | -- | 50,000 | 0 | Eval-only (pretrained) |
| ImageNet-C | 1,000 | -- | -- | 50,000×95 | 19 corruptions × 5 severities | Eval-only (pretrained) |
| Lens (ImageNet-ES) | 200 | 7,200 | 800 | 1,000 | 192 | Fine-tune from pretrained |

### 1.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Max Epochs | 20 |
| Early Stopping Patience | 5 (on validation loss) |
| Random Seed | 42 |
| Pretrained Weights | ImageNet-1K |

All models were initialized with ImageNet-1K pretrained weights. The final classification head was replaced to match each dataset's number of classes (except ImageNet-A, which uses the original 1000-class head mapped to the 200-class subset).

---

## 2. Results by Dataset

### 2.1 CIFAR-10

All models were fine-tuned from ImageNet pretrained weights on CIFAR-10 (10 classes), then evaluated on the clean test set and 19 corruption types from CIFAR-10-C (averaged across all severity levels).

**Input:** 224x224, normalized with CIFAR-10 statistics (mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]).

| Model | Val Acc | Test Acc | Mean Corrupt Acc | Robustness Drop |
|-------|---------|----------|------------------|-----------------|
| EfficientNetV2-RW-S | 86.80% | 85.29% | 73.33% | 11.96 pp |
| GhostNetV2-100 | 85.40% | 85.01% | 67.05% | 17.96 pp |
| MobileNetV3-Large | 86.66% | 84.94% | 67.91% | 17.03 pp |
| ResNet-18 | 84.40% | 83.21% | 63.78% | 19.43 pp |
| ShuffleNetV2 x1.0 | 82.78% | 82.01% | 67.05% | 14.96 pp |
| ConvNeXtV2-Atto | 81.96% | 80.26% | 73.04% | 7.22 pp |
| SqueezeNet 1.1 | 77.94% | 77.74% | 59.44% | 18.30 pp |
| MobileNetV4-Conv-Small | 76.86% | 76.48% | 66.66% | 9.82 pp |

#### CIFAR-10-C Per-Corruption Breakdown (Test Accuracy %)

| Corruption | ResNet-18 | SqueezeNet | MobileNetV3 | ShuffleNetV2 | MobileNetV4 | EfficientNetV2 | GhostNetV2 | ConvNeXtV2 |
|------------|-----------|------------|-------------|--------------|-------------|-----------------|------------|------------|
| brightness | 81.19 | 71.68 | 83.03 | 78.41 | 70.69 | 81.90 | 82.57 | 77.24 |
| contrast | 58.59 | 44.83 | 60.45 | 55.18 | 49.89 | 64.06 | 61.70 | 58.64 |
| defocus_blur | 68.49 | 66.37 | 71.73 | 75.60 | 69.58 | 79.02 | 75.31 | 77.34 |
| elastic_transform | 68.85 | 65.06 | 71.50 | 74.09 | 68.09 | 76.87 | 74.65 | 74.21 |
| fog | 72.58 | 62.16 | 75.92 | 71.20 | 62.02 | 78.74 | 74.57 | 71.07 |
| frost | 64.33 | 58.61 | 72.61 | 69.29 | 64.29 | 76.33 | 68.46 | 72.90 |
| gaussian_blur | 62.11 | 61.84 | 65.18 | 72.40 | 67.08 | 76.34 | 70.38 | 75.97 |
| gaussian_noise | 47.55 | 44.76 | 48.58 | 46.82 | 63.31 | 60.57 | 40.08 | 68.08 |
| glass_blur | 38.96 | 52.01 | 53.71 | 64.40 | 68.80 | 65.50 | 55.75 | 74.10 |
| impulse_noise | 51.80 | 34.78 | 56.67 | 38.43 | 61.12 | 56.91 | 50.12 | 70.26 |
| jpeg_compression | 73.89 | 71.85 | 77.05 | 76.43 | 73.49 | 79.05 | 77.62 | 75.29 |
| motion_blur | 57.80 | 55.76 | 66.17 | 68.87 | 63.37 | 75.74 | 69.61 | 71.45 |
| pixelate | 72.51 | 73.44 | 76.49 | 78.57 | 74.56 | 78.88 | 76.55 | 78.67 |
| saturate | 78.13 | 71.57 | 80.69 | 75.82 | 69.80 | 79.71 | 81.60 | 74.35 |
| shot_noise | 56.97 | 51.40 | 57.08 | 55.90 | 67.71 | 67.31 | 48.22 | 71.65 |
| snow | 67.22 | 61.72 | 73.36 | 70.11 | 66.96 | 75.46 | 71.87 | 73.81 |
| spatter | 71.48 | 67.82 | 75.91 | 72.49 | 71.64 | 77.06 | 74.71 | 75.57 |
| speckle_noise | 58.60 | 52.18 | 58.48 | 58.62 | 68.00 | 68.37 | 49.68 | 71.83 |
| zoom_blur | 60.75 | 61.58 | 65.59 | 71.28 | 66.04 | 75.38 | 70.59 | 75.26 |

---

### 2.2 ScrewSet

All models were fine-tuned from ImageNet pretrained weights on ScrewSet (40 screw classes), then evaluated on the clean test set and 6 domain-specific corruption types from ScrewSet-C.

**Input:** 240x320, normalized with ScrewSet statistics (mean=[0.775, 0.7343, 0.6862], std=[0.0802, 0.0838, 0.0871]).

| Model | Val Acc | Test Acc | Mean Corrupt Acc | Robustness Drop |
|-------|---------|----------|------------------|-----------------|
| ResNet-18 | 97.95% | 98.02% | 9.23% | 88.79 pp |
| ShuffleNetV2 x1.0 | 96.19% | 96.18% | 12.30% | 83.87 pp |
| MobileNetV4-Conv-Small | 96.29% | 96.03% | 7.80% | 88.23 pp |
| MobileNetV3-Large | 95.68% | 95.95% | 11.08% | 84.87 pp |
| EfficientNetV2-RW-S | 95.44% | 95.45% | 11.28% | 84.18 pp |
| SqueezeNet 1.1 | 94.79% | 95.09% | 16.63% | 78.46 pp |
| GhostNetV2-100 | 94.75% | 95.02% | 11.63% | 83.40 pp |
| ConvNeXtV2-Atto | 91.71% | 91.67% | 8.29% | 83.38 pp |

#### ScrewSet-C Per-Corruption Breakdown (Test Accuracy %)

| Corruption | ResNet-18 | SqueezeNet | MobileNetV3 | ShuffleNetV2 | MobileNetV4 | EfficientNetV2 | GhostNetV2 | ConvNeXtV2 |
|------------|-----------|------------|-------------|--------------|-------------|-----------------|------------|------------|
| multi_object | 18.05 | 25.62 | 23.91 | 25.00 | 18.36 | 25.16 | 21.41 | 17.58 |
| occlusion_bottom_right | 2.42 | 2.19 | 2.50 | 2.73 | 2.50 | 2.27 | 2.50 | 2.50 |
| occlusion_top_left | 2.50 | 2.19 | 2.50 | 3.83 | 2.73 | 2.50 | 2.50 | 4.22 |
| reflection | 2.66 | 11.64 | 2.50 | 2.66 | 5.08 | 2.50 | 4.61 | 9.53 |
| scrap_paper | 23.83 | 45.31 | 31.72 | 34.84 | 9.84 | 32.73 | 33.28 | 5.55 |
| shadow | 5.94 | 12.81 | 3.36 | 4.77 | 8.28 | 2.50 | 5.47 | 10.39 |

**Note:** Occlusion corruptions (top-left and bottom-right) reduce all models to near-random chance (2.50% = 1/40 classes). These corruptions are extremely destructive. The multi_object and scrap_paper corruptions are relatively less severe but still cause massive accuracy drops.

---

### 2.3 ImageNet-A

All models were evaluated in eval-only mode using their original ImageNet-1K pretrained weights, mapped to the 200-class ImageNet-A subset. No fine-tuning was performed. ImageNet-A is an adversarially filtered dataset containing natural images that cause classifiers to fail.

**Input:** Resize to 256, center crop to 224, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

| Model | ImageNet-A Accuracy |
|-------|---------------------|
| EfficientNetV2-RW-S | 9.24% |
| ConvNeXtV2-Atto | 2.61% |
| GhostNetV2-100 | 1.73% |
| MobileNetV3-Large | 1.68% |
| MobileNetV4-Conv-Small | 1.21% |
| ShuffleNetV2 x1.0 | 0.64% |
| ResNet-18 | 0.29% |
| SqueezeNet 1.1 | 0.28% |

All accuracies are extremely low, which is expected behavior. ImageNet-A is specifically curated to contain images that fool pretrained models. The random-chance baseline for 200 classes is 0.50%. Only EfficientNetV2-RW-S achieves meaningfully above chance, consistent with its larger capacity (22.3M params).

---

### 2.4 ImageNet Validation (Clean)

All models were evaluated in eval-only mode using their original ImageNet-1K pretrained weights on the full 50,000-image ILSVRC-2012 validation set (1,000 classes). This establishes the clean baseline for ImageNet-C robustness analysis.

**Input:** Resize to 256, center crop to 224, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

| Model | Top-1 Accuracy | Loss |
|-------|----------------|------|
| EfficientNetV2-RW-S | **81.03%** | 0.8297 |
| ConvNeXtV2-Atto | 76.30% | 1.0168 |
| GhostNetV2-100 | 75.13% | 1.0559 |
| MobileNetV3-Large | 74.06% | 1.1028 |
| MobileNetV4-Conv-Small | 73.74% | 1.1058 |
| ResNet-18 | 69.76% | 1.2469 |
| ShuffleNetV2 x1.0 | 69.34% | 1.2767 |
| SqueezeNet 1.1 | 58.18% | 1.8527 |

All results closely match published reference numbers, confirming correct data loading and preprocessing. EfficientNetV2-RW-S leads with 81.03%, consistent with its 22.3M parameter capacity. SqueezeNet 1.1 (0.73M params) trails at 58.18%.

---

### 2.5 ImageNet-C

All models were evaluated in eval-only mode using their original ImageNet-1K pretrained weights on all 19 corruption types at all 5 severity levels from ImageNet-C (Hendrycks & Dietterich, 2019). Each evaluation uses the same 50,000 images as ImageNet validation but with applied corruptions, yielding 95 separate evaluations per model.

**Input:** Resize to 256, center crop to 224, ImageNet normalization.

**Corruption groups:** noise (3), blur (4), weather (4), digital (4) = 15 standard corruptions, plus 4 extra corruptions (speckle_noise, gaussian_blur, spatter, saturate).

| Model | Clean Acc | Mean Acc (15 std) | mCE (15 std) | Mean Acc (all 19) | mCE (all 19) | Drop (pp) |
|-------|-----------|-------------------|--------------|-------------------|--------------|-----------|
| EfficientNetV2-RW-S | 81.03% | **56.92%** | **0.4308** | **58.32%** | **0.4168** | 24.12 |
| ConvNeXtV2-Atto | 76.30% | 43.82% | 0.5618 | 45.37% | 0.5463 | 32.48 |
| GhostNetV2-100 | 75.13% | 39.62% | 0.6038 | 41.37% | 0.5863 | 35.52 |
| MobileNetV3-Large | 74.06% | 38.71% | 0.6129 | 40.46% | 0.5954 | 35.35 |
| MobileNetV4-Conv-Small | 73.74% | 37.60% | 0.6240 | 39.50% | 0.6050 | 36.14 |
| ResNet-18 | 69.76% | 31.73% | 0.6827 | 33.18% | 0.6682 | 38.03 |
| ShuffleNetV2 x1.0 | 69.34% | 28.61% | 0.7139 | 30.02% | 0.6998 | 40.74 |
| SqueezeNet 1.1 | 58.18% | 18.34% | 0.8166 | 19.59% | 0.8041 | 39.84 |

**Note:** mCE (mean Corruption Error) = 1 − mean accuracy across corruptions. Lower mCE indicates greater robustness. "Drop" = clean accuracy minus mean corrupted accuracy (15 std).

#### ImageNet-C Per-Corruption Breakdown (Mean Accuracy % across 5 Severity Levels)

| Corruption | ResNet-18 | SqueezeNet | MobileNetV3 | MobileNetV4 | ShuffleNetV2 | EfficientNetV2 | GhostNetV2 | ConvNeXtV2 |
|------------|-----------|------------|-------------|-------------|--------------|-----------------|------------|------------|
| gaussian_noise | 23.12 | 8.60 | 33.44 | 30.77 | 18.88 | 56.56 | 36.15 | 40.61 |
| shot_noise | 21.39 | 8.65 | 32.73 | 30.29 | 17.74 | 55.05 | 34.83 | 39.36 |
| impulse_noise | 19.25 | 5.10 | 33.16 | 28.35 | 13.80 | 56.65 | 36.10 | 37.74 |
| defocus_blur | 27.92 | 15.82 | 32.36 | 34.06 | 25.47 | 48.83 | 33.09 | 35.51 |
| glass_blur | 22.86 | 14.22 | 25.69 | 21.80 | 22.58 | 38.04 | 25.22 | 28.38 |
| motion_blur | 29.50 | 19.44 | 36.75 | 36.77 | 28.77 | 54.65 | 37.00 | 41.21 |
| zoom_blur | 29.43 | 20.05 | 31.62 | 28.72 | 27.93 | 49.03 | 32.68 | 34.06 |
| snow | 23.75 | 12.36 | 28.46 | 28.75 | 22.06 | 55.21 | 33.35 | 38.40 |
| frost | 27.92 | 14.22 | 30.95 | 31.32 | 25.60 | 57.19 | 32.76 | 38.31 |
| fog | 33.56 | 18.64 | 41.12 | 42.95 | 30.69 | 59.07 | 40.71 | 46.97 |
| brightness | 58.64 | 42.35 | 63.51 | 63.94 | 55.16 | 75.48 | 65.37 | 67.46 |
| contrast | 30.58 | 14.58 | 46.05 | 45.35 | 24.67 | 62.47 | 37.53 | 51.72 |
| elastic_transform | 39.47 | 28.66 | 43.00 | 42.13 | 40.99 | 54.46 | 43.67 | 47.47 |
| pixelate | 42.00 | 25.78 | 48.41 | 46.04 | 35.53 | 64.16 | 50.61 | 52.77 |
| jpeg_compression | 46.54 | 26.69 | 53.44 | 52.81 | 39.23 | 66.90 | 55.18 | 57.32 |
| *speckle_noise* | 28.33 | 14.11 | 42.10 | 39.35 | 24.33 | 61.67 | 42.59 | 46.72 |
| *gaussian_blur* | 31.86 | 19.33 | 35.87 | 38.23 | 29.45 | 51.62 | 36.43 | 39.67 |
| *spatter* | 41.53 | 27.09 | 49.33 | 45.90 | 37.95 | 66.94 | 51.49 | 53.25 |
| *saturate* | 52.79 | 36.48 | 60.84 | 62.91 | 49.64 | 74.03 | 61.32 | 65.12 |

*Italicized corruptions are from the "extra" set (not part of the standard 15-corruption benchmark).*

**Observations:**
- **Brightness** is the least damaging corruption across all models, with EfficientNetV2-RW-S retaining 75.48% accuracy.
- **Noise corruptions** (impulse, shot, gaussian) are the most destructive, with SqueezeNet dropping to 5-9% accuracy.
- **EfficientNetV2-RW-S** is the most robust model on every single corruption type, often by a wide margin.
- **ConvNeXtV2-Atto** is the second most robust model, despite having only 3.39M parameters (vs EfficientNetV2's 22.3M), suggesting its architectural design contributes to corruption robustness.
- Glass blur is universally the hardest blur corruption, while brightness and saturate are the easiest corruptions overall.

---

### 2.6 Lens (ImageNet-ES)

All models were fine-tuned from ImageNet pretrained weights on the Lens subset of ImageNet-ES (200 classes, 7,200 train / 800 val / 1,000 test images), then evaluated on 192 corruption variants organized into two corruption groups: auto_exposure (30 variants across 6 lighting levels x 5 parameter values) and param_control (162 variants across 6 parameter sets x 27 parameter combinations).

**Input:** Resize to 256, center crop to 224, ImageNet normalization.

| Model | Val Acc | Test Acc | Mean Corrupt Acc | Robustness Drop |
|-------|---------|----------|------------------|-----------------|
| EfficientNetV2-RW-S | 83.12% | 82.20% | 15.40% | 66.80 pp |
| GhostNetV2-100 | 73.38% | 70.20% | 8.32% | 61.88 pp |
| MobileNetV3-Large | 70.00% | 62.60% | 6.63% | 55.97 pp |
| MobileNetV4-Conv-Small | 57.38% | 53.60% | 2.81% | 50.79 pp |
| ShuffleNetV2 x1.0 | 54.75% | 53.10% | 3.37% | 49.73 pp |
| ResNet-18 | 54.00% | 50.60% | 4.05% | 46.55 pp |
| ConvNeXtV2-Atto | 1.25% | 1.20% | 0.70% | 0.50 pp |
| SqueezeNet 1.1 | 0.50% | 0.50% | 0.50% | 0.00 pp |

#### Lens Corruption Group Breakdown (Mean Accuracy %)

| Corruption Group | Variants | ResNet-18 | SqueezeNet | MobileNetV3 | ShuffleNetV2 | MobileNetV4 | EfficientNetV2 | GhostNetV2 | ConvNeXtV2 |
|------------------|----------|-----------|------------|-------------|--------------|-------------|-----------------|------------|------------|
| auto_exposure | 30 | 5.72 | 0.50 | 8.74 | 4.80 | 3.51 | 22.39 | 12.33 | 1.03 |
| param_control | 162 | 3.74 | 0.50 | 6.24 | 3.11 | 2.68 | 14.10 | 7.58 | 0.64 |

**Note:** Two models effectively failed to learn on Lens: SqueezeNet 1.1 (0.50% test, equal to random chance for 200 classes) and ConvNeXtV2-Atto (1.20% test). SqueezeNet likely lacks capacity for 200 classes given its 0.7M parameters. ConvNeXtV2-Atto likely required a different learning rate or longer warm-up to converge with the default Adam/LR=0.001 configuration.

---

## 3. Cross-Dataset Analysis

### 3.1 Robustness Drop Summary

Robustness drop = Clean test accuracy minus mean corrupted accuracy, measured in percentage points (pp). Higher values indicate greater vulnerability to corruption.

| Model | CIFAR-10 Drop | ScrewSet Drop | ImageNet-C Drop | Lens Drop |
|-------|---------------|---------------|-----------------|-----------|
| ResNet-18 | 19.43 pp | 88.79 pp | 38.03 pp | 46.55 pp |
| SqueezeNet 1.1 | 18.30 pp | 78.46 pp | 39.84 pp | 0.00 pp * |
| MobileNetV3-Large | 17.03 pp | 84.87 pp | 35.35 pp | 55.97 pp |
| ShuffleNetV2 x1.0 | 14.96 pp | 83.87 pp | 40.74 pp | 49.73 pp |
| MobileNetV4-Conv-Small | 9.82 pp | 88.23 pp | 36.14 pp | 50.79 pp |
| EfficientNetV2-RW-S | 11.96 pp | 84.18 pp | 24.12 pp | 66.80 pp |
| GhostNetV2-100 | 17.96 pp | 83.40 pp | 35.52 pp | 61.88 pp |
| ConvNeXtV2-Atto | 7.22 pp | 83.38 pp | 32.48 pp | 0.50 pp * |

\* SqueezeNet and ConvNeXtV2-Atto show near-zero Lens drop because they failed to learn the task (test accuracy at or near random chance), not because they are robust.

### 3.2 Relative Robustness Ratio

Robustness ratio = Mean corrupted accuracy / Clean test accuracy. Values closer to 1.0 indicate better relative robustness. Only computed for models that successfully learned each task.

| Model | CIFAR-10 Ratio | ScrewSet Ratio | ImageNet-C Ratio | Lens Ratio |
|-------|----------------|----------------|------------------|------------|
| ResNet-18 | 0.766 | 0.094 | 0.455 | 0.080 |
| SqueezeNet 1.1 | 0.765 | 0.175 | 0.315 | -- |
| MobileNetV3-Large | 0.800 | 0.116 | 0.523 | 0.106 |
| ShuffleNetV2 x1.0 | 0.818 | 0.128 | 0.413 | 0.063 |
| MobileNetV4-Conv-Small | 0.872 | 0.081 | 0.510 | 0.052 |
| EfficientNetV2-RW-S | 0.860 | 0.118 | 0.702 | 0.187 |
| GhostNetV2-100 | 0.789 | 0.122 | 0.527 | 0.119 |
| ConvNeXtV2-Atto | 0.910 | 0.090 | 0.574 | -- |

---

## 4. Comprehensive Comparison Table

The table below consolidates all Phase 1 results across all 8 models and 6 datasets. Accuracy values are percentages. Best result per column is marked in bold text. "Drop" is in percentage points. ImageNet-A and ImageNet Val have no corruption variants.

| Model | Params | Size | CIFAR-10 Clean | CIFAR-10 Corrupt | Drop | ScrewSet Clean | ScrewSet Corrupt | Drop | ImageNet-A | ImageNet Val | ImageNet-C (15 std) | ImageNet-C Drop | mCE | Lens Clean | Lens Corrupt | Drop |
|-------|--------|------|----------------|------------------|------|----------------|------------------|------|------------|--------------|---------------------|-----------------|-----|------------|--------------|------|
| ResNet-18 | 11.19M | 43 MB | 83.21 | 63.78 | 19.43 | **98.02** | 9.23 | 88.79 | 0.29 | 69.76 | 31.73 | 38.03 | 0.683 | 50.60 | 4.05 | 46.55 |
| SqueezeNet 1.1 | 0.73M | 2.8 MB | 77.74 | 59.44 | 18.30 | 95.09 | **16.63** | 78.46 | 0.28 | 58.18 | 18.34 | 39.84 | 0.817 | 0.50 | 0.50 | 0.00 |
| MobileNetV3-Large | 4.24M | 17 MB | 84.94 | 67.91 | 17.03 | 95.95 | 11.08 | 84.87 | 1.68 | 74.06 | 38.71 | 35.35 | 0.613 | 62.60 | 6.63 | 55.97 |
| ShuffleNetV2 x1.0 | 1.28M | 5.0 MB | 82.01 | 67.05 | 14.96 | 96.18 | 12.30 | 83.87 | 0.64 | 69.34 | 28.61 | 40.74 | 0.714 | 53.10 | 3.37 | 49.73 |
| MobileNetV4-Conv-Small | 2.53M | 9.8 MB | 76.48 | 66.66 | 9.82 | 96.03 | 7.80 | 88.23 | 1.21 | 73.74 | 37.60 | 36.14 | 0.624 | 53.60 | 2.81 | 50.79 |
| EfficientNetV2-RW-S | 22.33M | 86 MB | **85.29** | **73.33** | 11.96 | 95.45 | 11.28 | 84.18 | **9.24** | **81.03** | **56.92** | **24.12** | **0.431** | **82.20** | **15.40** | 66.80 |
| GhostNetV2-100 | 4.95M | 20 MB | 85.01 | 67.05 | 17.96 | 95.02 | 11.63 | 83.40 | 1.73 | 75.13 | 39.62 | 35.52 | 0.604 | 70.20 | 8.32 | 61.88 |
| ConvNeXtV2-Atto | 3.39M | 13 MB | 80.26 | 73.04 | **7.22** | 91.67 | 8.29 | 83.38 | 2.61 | 76.30 | 43.82 | 32.48 | 0.562 | 1.20 | 0.70 | 0.50 |

---

## 5. Key Findings

### 5.1 Clean Accuracy

- **EfficientNetV2-RW-S** achieves the highest clean accuracy on 4 out of 6 datasets (CIFAR-10: 85.29%, ImageNet-A: 9.24%, ImageNet Val: 81.03%, Lens: 82.20%). This is consistent with it being the largest model at 22.3M parameters.
- **ResNet-18** leads on ScrewSet (98.02%), demonstrating strong performance on the industrial screw classification task despite being a relatively simple architecture.
- On **ImageNet Val**, all models match their published reference accuracies, confirming correct implementation. Ranking follows model capacity: EfficientNetV2 (81.03%) > ConvNeXtV2 (76.30%) > GhostNetV2 (75.13%) > MobileNetV3 (74.06%) > MobileNetV4 (73.74%) > ResNet-18 (69.76%) > ShuffleNetV2 (69.34%) > SqueezeNet (58.18%).
- The smallest model, **SqueezeNet 1.1** (0.73M params), consistently ranks last or near-last on clean accuracy but remains competitive on ScrewSet (95.09%).

### 5.2 Corruption Robustness

- On **CIFAR-10-C**, **ConvNeXtV2-Atto** shows the smallest robustness drop (7.22 pp) and the highest corruption-to-clean ratio (0.910), despite having only middling clean accuracy (80.26%). Its architectural design (depthwise convolutions, LayerNorm, GELU) appears to inherently confer robustness to common corruptions.
- On **ScrewSet-C**, all models suffer catastrophic degradation (78-89 pp drops). Occlusion corruptions reduce every model to near-random chance (2.50%), indicating that the models rely heavily on the full screw being visible. The multi_object and scrap_paper corruptions are less severe but still cause 75-90% accuracy loss.
- On **ImageNet-C**, **EfficientNetV2-RW-S** is the most robust model by a large margin (mCE = 0.431 vs next best ConvNeXtV2 at 0.562). It retains 70.2% of its clean accuracy under corruption, while SqueezeNet retains only 31.5%. Noise corruptions (impulse, shot, gaussian) are the most destructive, while brightness and saturate are the least. The rankings closely mirror clean accuracy, suggesting that on ImageNet-scale corruption benchmarks, model capacity is the dominant factor.
- On **Lens (ImageNet-ES)**, corruption robustness is uniformly poor across all successful models. Even the best performer (EfficientNetV2-RW-S) drops from 82.20% to 15.40% under corruption. The auto_exposure corruptions are marginally less damaging than param_control corruptions.

### 5.3 Capacity vs. Robustness

- There is no simple monotonic relationship between model size and corruption robustness on CIFAR-10-C. ConvNeXtV2-Atto (3.39M params) is the most robust on CIFAR-10-C, while larger models like ResNet-18 (11.19M) show larger robustness drops.
- However, on **ImageNet-C** the relationship is much clearer: robustness is strongly correlated with model capacity and clean accuracy. The robustness ratio ranges from 0.315 (SqueezeNet) to 0.702 (EfficientNetV2), following a near-monotonic ordering by clean accuracy. This suggests that the architectural robustness advantages seen on CIFAR-10 (e.g., ConvNeXtV2) diminish when evaluated on the more challenging ImageNet-scale corruptions.
- Two models (SqueezeNet 1.1 and ConvNeXtV2-Atto) failed to converge on the Lens task, suggesting that very small models or architectures with specific inductive biases may require different hyperparameter tuning for 200-class fine-tuning.

### 5.4 ImageNet-A

- All lightweight models perform at or near random chance on ImageNet-A (0.28% to 2.61% for 200 classes). Only EfficientNetV2-RW-S (9.24%) demonstrates any meaningful ability to handle adversarially challenging natural images.
- This result confirms that ImageNet-A remains an extremely difficult benchmark for small-to-medium CNNs and that model scale matters significantly for natural adversarial robustness.

### 5.5 ImageNet-C vs. CIFAR-10-C Robustness Comparison

The corruption robustness rankings differ between CIFAR-10-C and ImageNet-C:
- On **CIFAR-10-C**, ConvNeXtV2-Atto achieves the best robustness ratio (0.910), demonstrating strong architectural robustness at low resolution.
- On **ImageNet-C**, EfficientNetV2-RW-S dominates (0.702 ratio), and ConvNeXtV2 drops to second (0.574 ratio).
- This divergence suggests that architectural robustness (e.g., LayerNorm, depthwise convolutions) matters more for low-complexity tasks (CIFAR-10), while raw capacity matters more for high-complexity corruption robustness (ImageNet).

---

## 6. File Inventory

All result JSON files and model checkpoints are stored in `results/phase1/`.

### JSON Results (48 files)

```
results/phase1/
    {model}_{dataset}_baselines.json    (8 models x 6 datasets = 48 files)
```

### Model Checkpoints (24 files)

Checkpoints are saved for datasets that involve training (CIFAR-10, ScrewSet, Lens). ImageNet-A, ImageNet Val, and ImageNet-C are eval-only and have no checkpoints.

```
results/phase1/models/
    {model}_{dataset}_best.pth          (8 models x 3 datasets = 24 files)
```

### Training Script

```
scripts/phase1/phase1_baselines.py
```

---

## 7. Reproducibility

To reproduce any single experiment:

```bash
conda activate rapids-25.10
cd ~/ARCADE--Screwset
python scripts/phase1/phase1_baselines.py \
    --model <model_name> \
    --dataset <dataset> \
    --batch-size 256 \
    --num-epochs 20 \
    --learning-rate 0.001 \
    --patience 5
```

Where `<model_name>` is one of: `resnet18`, `squeezenet1_1`, `mobilenet_v3_large`, `shufflenet_v2_x1_0`, `mobilenetv4_conv_small`, `efficientnetv2_rw_s`, `ghostnetv2_100`, `convnextv2_atto`.

Where `<dataset>` is one of: `cifar10`, `screwset`, `imagenet_a`, `imagenet_val`, `imagenet_c`, `lens`.
