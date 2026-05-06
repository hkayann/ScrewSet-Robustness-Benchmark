# Phase 2: FP32 Vision Transformer Baseline Results

**Date:** February 19–20, 2026  
**Precision:** FP32 (full precision, no quantization)  
**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)  
**Framework:** PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, timm 1.0.24

> Canonical cross-phase summary is maintained in `results/ALL_RESULTS.md`. This phase file is a detailed breakdown of Phase 2 only.

---

## 1. Experimental Setup

### 1.1 Models

Eight Vision Transformer architectures were evaluated, spanning a range of parameter counts from 5.6M to 28M:

| # | Model | Source | Family | Parameters | Pretrained Weights |
|---|-------|--------|--------|------------|-------------------|
| 1 | ViT-Ti/16 | timm | ViT | 5.7M | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` |
| 2 | ViT-S/16 | timm | ViT | 22M | `vit_small_patch16_224.augreg_in21k_ft_in1k` |
| 3 | DeiT-Ti/16 | timm | DeiT | 5.7M | `deit_tiny_patch16_224.fb_in1k` |
| 4 | DeiT-S/16 | timm | DeiT | 22M | `deit_small_patch16_224.fb_in1k` |
| 5 | Swin-T | timm | Swin | 28M | `swin_tiny_patch4_window7_224.ms_in1k` |
| 6 | MobileViT-S | timm | MobileViT | 5.6M | `mobilevit_s.cvnets_in1k` |
| 7 | EfficientFormer-L1 | timm | EfficientFormer | 12M | `efficientformer_l1.snap_dist_in1k` |
| 8 | ConvNeXt-T | timm | ConvNeXt | 28M | `convnext_tiny.fb_in1k` |

### 1.2 Model-Specific Preprocessing

Each model uses its timm-recommended preprocessing (normalization, input resolution, crop percentage):

| Model | Input Size | Resize | Crop % | Mean | Std |
|-------|-----------|--------|--------|------|-----|
| ViT-Ti/16 | 224×224 | 248 | 0.90 | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) |
| ViT-S/16 | 224×224 | 248 | 0.90 | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) |
| DeiT-Ti/16 | 224×224 | 248 | 0.90 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| DeiT-S/16 | 224×224 | 248 | 0.90 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| Swin-T | 224×224 | 248 | 0.90 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| MobileViT-S | 256×256 | 284 | 0.90 | (0.0, 0.0, 0.0) | (1.0, 1.0, 1.0) |
| EfficientFormer-L1 | 224×224 | 235 | 0.95 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| ConvNeXt-T | 224×224 | 256 | 0.875 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |

### 1.3 Datasets

| Dataset | Classes | Train | Val | Test | Corruption Variants | Task |
|---------|---------|-------|-----|------|---------------------|------|
| CIFAR-10 | 10 | 45,000 | 5,000 | 10,000 | 19 (CIFAR-10-C) | Fine-tune from pretrained |
| ScrewSet | 40 | variable | variable | variable | 6 (ScrewSet-C) | Fine-tune from pretrained |
| ImageNet-A | 200 | — | — | 7,500 | 0 | Eval-only (pretrained) |
| ImageNet Val | 1,000 | — | — | 50,000 | 0 | Eval-only (pretrained) |
| ImageNet-C | 1,000 | — | — | 50,000×95 | 19 corruptions × 5 severities | Eval-only (pretrained) |
| Lens (ImageNet-ES) | 200 | 7,200 | 800 | 1,000 | 192 | Fine-tune from pretrained |

### 1.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5×10⁻⁴ (peak) |
| Min LR | 1×10⁻⁶ |
| Weight Decay | 0.05 |
| Scheduler | Cosine annealing with linear warmup |
| Warmup Epochs | 5 |
| Batch Size | 256 (train), 512 (eval) |
| Max Epochs | 30 |
| Early Stopping Patience | 7 (on validation accuracy) |
| Mixed Precision | FP16 AMP |
| Gradient Clipping | max_norm = 1.0 |
| Random Seed | 42 |
| Pretrained Weights | ImageNet-1K (or ImageNet-21K → 1K for ViT) |

All models were initialized with pretrained weights. The final classification head was replaced by timm to match each dataset's number of classes. Eval-only datasets (ImageNet-A, ImageNet Val, ImageNet-C) used the pretrained 1000-class head directly.

---

## 2. Results by Dataset

### 2.1 CIFAR-10

All models were fine-tuned from pretrained weights on CIFAR-10 (10 classes), then evaluated on the clean test set and 19 corruption types from CIFAR-10-C (averaged across all severity levels).

| Model | Val Acc | Test Acc | Test Loss | Mean Corrupt Acc | Robustness Drop |
|-------|---------|----------|-----------|------------------|-----------------|
| ConvNeXt-T | 98.20% | 98.03% | 0.0965 | 86.29% | 11.74 pp |
| MobileViT-S | 97.70% | 97.30% | 0.1518 | 78.48% | 18.82 pp |
| ViT-S/16 | 97.14% | 97.18% | 0.0872 | 88.20% | 8.98 pp |
| DeiT-S/16 | 96.60% | 96.52% | 0.1093 | 87.29% | 9.23 pp |
| Swin-T | 96.50% | 96.10% | 0.1133 | 83.23% | 12.87 pp |
| EfficientFormer-L1 | 96.28% | 96.07% | 0.2623 | 75.74% | 20.33 pp |
| ViT-Ti/16 | 95.68% | 95.70% | 0.1340 | 81.37% | 14.33 pp |
| DeiT-Ti/16 | 93.60% | 93.15% | 0.2039 | 80.59% | 12.56 pp |

#### CIFAR-10-C Per-Corruption Breakdown (Test Accuracy %)

| Corruption | ViT-Ti | ViT-S | DeiT-Ti | DeiT-S | Swin-T | MobileViT-S | EffFormer-L1 | ConvNeXt-T |
|------------|--------|-------|---------|--------|--------|-------------|--------------|------------|
| brightness | 94.16 | 96.22 | 92.04 | 95.30 | 95.10 | 96.32 | 94.61 | 97.38 |
| contrast | 84.51 | 89.60 | 84.66 | 92.54 | 91.10 | 88.23 | 79.15 | 95.03 |
| defocus_blur | 91.72 | 93.59 | 85.45 | 91.40 | 90.16 | 91.17 | 85.83 | 93.75 |
| elastic_transform | 87.48 | 89.87 | 84.27 | 89.07 | 84.97 | 88.22 | 86.23 | 91.37 |
| fog | 89.15 | 91.30 | 87.76 | 92.50 | 92.06 | 92.82 | 92.24 | 95.98 |
| frost | 86.43 | 91.00 | 88.09 | 92.54 | 88.87 | 88.15 | 85.56 | 92.30 |
| gaussian_blur | 90.13 | 92.32 | 82.27 | 89.61 | 88.03 | 87.99 | 81.06 | 91.21 |
| gaussian_noise | 54.34 | 77.85 | 62.37 | 76.26 | 62.72 | 33.32 | 31.77 | 63.96 |
| glass_blur | 63.88 | 69.69 | 70.91 | 71.66 | 59.20 | 57.51 | 55.61 | 67.80 |
| impulse_noise | 73.11 | 88.94 | 70.95 | 81.76 | 79.47 | 59.92 | 53.26 | 67.42 |
| jpeg_compression | 79.14 | 84.75 | 79.19 | 82.76 | 81.46 | 78.66 | 82.15 | 83.98 |
| motion_blur | 84.75 | 88.46 | 80.64 | 86.76 | 85.08 | 87.13 | 81.85 | 91.46 |
| pixelate | 79.04 | 84.84 | 78.48 | 83.65 | 76.24 | 71.46 | 77.43 | 85.20 |
| saturate | 92.28 | 94.86 | 89.08 | 94.38 | 93.63 | 94.49 | 92.43 | 96.38 |
| shot_noise | 62.72 | 82.20 | 69.05 | 81.35 | 71.10 | 47.86 | 46.54 | 72.12 |
| snow | 87.91 | 91.70 | 86.68 | 92.49 | 90.29 | 90.97 | 87.51 | 93.44 |
| spatter | 90.59 | 93.39 | 87.32 | 93.25 | 91.13 | 92.85 | 90.50 | 94.42 |
| speckle_noise | 64.60 | 82.96 | 69.89 | 81.65 | 73.25 | 55.31 | 51.41 | 73.66 |
| zoom_blur | 90.10 | 92.27 | 82.12 | 89.56 | 87.60 | 88.69 | 83.97 | 92.61 |

---

### 2.2 ScrewSet

All models were fine-tuned from pretrained weights on ScrewSet (40 screw classes), then evaluated on the clean test set and 6 real-world corruption types from ScrewSet-C.

**Key finding:** All models achieve near-perfect clean accuracy (>99.4%) but suffer dramatic accuracy drops under real-world corruptions, with average corruption accuracy ranging from 16% to 45%.

| Model | Val Acc | Test Acc | Test Loss | Mean Corrupt Acc | Robustness Gap |
|-------|---------|----------|-----------|------------------|----------------|
| ConvNeXt-T | 99.98% | 99.98% | 0.0005 | 45.39% | 54.59 pp |
| Swin-T | 99.92% | 99.94% | 0.0020 | 33.39% | 66.55 pp |
| MobileViT-S | 100.00% | 100.00% | 0.0001 | 28.09% | 71.91 pp |
| ViT-S/16 | 99.75% | 99.79% | 0.0070 | 24.40% | 75.38 pp |
| ViT-Ti/16 | 99.64% | 99.66% | 0.0099 | 21.17% | 78.49 pp |
| DeiT-S/16 | 99.69% | 99.65% | 0.0107 | 19.48% | 80.17 pp |
| DeiT-Ti/16 | 99.57% | 99.55% | 0.0164 | 16.91% | 82.64 pp |
| EfficientFormer-L1 | 99.95% | 99.91% | 0.0028 | 16.15% | 83.77 pp |

#### ScrewSet-C Per-Corruption Breakdown (Accuracy %)

| Corruption | ViT-Ti | ViT-S | DeiT-Ti | DeiT-S | Swin-T | MobileViT-S | EffFormer-L1 | ConvNeXt-T |
|------------|--------|-------|---------|--------|--------|-------------|--------------|------------|
| multi_object | 45.23 | 49.77 | 45.86 | 44.06 | 57.03 | 42.03 | 43.36 | 53.52 |
| occlusion_bottom_right | 2.81 | 2.50 | 2.50 | 2.66 | 4.53 | 6.17 | 2.97 | 33.12 |
| occlusion_top_left | 3.83 | 3.59 | 2.50 | 3.67 | 12.03 | 6.09 | 7.42 | 33.44 |
| reflection | 25.23 | 27.34 | 16.88 | 21.80 | 31.25 | 20.23 | 5.86 | 43.20 |
| scrap_paper | 33.98 | 44.92 | 17.66 | 30.78 | 75.08 | 72.97 | 18.28 | 76.02 |
| shadow | 15.94 | 18.28 | 16.09 | 13.91 | 20.39 | 21.02 | 18.98 | 33.05 |

**Observations:**
- **Occlusion** corruptions (top-left, bottom-right) are the most devastating, dropping all ViT/DeiT models to 2–4% accuracy. ConvNeXt-T is notably more robust (33%).
- **Scrap paper** shows the widest variance: from 17.7% (DeiT-Ti) to 76.0% (ConvNeXt-T).
- **ConvNeXt-T** is the most robust ViT-era model overall, likely due to its CNN-like inductive biases.
- The **massive robustness gap** (55–84 pp) between clean and corrupted accuracy demonstrates that real-world corruptions are far more challenging than simulated ones.

---

### 2.3 ImageNet Validation (Clean)

Pretrained models evaluated on the standard ImageNet-1K validation set (50,000 images, 1000 classes). No fine-tuning. Results cross-validated against published reference accuracies.

| Model | Top-1 Acc | Top-1 Loss | Published Ref | Δ |
|-------|-----------|------------|---------------|---|
| ConvNeXt-T | 81.87% | 0.7920 | 82.1% | −0.23 pp |
| ViT-S/16 | 81.40% | 0.6707 | 81.4% | −0.00 pp |
| Swin-T | 80.91% | 0.8272 | 81.2% | −0.29 pp |
| EfficientFormer-L1 | 80.18% | 0.7981 | 80.2% | −0.02 pp |
| DeiT-S/16 | 79.72% | 0.8885 | 79.8% | −0.08 pp |
| MobileViT-S | 78.30% | 0.9129 | 78.4% | −0.10 pp |
| ViT-Ti/16 | 75.45% | 0.9425 | 75.5% | −0.05 pp |
| DeiT-Ti/16 | 72.03% | 1.2280 | 72.2% | −0.17 pp |

All results are within 0.3 percentage points of published values, confirming correct model-specific preprocessing.

---

### 2.4 ImageNet-A (Natural Adversarial Examples)

Pretrained models evaluated on the ImageNet-A benchmark (7,500 naturally adversarial images across 200 classes). No fine-tuning.

| Model | Top-1 Acc |
|-------|-----------|
| ViT-S/16 | 13.16% |
| ConvNeXt-T | 10.43% |
| Swin-T | 8.79% |
| DeiT-S/16 | 8.40% |
| MobileViT-S | 5.52% |
| EfficientFormer-L1 | 5.49% |
| ViT-Ti/16 | 3.69% |
| DeiT-Ti/16 | 2.60% |

**Observations:**
- All models score below 14%, consistent with the extreme difficulty of ImageNet-A.
- ViT-S/16 leads at 13.2%, benefiting from ImageNet-21K pretraining.
- Larger models generally handle natural adversarial examples better.

---

### 2.5 ImageNet-C (Corruption Robustness)

Pretrained models evaluated on ImageNet-C (19 corruptions × 5 severity levels). No fine-tuning. mCE (mean Corruption Error) = 1 − mean accuracy (lower is better).

| Model | Mean Acc (15 std) | mCE (15 std) | Mean Acc (all 19) | mCE (all 19) |
|-------|-------------------|--------------|-------------------|--------------|
| ViT-S/16 | 58.00% | 42.00% | 59.40% | 40.60% |
| ConvNeXt-T | 57.05% | 42.95% | 58.27% | 41.73% |
| DeiT-S/16 | 55.54% | 44.46% | 56.66% | 43.34% |
| Swin-T | 52.93% | 47.07% | 54.34% | 45.66% |
| EfficientFormer-L1 | 49.09% | 50.91% | 50.62% | 49.38% |
| DeiT-Ti/16 | 43.10% | 56.90% | 44.48% | 55.52% |
| ViT-Ti/16 | 42.72% | 57.28% | 44.27% | 55.73% |
| MobileViT-S | 41.39% | 58.61% | 43.32% | 56.68% |

#### ImageNet-C Per-Corruption Mean Accuracy (%)

| Corruption | ViT-Ti | ViT-S | DeiT-Ti | DeiT-S | Swin-T | MobileViT-S | EffFormer-L1 | ConvNeXt-T |
|------------|--------|-------|---------|--------|--------|-------------|--------------|------------|
| gaussian_noise | 38.93 | 58.30 | 42.79 | 57.33 | 54.31 | 33.46 | 46.70 | 60.18 |
| shot_noise | 37.13 | 56.35 | 40.24 | 54.14 | 51.84 | 31.25 | 44.56 | 58.31 |
| impulse_noise | 36.13 | 57.14 | 40.83 | 55.15 | 51.77 | 32.41 | 44.64 | 58.25 |
| defocus_blur | 39.98 | 53.38 | 36.08 | 47.38 | 43.57 | 37.35 | 41.00 | 47.59 |
| glass_blur | 32.06 | 42.86 | 28.53 | 37.82 | 31.76 | 25.91 | 28.39 | 34.42 |
| motion_blur | 44.12 | 59.22 | 40.46 | 51.75 | 49.00 | 40.51 | 45.47 | 54.54 |
| zoom_blur | 34.66 | 47.83 | 32.49 | 42.53 | 42.23 | 37.91 | 41.89 | 45.96 |
| snow | 32.91 | 54.95 | 39.00 | 53.52 | 50.86 | 37.17 | 42.79 | 54.40 |
| frost | 37.40 | 53.99 | 45.46 | 59.29 | 56.61 | 38.39 | 46.62 | 59.01 |
| fog | 39.71 | 60.07 | 45.47 | 60.41 | 60.93 | 45.86 | 58.31 | 61.94 |
| brightness | 66.26 | 75.20 | 63.21 | 72.51 | 72.82 | 68.73 | 72.20 | 74.91 |
| contrast | 37.77 | 55.24 | 50.58 | 62.81 | 61.44 | 39.70 | 59.69 | 65.75 |
| elastic_transform | 50.22 | 59.03 | 45.21 | 56.23 | 51.27 | 45.85 | 48.38 | 53.59 |
| pixelate | 56.80 | 68.83 | 42.84 | 58.56 | 53.49 | 51.57 | 53.48 | 60.44 |
| jpeg_compression | 56.74 | 67.59 | 53.26 | 63.62 | 61.99 | 54.70 | 62.20 | 66.47 |
| speckle_noise | 45.21 | 63.44 | 46.55 | 58.90 | 57.01 | 40.07 | 51.52 | 62.26 |
| gaussian_blur | 43.20 | 55.87 | 39.27 | 50.22 | 46.35 | 40.72 | 43.90 | 50.18 |
| spatter | 51.63 | 68.11 | 52.25 | 63.89 | 64.90 | 57.33 | 59.68 | 65.69 |
| saturate | 60.22 | 71.15 | 60.59 | 70.54 | 70.25 | 64.10 | 70.35 | 73.30 |

---

### 2.6 Lens (ImageNet-ES)

All models were fine-tuned on the Lens dataset (200 ImageNet synsets recaptured through camera, 7,200 train / 800 val / 1,000 test images), then evaluated on 192 corruption subsets.

| Model | Val Acc | Test Acc | Test Loss | Mean Corrupt Acc | Robustness Drop |
|-------|---------|----------|-----------|------------------|-----------------|
| ViT-S/16 | 85.88% | 83.90% | 0.7680 | 18.81% | 65.09 pp |
| ConvNeXt-T | 85.62% | 82.20% | 0.7061 | 16.00% | 66.20 pp |
| DeiT-S/16 | 84.25% | 81.70% | 1.3882 | 16.11% | 65.59 pp |
| Swin-T | 81.63% | 81.80% | 0.7228 | 14.31% | 67.49 pp |
| MobileViT-S | 80.63% | 78.40% | 1.1359 | 8.37% | 70.03 pp |
| EfficientFormer-L1 | 76.75% | 75.30% | 1.0147 | 6.64% | 68.66 pp |
| DeiT-Ti/16 | 71.37% | 69.70% | 1.4794 | 9.27% | 60.43 pp |
| ViT-Ti/16 | 75.25% | 69.40% | 1.2528 | 8.91% | 60.49 pp |

---

## 3. Cross-Dataset Summary

### 3.1 Rankings by Dataset

| Rank | CIFAR-10 | ScrewSet | ImageNet-Val | ImageNet-A | ImageNet-C (mCE↓) | Lens |
|------|----------|----------|--------------|------------|-------------------|------|
| 1 | ConvNeXt-T (98.0%) | ConvNeXt-T (45.4%) | ConvNeXt-T (81.9%) | ViT-S (13.2%) | ViT-S (42.0%) | ViT-S (83.9%) |
| 2 | MobileViT-S (97.3%) | Swin-T (33.4%) | ViT-S (81.4%) | ConvNeXt-T (10.4%) | ConvNeXt-T (43.0%) | ConvNeXt-T (82.2%) |
| 3 | ViT-S (97.2%) | MobileViT-S (28.1%) | Swin-T (80.9%) | Swin-T (8.8%) | DeiT-S (44.5%) | Swin-T (81.8%) |
| 4 | DeiT-S (96.5%) | ViT-S (24.4%) | EffFormer-L1 (80.2%) | DeiT-S (8.4%) | Swin-T (47.1%) | DeiT-S (81.7%) |
| 5 | Swin-T (96.1%) | ViT-Ti (21.2%) | DeiT-S (79.7%) | MobileViT-S (5.5%) | EffFormer-L1 (50.9%) | MobileViT-S (78.4%) |
| 6 | EffFormer-L1 (96.1%) | DeiT-S (19.5%) | MobileViT-S (78.3%) | EffFormer-L1 (5.5%) | DeiT-Ti (56.9%) | EffFormer-L1 (75.3%) |
| 7 | ViT-Ti (95.7%) | DeiT-Ti (16.9%) | ViT-Ti (75.5%) | ViT-Ti (3.7%) | ViT-Ti (57.3%) | DeiT-Ti (69.7%) |
| 8 | DeiT-Ti (93.2%) | EffFormer-L1 (16.2%) | DeiT-Ti (72.0%) | DeiT-Ti (2.6%) | MobileViT-S (58.6%) | ViT-Ti (69.4%) |

> ScrewSet column shows **mean corruption accuracy** (higher = more robust).  
> ImageNet-C column shows **mCE on 15 standard corruptions** (lower = more robust).

### 3.2 Key Findings

1. **ConvNeXt-T consistently ranks #1–2** across all datasets, suggesting that CNN-like inductive biases (locality, translation equivariance) remain beneficial even in the transformer era.

2. **ViT-S/16 is the most robust pure transformer**, benefiting significantly from ImageNet-21K pretraining (vs. ImageNet-1K only for DeiT-S).

3. **ScrewSet reveals catastrophic fragility**: All models achieve >99.4% clean accuracy but collapse to 16–45% under real-world corruptions. The robustness gap of 55–84 percentage points far exceeds what is observed on synthetic benchmarks like CIFAR-10-C (9–20 pp) or ImageNet-C (42–59 pp mCE).

4. **Occlusion is the hardest corruption** for ViTs: most models drop to 2–7% accuracy on occluded screws, while ConvNeXt-T maintains ~33% — still a dramatic drop from 99.98% clean.

5. **Model scale matters more than architecture** for corruption robustness on synthetic benchmarks: ViT-S > ViT-Ti, DeiT-S > DeiT-Ti consistently.

6. **MobileViT-S achieves perfect clean accuracy** (100.000%) on ScrewSet but still drops to 28% average under real corruptions, confirming that memorization of clean patterns does not transfer to corrupted inputs.

---

## 4. Integrity Checks

| Check | Status |
|-------|--------|
| All 48 JSON files present (8 models × 6 datasets) | ✅ |
| ImageNet-Val accuracy within ±0.3 pp of published references | ✅ |
| All eval-only datasets use model-specific normalization from timm | ✅ |
| CIFAR-10-C: 19 corruption types per model | ✅ |
| ScrewSet-C: 6 corruption types per model | ✅ |
| ImageNet-C: 19 corruption types × 5 severities per model | ✅ |
| Lens: 192 corruption subsets per model | ✅ |
| No anomalous accuracy values (all within expected ranges) | ✅ |

---

## 5. File Listing

All result JSON files are located in `results/phase2/`:

```
convnext_tiny_cifar10_baselines.json
convnext_tiny_imagenet_a_baselines.json
convnext_tiny_imagenet_c_baselines.json
convnext_tiny_imagenet_val_baselines.json
convnext_tiny_lens_baselines.json
convnext_tiny_screwset_baselines.json
deit_small_patch16_224_cifar10_baselines.json
deit_small_patch16_224_imagenet_a_baselines.json
deit_small_patch16_224_imagenet_c_baselines.json
deit_small_patch16_224_imagenet_val_baselines.json
deit_small_patch16_224_lens_baselines.json
deit_small_patch16_224_screwset_baselines.json
deit_tiny_patch16_224_cifar10_baselines.json
deit_tiny_patch16_224_imagenet_a_baselines.json
deit_tiny_patch16_224_imagenet_c_baselines.json
deit_tiny_patch16_224_imagenet_val_baselines.json
deit_tiny_patch16_224_lens_baselines.json
deit_tiny_patch16_224_screwset_baselines.json
efficientformer_l1_cifar10_baselines.json
efficientformer_l1_imagenet_a_baselines.json
efficientformer_l1_imagenet_c_baselines.json
efficientformer_l1_imagenet_val_baselines.json
efficientformer_l1_lens_baselines.json
efficientformer_l1_screwset_baselines.json
mobilevit_s_cifar10_baselines.json
mobilevit_s_imagenet_a_baselines.json
mobilevit_s_imagenet_c_baselines.json
mobilevit_s_imagenet_val_baselines.json
mobilevit_s_lens_baselines.json
mobilevit_s_screwset_baselines.json
swin_tiny_patch4_window7_224_cifar10_baselines.json
swin_tiny_patch4_window7_224_imagenet_a_baselines.json
swin_tiny_patch4_window7_224_imagenet_c_baselines.json
swin_tiny_patch4_window7_224_imagenet_val_baselines.json
swin_tiny_patch4_window7_224_lens_baselines.json
swin_tiny_patch4_window7_224_screwset_baselines.json
vit_small_patch16_224_cifar10_baselines.json
vit_small_patch16_224_imagenet_a_baselines.json
vit_small_patch16_224_imagenet_c_baselines.json
vit_small_patch16_224_imagenet_val_baselines.json
vit_small_patch16_224_lens_baselines.json
vit_small_patch16_224_screwset_baselines.json
vit_tiny_patch16_224_cifar10_baselines.json
vit_tiny_patch16_224_imagenet_a_baselines.json
vit_tiny_patch16_224_imagenet_c_baselines.json
vit_tiny_patch16_224_imagenet_val_baselines.json
vit_tiny_patch16_224_lens_baselines.json
vit_tiny_patch16_224_screwset_baselines.json
```

Training script: `scripts/phase2/phase2_vit_baselines.py`
