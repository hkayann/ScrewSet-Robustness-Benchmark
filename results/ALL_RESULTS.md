# ARCADE–ScrewSet: Comprehensive Results

> **Generated:** 2025-07-14  
> **Repository:** ARCADE--Screwset  
> **Hardware:** NVIDIA RTX PRO 6000 Blackwell Max-Q (95 GB VRAM)  
> **Framework:** PyTorch 2.10.0+cu128 · timm 1.0.24 · open_clip 3.2.0 · transformers 5.2.0

---

## Table of Contents

1. [Experimental Overview](#1-experimental-overview)
2. [Datasets](#2-datasets)
3. [Master Comparison Table (24 Models × 9 Metrics)](#3-master-comparison-table)
4. [Phase 1 — CNN Baselines](#4-phase-1--cnn-baselines)
5. [Phase 2 — Vision Transformer Baselines](#5-phase-2--vision-transformer-baselines)
6. [Phase 3 — Vision-Language Model Baselines](#6-phase-3--vision-language-model-baselines)
7. [VQA Ablation Study (BLIP-2 & LLaVA)](#7-vqa-ablation-study)
8. [Key Findings](#8-key-findings)

---

## 1. Experimental Overview

| Property | Value |
|---|---|
| Total models evaluated | 24 (8 CNN + 8 ViT + 8 VLM) |
| Datasets | 6 (ScrewSet, CIFAR-10, Lens, ImageNet-val, ImageNet-C, ImageNet-A) |
| Total JSON result files | 146 (144 baselines + 2 VQA ablations) |
| Corruption types | 19 per corruption dataset, severity 3 |
| Phase 1–2 evaluation mode | Fine-tuned (transfer learning from ImageNet pretrained weights) |
| Phase 3 evaluation mode | Zero-shot (prompt-based, no fine-tuning) |

---

## 2. Datasets

| Dataset | Classes | Clean Samples | Corruption Variants | Description |
|---|---|---|---|---|
| **ScrewSet** | 40 | 40,960 | 19 × 40,960 = 778,240 | Industrial screw classification (ScrewSet-S clean + ScrewSet-C corrupted) |
| **CIFAR-10** | 10 | 10,000 | 19 × 10,000 = 190,000 | Standard 10-class benchmark |
| **Lens** | ~20 | ~1,000 | 19 × ~1,000 = ~19,000 | Optical lens defect dataset |
| **ImageNet-val** | 1,000 | 50,000 | — | ILSVRC2012 validation set |
| **ImageNet-C** | 1,000 | 19 × 50,000 = 950,000 | 19 corruption types | Corrupted ImageNet (severity 3) |
| **ImageNet-A** | 200 | 7,500 | — | Natural adversarial examples |

---

## 3. Master Comparison Table

All accuracy values in **percentage (%)**.  
`—` indicates the evaluation was not performed (generative VLMs on ImageNet-C are computationally infeasible: 950K images with autoregressive decoding).

### Legend

- **Params**: Total model parameters (millions or billions)
- **Mode**: `FT` = Fine-Tuned, `ZS` = Zero-Shot
- **SS**: ScrewSet clean accuracy
- **SS-C**: ScrewSet-C mean corruption accuracy
- **C10**: CIFAR-10 clean accuracy
- **C10-C**: CIFAR-10-C mean corruption accuracy
- **Lens**: Lens clean accuracy
- **Lens-C**: Lens-C mean corruption accuracy
- **IN-val**: ImageNet validation accuracy
- **IN-C**: ImageNet-C mean accuracy
- **IN-A**: ImageNet-A accuracy

### Phase 1 — CNNs (Fine-Tuned)

| # | Model | Family | Params | Mode | SS | SS-C | C10 | C10-C | Lens | Lens-C | IN-val | IN-C | IN-A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-18 | ResNet | 11.7M | FT | 98.02 | 9.23 | 83.21 | 63.78 | 50.60 | 4.05 | 69.76 | 33.18 | 0.29 |
| 2 | SqueezeNet 1.1 | SqueezeNet | 1.2M | FT | 95.09 | 16.63 | 77.74 | 59.44 | 0.50 | 0.50 | 58.18 | 19.59 | 0.28 |
| 3 | MobileNetV3-L | MobileNet | 5.5M | FT | 95.95 | 11.08 | 84.94 | 67.91 | 62.60 | 6.63 | 74.06 | 40.46 | 1.68 |
| 4 | ShuffleNetV2 | ShuffleNet | 2.3M | FT | 96.18 | 12.30 | 82.01 | 67.05 | 53.10 | 3.37 | 69.34 | 30.02 | 0.64 |
| 5 | EfficientNetV2-S | EfficientNet | 23.9M | FT | 95.45 | 11.28 | 85.29 | 73.33 | 82.20 | 15.40 | 81.03 | 58.32 | 9.24 |
| 6 | GhostNetV2 | GhostNet | 6.2M | FT | 95.02 | 11.63 | 85.01 | 67.05 | 70.20 | 8.32 | 75.13 | 41.37 | 1.73 |
| 7 | ConvNeXtV2-Atto | ConvNeXt | 3.7M | FT | 91.67 | 8.29 | 80.26 | 73.04 | 1.20 | 0.70 | 76.30 | 45.37 | 2.61 |
| 8 | MobileNetV4-S | MobileNet | 3.8M | FT | 96.03 | 7.80 | 76.48 | 66.66 | 53.60 | 2.81 | 73.74 | 39.50 | 1.21 |

### Phase 2 — Vision Transformers (Fine-Tuned)

| # | Model | Family | Params | Mode | SS | SS-C | C10 | C10-C | Lens | Lens-C | IN-val | IN-C | IN-A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | ViT-Tiny | ViT | 5.7M | FT | 99.66 | 21.17 | 95.70 | 81.37 | 69.40 | 8.91 | 75.45 | 44.27 | 3.69 |
| 10 | ViT-Small | ViT | 22.1M | FT | 99.79 | 24.40 | 97.18 | 88.20 | 83.90 | 18.81 | 81.40 | 59.40 | 13.16 |
| 11 | DeiT-Tiny | DeiT | 5.7M | FT | 99.55 | 16.91 | 93.15 | 80.59 | 69.70 | 9.27 | 72.03 | 44.48 | 2.60 |
| 12 | DeiT-Small | DeiT | 22.1M | FT | 99.65 | 19.48 | 96.52 | 87.29 | 81.70 | 16.11 | 79.72 | 56.66 | 8.40 |
| 13 | Swin-Tiny | Swin | 28.3M | FT | 99.94 | 33.39 | 96.10 | 83.23 | 81.80 | 14.31 | 80.91 | 54.34 | 8.79 |
| 14 | ConvNeXt-Tiny | ConvNeXt | 28.6M | FT | 99.98 | 45.39 | 98.03 | 86.29 | 82.20 | 16.00 | 81.87 | 58.27 | 10.43 |
| 15 | MobileViT-S | MobileViT | 5.6M | FT | 100.00 | 28.09 | 97.30 | 78.48 | 78.40 | 8.37 | 78.30 | 43.32 | 5.52 |
| 16 | EfficientFormer-L1 | EfficientFormer | 12.3M | FT | 99.91 | 16.15 | 96.07 | 75.74 | 75.30 | 6.64 | 80.18 | 50.62 | 5.49 |

### Phase 3 — Vision-Language Models (Zero-Shot)

| # | Model | Family | Params | Mode | SS | SS-C | C10 | C10-C | Lens | Lens-C | IN-val | IN-C | IN-A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | CLIP ViT-B/32 | CLIP | 151M | ZS | 6.71 | 3.44 | 86.69 | 70.92 | 80.00 | 19.64 | 56.46 | 35.00 | 14.25 |
| 18 | CLIP ViT-B/16 | CLIP | 150M | ZS | 3.31 | 3.22 | 88.41 | 73.10 | 84.60 | 21.57 | 61.20 | 38.26 | 26.32 |
| 19 | CLIP ViT-L/14 | CLIP | 428M | ZS | 5.21 | 5.08 | 93.66 | 82.01 | 88.70 | 27.21 | 69.76 | 49.86 | 44.81 |
| 20 | OpenCLIP ViT-B/16 | OpenCLIP | 150M | ZS | 4.90 | 4.23 | 95.00 | 81.38 | 85.40 | 20.69 | 66.41 | 42.86 | 18.72 |
| 21 | SigLIP ViT-B/16 | SigLIP | 150M | ZS | 10.78 | 7.34 | 92.39 | 72.63 | 88.10 | 22.46 | 67.31 | 44.09 | 25.25 |
| 22 | EVA02-CLIP ViT-B/16 | EVA-CLIP | 150M | ZS | 4.61 | 2.62 | 98.48 | 90.97 | 89.90 | 26.66 | 71.36 | 53.90 | 32.67 |
| 23 | BLIP-2 | BLIP-2 | 3.4B | ZS | 0.95 | 0.49 | 67.10 | 52.41 | 59.70 | 8.76 | 53.35 | — | 41.36 |
| 24 | LLaVA-1.5-7B | LLaVA | 7.1B | ZS | 2.50 | 2.50 | 92.15 | 82.58 | 37.40 | 7.68 | 28.54 | — | 22.67 |

---

## 4. Phase 1 — CNN Baselines

### 4.1 Setup

- **Models:** 8 CNNs from timm / torchvision, transfer learning from ImageNet-1K pretrained weights
- **Training:** Fine-tuned final classifier layer on ScrewSet-S (40 classes), CIFAR-10 (10 classes), or Lens (~20 classes)
- **ImageNet:** Evaluated using pre-trained weights (no fine-tuning, original 1000-class head)
- **Input size:** 224 × 224 (standard)

### 4.2 Detailed Results

#### ScrewSet (40 classes)

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| ResNet-18 | 98.02 | 9.23 | −88.79 |
| SqueezeNet 1.1 | 95.09 | 16.63 | −78.46 |
| MobileNetV3-L | 95.95 | 11.08 | −84.87 |
| ShuffleNetV2 | 96.18 | 12.30 | −83.88 |
| EfficientNetV2-S | 95.45 | 11.28 | −84.17 |
| GhostNetV2 | 95.02 | 11.63 | −83.39 |
| ConvNeXtV2-Atto | 91.67 | 8.29 | −83.38 |
| MobileNetV4-S | 96.03 | 7.80 | −88.23 |

#### CIFAR-10 (10 classes)

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| ResNet-18 | 83.21 | 63.78 | −19.43 |
| SqueezeNet 1.1 | 77.74 | 59.44 | −18.30 |
| MobileNetV3-L | 84.94 | 67.91 | −17.03 |
| ShuffleNetV2 | 82.01 | 67.05 | −14.96 |
| EfficientNetV2-S | 85.29 | 73.33 | −11.96 |
| GhostNetV2 | 85.01 | 67.05 | −17.96 |
| ConvNeXtV2-Atto | 80.26 | 73.04 | −7.22 |
| MobileNetV4-S | 76.48 | 66.66 | −9.82 |

#### Lens

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| ResNet-18 | 50.60 | 4.05 | −46.55 |
| SqueezeNet 1.1 | 0.50 | 0.50 | 0.00 |
| MobileNetV3-L | 62.60 | 6.63 | −55.97 |
| ShuffleNetV2 | 53.10 | 3.37 | −49.73 |
| EfficientNetV2-S | 82.20 | 15.40 | −66.80 |
| GhostNetV2 | 70.20 | 8.32 | −61.88 |
| ConvNeXtV2-Atto | 1.20 | 0.70 | −0.50 |
| MobileNetV4-S | 53.60 | 2.81 | −50.79 |

#### ImageNet (pre-trained, 1000 classes)

| Model | Val Acc (%) | IN-C Mean Acc (%) | IN-C mCE (%) | IN-A Acc (%) |
|---|---|---|---|---|
| ResNet-18 | 69.76 | 33.18 | 66.82 | 0.29 |
| SqueezeNet 1.1 | 58.18 | 19.59 | 80.41 | 0.28 |
| MobileNetV3-L | 74.06 | 40.46 | 59.54 | 1.68 |
| ShuffleNetV2 | 69.34 | 30.02 | 69.98 | 0.64 |
| EfficientNetV2-S | 81.03 | 58.32 | 41.68 | 9.24 |
| GhostNetV2 | 75.13 | 41.37 | 58.63 | 1.73 |
| ConvNeXtV2-Atto | 76.30 | 45.37 | 54.63 | 2.61 |
| MobileNetV4-S | 73.74 | 39.50 | 60.50 | 1.21 |

---

## 5. Phase 2 — Vision Transformer Baselines

### 5.1 Setup

- **Models:** 8 ViT-family architectures from timm, transfer learning from ImageNet-1K pretrained weights
- **Training:** Fine-tuned final classifier layer (same as Phase 1)
- **Input size:** 224 × 224

### 5.2 Detailed Results

#### ScrewSet (40 classes)

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| ViT-Tiny | 99.66 | 21.17 | −78.49 |
| ViT-Small | 99.79 | 24.40 | −75.39 |
| DeiT-Tiny | 99.55 | 16.91 | −82.64 |
| DeiT-Small | 99.65 | 19.48 | −80.17 |
| Swin-Tiny | 99.94 | 33.39 | −66.55 |
| ConvNeXt-Tiny | 99.98 | 45.39 | −54.59 |
| MobileViT-S | 100.00 | 28.09 | −71.91 |
| EfficientFormer-L1 | 99.91 | 16.15 | −83.76 |

#### CIFAR-10 (10 classes)

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| ViT-Tiny | 95.70 | 81.37 | −14.33 |
| ViT-Small | 97.18 | 88.20 | −8.98 |
| DeiT-Tiny | 93.15 | 80.59 | −12.56 |
| DeiT-Small | 96.52 | 87.29 | −9.23 |
| Swin-Tiny | 96.10 | 83.23 | −12.87 |
| ConvNeXt-Tiny | 98.03 | 86.29 | −11.74 |
| MobileViT-S | 97.30 | 78.48 | −18.82 |
| EfficientFormer-L1 | 96.07 | 75.74 | −20.33 |

#### Lens

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| ViT-Tiny | 69.40 | 8.91 | −60.49 |
| ViT-Small | 83.90 | 18.81 | −65.09 |
| DeiT-Tiny | 69.70 | 9.27 | −60.43 |
| DeiT-Small | 81.70 | 16.11 | −65.59 |
| Swin-Tiny | 81.80 | 14.31 | −67.49 |
| ConvNeXt-Tiny | 82.20 | 16.00 | −66.20 |
| MobileViT-S | 78.40 | 8.37 | −70.03 |
| EfficientFormer-L1 | 75.30 | 6.64 | −68.66 |

#### ImageNet (pre-trained, 1000 classes)

| Model | Val Acc (%) | IN-C Mean Acc (%) | IN-C mCE (%) | IN-A Acc (%) |
|---|---|---|---|---|
| ViT-Tiny | 75.45 | 44.27 | 55.73 | 3.69 |
| ViT-Small | 81.40 | 59.40 | 40.60 | 13.16 |
| DeiT-Tiny | 72.03 | 44.48 | 55.52 | 2.60 |
| DeiT-Small | 79.72 | 56.66 | 43.34 | 8.40 |
| Swin-Tiny | 80.91 | 54.34 | 45.66 | 8.79 |
| ConvNeXt-Tiny | 81.87 | 58.27 | 41.73 | 10.43 |
| MobileViT-S | 78.30 | 43.32 | 56.68 | 5.52 |
| EfficientFormer-L1 | 80.18 | 50.62 | 49.38 | 5.49 |

---

## 6. Phase 3 — Vision-Language Model Baselines

### 6.1 Setup

- **Models:** 6 contrastive (CLIP-family) + 2 generative (BLIP-2, LLaVA)
- **Evaluation:** Zero-shot classification via text prompts (no fine-tuning)
- **Contrastive models:** Cosine similarity between image embedding and class-name text embeddings
- **Generative models:** Free-form answer parsed and matched to nearest class name
- **ImageNet-C for generative models:** Skipped — 950K images × autoregressive decoding is computationally infeasible

### 6.2 Detailed Results

#### ScrewSet (40 classes) — Zero-Shot

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| CLIP ViT-B/32 | 6.71 | 3.44 | −3.27 |
| CLIP ViT-B/16 | 3.31 | 3.22 | −0.09 |
| CLIP ViT-L/14 | 5.21 | 5.08 | −0.13 |
| OpenCLIP ViT-B/16 | 4.90 | 4.23 | −0.67 |
| SigLIP ViT-B/16 | 10.78 | 7.34 | −3.44 |
| EVA02-CLIP ViT-B/16 | 4.61 | 2.62 | −1.99 |
| BLIP-2 | 0.95 | 0.49 | −0.46 |
| LLaVA-1.5-7B | 2.50 | 2.50 | 0.00 |

> **Note:** ScrewSet has 40 fine-grained screw classes (random chance = 2.5%). Generative models show strong position/label bias (LLaVA always predicts the same class). All zero-shot models score near random chance, highlighting the domain gap for specialized industrial classification.

#### CIFAR-10 (10 classes) — Zero-Shot

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| CLIP ViT-B/32 | 86.69 | 70.92 | −15.77 |
| CLIP ViT-B/16 | 88.41 | 73.10 | −15.31 |
| CLIP ViT-L/14 | 93.66 | 82.01 | −11.65 |
| OpenCLIP ViT-B/16 | 95.00 | 81.38 | −13.62 |
| SigLIP ViT-B/16 | 92.39 | 72.63 | −19.76 |
| EVA02-CLIP ViT-B/16 | 98.48 | 90.97 | −7.51 |
| BLIP-2 | 67.10 | 52.41 | −14.69 |
| LLaVA-1.5-7B | 92.15 | 82.58 | −9.57 |

#### Lens — Zero-Shot

| Model | Clean Acc (%) | Mean Corrupt Acc (%) | Δ (pp) |
|---|---|---|---|
| CLIP ViT-B/32 | 80.00 | 19.64 | −60.36 |
| CLIP ViT-B/16 | 84.60 | 21.57 | −63.03 |
| CLIP ViT-L/14 | 88.70 | 27.21 | −61.49 |
| OpenCLIP ViT-B/16 | 85.40 | 20.69 | −64.71 |
| SigLIP ViT-B/16 | 88.10 | 22.46 | −65.64 |
| EVA02-CLIP ViT-B/16 | 89.90 | 26.66 | −63.24 |
| BLIP-2 | 59.70 | 8.76 | −50.94 |
| LLaVA-1.5-7B | 37.40 | 7.68 | −29.72 |

#### ImageNet (1000 classes) — Zero-Shot

| Model | Val Acc (%) | IN-C Mean Acc (%) | IN-C mCE (%) | IN-A Acc (%) |
|---|---|---|---|---|
| CLIP ViT-B/32 | 56.46 | 35.00 | 65.00 | 14.25 |
| CLIP ViT-B/16 | 61.20 | 38.26 | 61.74 | 26.32 |
| CLIP ViT-L/14 | 69.76 | 49.86 | 50.14 | 44.81 |
| OpenCLIP ViT-B/16 | 66.41 | 42.86 | 57.14 | 18.72 |
| SigLIP ViT-B/16 | 67.31 | 44.09 | 55.91 | 25.25 |
| EVA02-CLIP ViT-B/16 | 71.36 | 53.90 | 46.10 | 32.67 |
| BLIP-2 | 53.35 | — | — | 41.36 |
| LLaVA-1.5-7B | 28.54 | — | — | 22.67 |

---

## 7. VQA Ablation Study

Comparing closed-form classification prompting vs. open-ended VQA on ScrewSet (40 classes):

| Model | Closed-Form Acc (%) | Open-Ended Acc (%) | Gap (pp) |
|---|---|---|---|
| BLIP-2 | 0.95 | 2.47 | −1.52 |
| LLaVA-1.5-7B | 2.50 | 2.46 | +0.04 |

- **Closed-form prompt:** _"Classify this image into one of the following categories: [list]. Answer with only the category name."_
- **Open-ended prompt:** _"Describe this screw precisely. Size, length, head type/color."_
- **Finding:** Both models perform near random chance (2.5%) regardless of prompt strategy, confirming that 40-class fine-grained screw classification is beyond current VLM zero-shot capability.

---

## 8. Key Findings

### 8.1 Clean Accuracy

1. **ViTs dominate on ScrewSet** — All 8 fine-tuned ViTs achieve 99.5–100% clean accuracy vs. 91.7–98.0% for CNNs
2. **Zero-shot VLMs fail on ScrewSet** — Best zero-shot accuracy is 10.8% (SigLIP), vs. 2.5% random chance. The 40-class fine-grained industrial task is far beyond current VLM capability
3. **EVA02-CLIP leads on CIFAR-10** — 98.48% zero-shot, outperforming most fine-tuned CNNs
4. **CLIP ViT-L/14 leads on Lens** — 88.70% zero-shot, competitive with fine-tuned ViTs

### 8.2 Corruption Robustness

1. **Massive clean→corrupt gap on ScrewSet** — Average drop of 83 pp for CNNs, 74 pp for ViTs. ScrewSet-C is an extremely challenging benchmark
2. **ViTs more robust than CNNs** — ConvNeXt-Tiny retains 45.39% under corruption (best overall); ResNet-18 retains only 9.23%
3. **Zero-shot models show smaller absolute drops** — Because clean accuracy is already low; relative robustness is similar
4. **CIFAR-10 corruption gap is moderate** — 10–20 pp drop for most models, reflecting the simpler class structure
5. **Lens corruption gap is extreme** — Often 60+ pp drop, suggesting lens defect features are highly fragile

### 8.3 ImageNet

1. **Best fine-tuned IN-val:** ConvNeXt-Tiny (81.87%), EfficientNetV2-S (81.03%), ViT-Small (81.40%)
2. **Best zero-shot IN-val:** EVA02-CLIP (71.36%), CLIP ViT-L/14 (69.76%)
3. **IN-A reveals adversarial fragility** — All CNNs score < 10% on ImageNet-A; CLIP ViT-L/14 achieves 44.81% (best overall), showing the advantage of large-scale pretraining for distribution shift robustness

### 8.4 Model Efficiency vs. Performance

| Model | Params | ScrewSet | IN-val | IN-C | IN-A |
|---|---|---|---|---|---|
| ConvNeXtV2-Atto | 3.7M | 91.67% | 76.30% | 45.37% | 2.61% |
| MobileViT-S | 5.6M | 100.00% | 78.30% | 43.32% | 5.52% |
| CLIP ViT-L/14 | 428M | 5.21% | 69.76% | 49.86% | 44.81% |
| BLIP-2 | 3.4B | 0.95% | 53.35% | — | 41.36% |

> Small fine-tuned models (3.7–5.6M params) achieve near-perfect ScrewSet accuracy while VLMs with 100–1000× more parameters fail in zero-shot settings. However, VLMs show dramatically better adversarial robustness (IN-A), suggesting complementary strengths.

---

*This file consolidates 146 JSON result files across 3 phase directories. All accuracy values are rounded to 2 decimal places for readability. Raw data is available in the respective JSON files under `results/phase{1,2,3}/`.*
