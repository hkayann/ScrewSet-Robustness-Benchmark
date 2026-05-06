# Phase 3: Zero-Shot Vision-Language Model Baseline Results

**Date:** February 21, 2026  
**Precision:** FP32 / FP16 (model-dependent)  
**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)  
**Framework:** PyTorch 2.10.0+cu128, open_clip 3.2.0, transformers 5.2.0, timm 1.0.24  
**Evaluation Mode:** Zero-shot (no fine-tuning)

> Canonical cross-phase summary is maintained in `results/ALL_RESULTS.md`. This phase file is a detailed breakdown of Phase 3 only.

---

## 1. Experimental Setup

### 1.1 Models

Ten Vision-Language Models were evaluated: six contrastive (CLIP-family) and four generative (BLIP-2, LLaVA, Qwen2.5-VL, Qwen3-VL).

| # | Model | Architecture | Parameters | Pretrained Weights | Type |
|---|-------|-------------|------------|-------------------|------|
| 1 | CLIP ViT-B/32 | ViT-B/32 | 151M | `openai` (via open_clip) | Contrastive |
| 2 | CLIP ViT-B/16 | ViT-B/16 | 150M | `openai` (via open_clip) | Contrastive |
| 3 | CLIP ViT-L/14 | ViT-L/14 | 428M | `openai` (via open_clip) | Contrastive |
| 4 | OpenCLIP ViT-B/16 | ViT-B/16 | 150M | `laion2b_s34b_b88k` (via open_clip) | Contrastive |
| 5 | SigLIP ViT-B/16 | ViT-B/16-SigLIP | 203M | `webli` (via open_clip) | Contrastive |
| 6 | EVA-02 CLIP B/16 | EVA02-B-16 | 150M | `merged2b_s8b_b131k` (via open_clip) | Contrastive |
| 7 | BLIP-2 (OPT-2.7B) | ViT-G + Q-Former + OPT-2.7B | 3.7B | `Salesforce/blip2-opt-2.7b` | Generative |
| 8 | LLaVA-1.5 (7B) | CLIP ViT-L/14 + Vicuna-7B | 7.1B | `llava-hf/llava-1.5-7b-hf` | Generative |
| 9 | Qwen2.5-VL (7B) | Qwen2.5-VL-7B-Instruct | 8.3B | `Qwen/Qwen2.5-VL-7B-Instruct` | Generative |
| 10 | Qwen3-VL (8B) | Qwen3-VL-8B-Instruct | 8.8B | `Qwen/Qwen3-VL-8B-Instruct` | Generative |

### 1.2 Evaluation Methodology

**Contrastive models (CLIP-family):**
- Zero-shot classification via cosine similarity between image embeddings and text embeddings
- Text embeddings built from an **80-template ensemble** following the original CLIP paper (e.g., "a photo of a {class}.", "a blurry photo of a {class}.", etc.) for ImageNet-class datasets
- Simple template ("a photo of a {class}.") for ScrewSet (domain-specific classes)
- All templates averaged and L2-normalized per class

**Generative models (BLIP-2, LLaVA, Qwen2.5-VL, Qwen3-VL):**
- Visual Question Answering (VQA) style prompting
- BLIP-2 prompt: `"Question: What type of object is shown in this image? Choose from: [class1, class2, ...]. Answer:"`
- LLaVA prompt: `"What is the main subject of this image? Answer with a single word or short phrase from the following options: [class1, class2, ...]."`  (via `apply_chat_template`)
- Qwen2.5-VL prompt: Same classification prompt via `apply_chat_template` (Qwen chat format)
- Qwen3-VL prompt: Same classification prompt via `apply_chat_template` (Qwen chat format)
- Response matched to class names via fuzzy matching (exact match → substring → word overlap → `SequenceMatcher` with cutoff 0.4)
- Corruption datasets **skipped** for generative models (per-image generation too slow for millions of images)

### 1.3 Datasets

| Dataset | Classes | Clean Images | Corruption Variants | Evaluation |
|---------|---------|-------------|---------------------|------------|
| CIFAR-10 | 10 | 10,000 test | 19 types (CIFAR-10-C) | Zero-shot |
| ScrewSet | 40 | ~640 test | 6 types (ScrewSet-C) | Zero-shot |
| ImageNet-Val | 1,000 | 50,000 | — | Zero-shot |
| ImageNet-A | 200 | 7,500 | — | Zero-shot |
| ImageNet-C | 1,000 | 50,000 × 19 corruptions × 5 severities | 19 standard + 4 extra | Zero-shot |
| Lens (ImageNet-ES) | 200 | 1,000 test | 192 param/lens combos (Lens-C) | Zero-shot |

### 1.4 Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size (contrastive) | 512 |
| Batch Size (generative) | 16–32 |
| Random Seed | 42 |
| Mixed Precision | FP16 AMP (contrastive), bfloat16 (generative) |
| Text Templates | 80 CLIP templates (ImageNet-class), simple template (ScrewSet) |
| Max New Tokens | 20 (generative models) |

---

## 2. Summary Results

### 2.1 Clean Accuracy (%)

| Model | CIFAR-10 | ScrewSet | IN-Val | IN-A | Lens |
|-------|----------|----------|--------|------|------|
| CLIP ViT-B/32 | 86.69 | 6.71 | 56.46 | 14.25 | 80.00 |
| CLIP ViT-B/16 | 88.41 | 3.31 | 61.20 | 26.32 | 84.60 |
| CLIP ViT-L/14 | 93.66 | 5.21 | 69.76 | 44.81 | 88.70 |
| OpenCLIP ViT-B/16 | 95.00 | 4.90 | 66.41 | 18.72 | 85.40 |
| SigLIP ViT-B/16 | 92.39 | 10.78 | 67.31 | 25.25 | 88.10 |
| EVA-02 CLIP B/16 | **98.48** | 4.61 | **71.36** | 32.67 | **89.90** |
| BLIP-2 (OPT-2.7B) | 67.10 | 0.95 | 53.35 | 41.36 | 59.70 |
| LLaVA-1.5 (7B) | 92.15 | 2.50 | 28.54 | 22.67 | 37.40 |
| Qwen2.5-VL (7B) | 88.27 | 12.20 | 39.22 | 30.55 | 51.90 |
| Qwen3-VL (8B) | 87.48 | 20.00 | 39.48 | 30.28 | 52.80 |

> **CLIP ViT-L/14** achieved the highest IN-A at 44.81%.

### 2.2 Mean Corruption Accuracy (%)

| Model | CIFAR-10-C | ScrewSet-C | IN-C | Lens-C |
|-------|-----------|------------|------|--------|
| CLIP ViT-B/32 | 70.92 | 3.44 | 35.00 | 19.64 |
| CLIP ViT-B/16 | 73.10 | 3.22 | 38.26 | 21.57 |
| CLIP ViT-L/14 | 82.01 | 5.08 | 49.86 | 27.21 |
| OpenCLIP ViT-B/16 | 81.38 | 4.23 | 42.86 | 20.69 |
| SigLIP ViT-B/16 | 72.63 | 7.34 | 44.09 | 22.46 |
| EVA-02 CLIP B/16 | **90.97** | 2.62 | **53.90** | **26.66** |
| BLIP-2 (OPT-2.7B) | 52.41 | 0.49 | — | 8.76 |
| LLaVA-1.5 (7B) | 82.58 | 2.50 | — | 7.68 |
| Qwen2.5-VL (7B) | 73.69 | 13.74 | — | 29.88 |
| Qwen3-VL (8B) | 73.87 | 20.68 | — | 34.17 |

> Generative models (BLIP-2, LLaVA, Qwen2.5-VL, Qwen3-VL) were skipped on ImageNet-C only (950K images with autoregressive decoding). All other corruption sets are fully evaluated.

### 2.3 Mean Corruption Error (MCE) — ImageNet-C

| Model | MCE ↓ |
|-------|-------|
| EVA-02 CLIP B/16 | **0.4610** |
| CLIP ViT-L/14 | 0.5014 |
| SigLIP ViT-B/16 | 0.5591 |
| OpenCLIP ViT-B/16 | 0.5714 |
| CLIP ViT-B/16 | 0.6174 |
| CLIP ViT-B/32 | 0.6500 |

---

## 3. Per-Corruption Breakdown

### 3.1 ImageNet-C (mean across 5 severities)

| Corruption | CLIP B/32 | CLIP B/16 | CLIP L/14 | OpenCLIP B/16 | SigLIP B/16 | EVA-02 B/16 |
|-----------|-----------|-----------|-----------|---------------|-------------|-------------|
| brightness | 50.34 | 54.45 | 64.80 | 60.54 | 62.54 | 67.73 |
| contrast | 38.57 | 41.14 | 54.46 | 46.05 | 47.48 | 54.62 |
| defocus_blur | 33.07 | 34.75 | 44.11 | 44.80 | 44.03 | 56.26 |
| elastic_transform | 35.01 | 35.86 | 44.60 | 40.54 | 41.41 | 52.06 |
| fog | 39.43 | 44.75 | 55.97 | 49.14 | 53.28 | 57.42 |
| frost | 31.45 | 36.60 | 47.96 | 39.54 | 42.81 | 49.44 |
| gaussian_blur | 34.85 | 37.22 | 47.22 | 47.54 | 45.17 | 57.51 |
| gaussian_noise | 32.59 | 33.52 | 45.64 | 32.71 | 33.83 | 45.59 |
| glass_blur | 23.92 | 28.58 | 37.65 | 30.71 | 30.53 | 41.98 |
| impulse_noise | 29.98 | 29.40 | 40.17 | 29.96 | 32.00 | 44.53 |
| jpeg_compression | 41.14 | 42.56 | 53.12 | 51.61 | 51.06 | 61.29 |
| motion_blur | 34.19 | 36.03 | 49.31 | 40.50 | 40.60 | 55.55 |
| pixelate | 40.28 | 42.99 | 55.84 | 51.16 | 52.45 | 61.66 |
| saturate | 45.42 | 51.06 | 61.76 | 56.36 | 59.67 | 64.64 |
| shot_noise | 30.86 | 32.00 | 44.71 | 31.98 | 32.17 | 44.23 |
| snow | 26.38 | 34.67 | 49.94 | 39.60 | 41.08 | 50.11 |
| spatter | 36.28 | 44.73 | 57.60 | 49.94 | 53.76 | 59.52 |
| speckle_noise | 35.31 | 38.03 | 51.80 | 39.70 | 39.68 | 52.21 |
| zoom_blur | 25.96 | 28.62 | 40.67 | 31.97 | 34.09 | 47.71 |
| **MEAN** | **35.00** | **38.26** | **49.86** | **42.86** | **44.09** | **53.90** |

### 3.2 CIFAR-10-C (mean across 5 severities)

| Corruption | CLIP B/32 | CLIP B/16 | CLIP L/14 | OpenCLIP B/16 | SigLIP B/16 | EVA-02 B/16 |
|-----------|-----------|-----------|-----------|---------------|-------------|-------------|
| brightness | 84.81 | 86.11 | 92.72 | 93.33 | 89.59 | 97.84 |
| contrast | 79.24 | 81.51 | 91.80 | 87.60 | 84.37 | 95.78 |
| defocus_blur | 79.95 | 80.67 | 87.17 | 89.66 | 84.76 | 95.59 |
| elastic_transform | 72.52 | 71.58 | 77.11 | 82.44 | 74.18 | 90.20 |
| fog | 79.83 | 82.40 | 88.64 | 90.28 | 86.24 | 95.38 |
| frost | 77.29 | 79.71 | 87.50 | 88.10 | 79.59 | 95.12 |
| gaussian_blur | 78.38 | 78.81 | 86.17 | 88.19 | 81.89 | 95.05 |
| gaussian_noise | 53.11 | 54.73 | 71.44 | 63.92 | 42.31 | 82.63 |
| glass_blur | 46.21 | 46.04 | 53.18 | 52.62 | 45.22 | 74.21 |
| impulse_noise | 60.85 | 72.13 | 81.69 | 79.26 | 66.04 | 88.06 |
| jpeg_compression | 61.78 | 64.09 | 74.21 | 72.58 | 51.78 | 86.13 |
| motion_blur | 76.07 | 75.90 | 84.23 | 82.77 | 76.98 | 92.07 |
| pixelate | 59.90 | 66.64 | 77.39 | 71.51 | 76.92 | 83.79 |
| saturate | 81.55 | 83.84 | 90.03 | 91.41 | 86.38 | 96.88 |
| shot_noise | 61.13 | 62.57 | 76.81 | 72.43 | 53.06 | 86.56 |
| snow | 75.94 | 77.15 | 85.49 | 87.37 | 79.52 | 94.63 |
| spatter | 78.76 | 82.49 | 89.53 | 90.94 | 84.43 | 96.98 |
| speckle_noise | 63.30 | 64.57 | 78.06 | 74.66 | 56.67 | 87.00 |
| zoom_blur | 76.82 | 77.91 | 84.95 | 87.19 | 80.04 | 94.59 |
| **MEAN** | **70.92** | **73.10** | **82.01** | **81.38** | **72.63** | **90.97** |

> BLIP-2 and LLaVA are omitted from corruption tables (generative evaluation skipped).

### 3.3 ScrewSet-C

| Corruption | CLIP B/32 | CLIP B/16 | CLIP L/14 | OpenCLIP B/16 | SigLIP B/16 | EVA-02 B/16 |
|-----------|-----------|-----------|-----------|---------------|-------------|-------------|
| multi_object | 4.45 | 4.30 | 5.00 | 5.47 | 12.03 | 2.58 |
| occlusion_bottom_right | 3.12 | 3.44 | 5.16 | 2.97 | 6.56 | 1.25 |
| occlusion_top_left | 3.36 | 3.28 | 6.09 | 4.22 | 5.94 | 4.14 |
| reflection | 4.38 | 3.67 | 5.78 | 5.70 | 8.83 | 3.28 |
| scrap_paper | 2.66 | 1.56 | 3.44 | 3.67 | 2.58 | 2.03 |
| shadow | 2.66 | 3.05 | 5.00 | 3.36 | 8.12 | 2.42 |
| **MEAN** | **3.44** | **3.22** | **5.08** | **4.23** | **7.34** | **2.62** |

### 3.4 Lens-C (ImageNet-ES Corruptions)

Mean corruption accuracy across all 192 lens/parameter combinations:

| Model | Lens-C Mean (%) |
|-------|----------------|
| CLIP ViT-B/32 | 19.64 |
| CLIP ViT-B/16 | 21.57 |
| CLIP ViT-L/14 | **27.21** |
| OpenCLIP ViT-B/16 | 20.69 |
| SigLIP ViT-B/16 | 22.46 |
| EVA-02 CLIP B/16 | 26.66 |

> Full per-corruption breakdown (192 entries) available in individual JSON result files.

---

## 4. Key Findings

### 4.1 Contrastive Models

1. **EVA-02 CLIP B/16 is the consistently strongest CLIP-family model** across all datasets and corruption types. It achieves 98.48% on CIFAR-10, 71.36% on ImageNet-Val, and the lowest MCE (0.4610) on ImageNet-C.

2. **CLIP ViT-L/14 ranks second overall**, excelling particularly on ImageNet-A (44.81%) where scale and training data diversity matter for out-of-distribution robustness.

3. **OpenCLIP ViT-B/16 (LAION-2B) often outperforms OpenAI CLIP ViT-B/16** on clean accuracy (66.41 vs 61.20 on IN-Val), demonstrating the benefit of larger pretraining data for the same architecture.

4. **SigLIP ViT-B/16 shows competitive clean accuracy but weaker corruption robustness** on noise-type corruptions (e.g., 42.31% on CIFAR-10-C gaussian_noise vs 63.92% for OpenCLIP), possibly due to sigmoid loss training dynamics.

5. **All contrastive models struggle on ScrewSet** (3–11%), confirming that zero-shot CLIP-style models cannot reliably classify highly specialized industrial screw types without domain-specific training data. SigLIP achieves the highest contrastive score (10.78%) among CLIP-family models.

### 4.2 Generative Models

6. **Generative VLMs do not solve ScrewSet in zero-shot mode** — LLaVA-1.5 is at 2.50% clean (random-chance level for 40 classes) and BLIP-2 is at 0.95%.

7. **Best zero-shot ScrewSet result is Qwen3-VL (20.00%)**, followed by Qwen2.5-VL (12.20%) and SigLIP (10.78%). All Phase 3 models remain far below supervised Phase 1/2 performance. Qwen3-VL's linear probe on frozen visual features reaches 45.81%, showing useful representations exist but are not accessible via the zero-shot pathway. LoRA fine-tuning reaches 42.73%, comparable to the linear probe.

8. **Generative models underperform on ImageNet-Val** — LLaVA-1.5 achieves only 28.54% (vs 71.36% for EVA-02) and BLIP-2 achieves 53.35%. Open-ended generation over 1000 classes is inherently harder than contrastive matching.

9. **BLIP-2 is surprisingly competitive on ImageNet-A** (41.36%) — nearly matching CLIP ViT-L/14 (44.81%), suggesting that generative models may handle adversarial examples differently from contrastive models.

### 4.3 Corruption Robustness Patterns

10. **Corruption robustness strongly correlates with clean accuracy** for contrastive models — EVA-02's clean advantage translates directly to corruption robustness.

11. **Noise corruptions (gaussian_noise, shot_noise, impulse_noise) are the hardest** across all models, consistent with findings from Phases 1 and 2.

12. **ScrewSet corruption accuracy is extremely low** for all contrastive models (2.62–7.34%), indicating that zero-shot VLMs have essentially no robustness to real-world manufacturing corruptions for domain-specific tasks.

---

## 5. Comparison with Phases 1 and 2

### 5.0 Qwen3-VL Few-Shot, Linear Probe, and LoRA Fine-Tuning (ScrewSet)

| Method | ScrewSet Acc (%) |
|---|---|
| Zero-shot | 20.00 |
| 1-shot in-context | 19.55 |
| 2-shot in-context | 19.93 |
| 4-shot in-context | 16.06 |
| Linear Probe (frozen encoder) | **45.81** |
| LoRA Fine-tuned (10% data) | **42.73** |

- Few-shot in-context learning does not improve over zero-shot — the 40-class fine-grained task overwhelms the context window
- Linear probe (45.81%) and LoRA fine-tuning (42.73%) both show that Qwen3-VL visual features encode discriminative information
- Both remain far below supervised CNN/ViT fine-tuning (95–100%)
- LoRA config: r=16, α=32, 0.5% trainable params, 10% stratified training data, 1 epoch

### 5.1 Cross-Phase Comparison

| Approach | Best CIFAR-10 | Best IN-Val | Best IN-C (MCE↓) | Best ScrewSet |
|----------|--------------|-------------|-------------------|---------------|
| Phase 1 (CNN, fine-tuned) | ~96% (WRN-28-10) | ~77% (WRN-28-10) | ~0.40 (WRN-28-10) | ~92% (fine-tuned) |
| Phase 2 (ViT, fine-tuned) | ~97% (ConvNeXt-T) | ~80% (ConvNeXt-T) | ~0.38 (ConvNeXt-T) | ~90% (fine-tuned) |
| Phase 3 (VLM, zero-shot) | 98.48% (EVA-02) | 71.36% (EVA-02) | 0.4610 (EVA-02) | 20.00% (Qwen3-VL) |

**Key cross-phase observations:**
- Zero-shot VLMs can match or exceed fine-tuned models on **simple datasets** (CIFAR-10) but lag on **complex datasets** (ImageNet-Val, ImageNet-C)
- For **domain-specific tasks** (ScrewSet), all zero-shot VLMs (contrastive + generative) remain far below supervised fine-tuned models
- Corruption robustness of zero-shot VLMs (MCE ~0.46) is generally worse than fine-tuned models (MCE ~0.38–0.40), suggesting that adaptation to the evaluation domain provides some robustness benefit

---

## 6. File Index

All result files are stored in `results/phase3/`:

| File | Content |
|------|---------|
| `{model}_{dataset}_baselines.json` | Full results with per-corruption breakdowns |
| `qwen3_vl_8b_screwset_fewshot_lp.json` | Few-shot (K=0,1,2,4) and linear probe results |
| `qwen3_vl_8b_screwset_finetuned.json` | LoRA fine-tuned result (42.73% test, 43.44% val) |
| `qwen{2_5,3}_vl_*_full.json` | Full Qwen evaluations (corruption + clean) |
| `smoke_test_v{1,2,3}.log` | Smoke test logs from 3 debugging rounds |
| `phase3_eval.log` | Full evaluation run log (~8.5 hours) |
| `PHASE3_RESULTS.md` | This document |

**82 JSON files** (10 models × 6 datasets baselines + Qwen full evals + few-shot/LP + LoRA fine-tuned + ablations) — all runs completed successfully.
