# AugMix Robustness Improvement Results

**Experiment:** Train 4 representative models on ScrewSet with [AugMix](https://arxiv.org/abs/1912.02781) data augmentation and evaluate on ScrewSet-C (6 corruption types).

**Goal:** Determine whether standard augmentation techniques can close the clean → corrupt accuracy gap observed in baseline experiments.

**AugMix Configuration:** severity=3, width=3, depth=-1 (random), all_ops=True

---

## Summary Table

| Model | Phase | Clean (Base) | Clean (AugMix) | Δ Clean | SS-C (Base) | SS-C (AugMix) | Δ SS-C | Gap Reduction |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | CNN | 98.02% | 98.24% | +0.22 | 9.23% | 26.05% | **+16.82** | 18.95% → partial |
| EfficientNetV2-S | CNN | 95.45% | 98.17% | +2.72 | 11.28% | 25.91% | **+14.63** | 15.87% → partial |
| ViT-Tiny | ViT | 99.66% | 99.79% | +0.14 | 21.17% | 41.56% | **+20.39** | 25.85% → partial |
| ConvNeXt-Tiny | ViT | 99.98% | 100.00% | +0.02 | 45.39% | 61.20% | **+15.81** | 21.72% → partial |
| **Average** | — | 98.28% | 99.05% | +0.77 | 21.77% | 38.68% | **+16.91** | — |

> **Key finding:** AugMix improves corruption robustness by **+14.6 to +20.4 percentage points** across all architectures, but a large gap remains (38.7% mean SS-C vs 99.1% clean accuracy). This confirms that ScrewSet-C corruptions are **structurally challenging** — not solvable by generic augmentation alone.

---

## Per-Corruption Breakdown

### Baseline vs AugMix per corruption type (accuracy %)

| Corruption | ResNet-18 Base | ResNet-18 AugMix | Δ | EffNetV2 Base | EffNetV2 AugMix | Δ | ViT-Tiny Base | ViT-Tiny AugMix | Δ | ConvNeXt Base | ConvNeXt AugMix | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multi_object | 18.05 | 20.39 | +2.34 | 25.16 | 24.53 | -0.63 | 45.23 | 45.78 | +0.55 | 53.52 | 59.30 | +5.78 |
| occlusion_br | 2.42 | 15.08 | +12.66 | 2.27 | 15.86 | +13.59 | 2.81 | 24.61 | +21.80 | 33.13 | 67.73 | +34.61 |
| occlusion_tl | 2.50 | 22.42 | +19.92 | 2.50 | 19.84 | +17.34 | 3.83 | 37.66 | +33.83 | 33.44 | 62.81 | +29.38 |
| reflection | 2.66 | 42.42 | +39.77 | 2.50 | 34.84 | +32.34 | 25.23 | 41.41 | +16.17 | 43.20 | 54.77 | +11.56 |
| scrap_paper | 23.83 | 37.42 | +13.59 | 32.73 | 40.86 | +8.13 | 33.98 | 76.64 | +42.66 | 76.02 | 87.89 | +11.87 |
| shadow | 5.94 | 18.59 | +12.66 | 2.50 | 19.53 | +17.03 | 15.94 | 23.28 | +7.34 | 33.05 | 34.69 | +1.64 |

### Observations by corruption type

- **Reflection:** Largest improvement for CNNs (+39.8pp ResNet-18, +32.3pp EfficientNetV2-S), moderate for ViTs (+11–16pp). AugMix's color/contrast jittering helps with reflective surfaces.
- **Occlusion (top-left & bottom-right):** Consistent strong improvements across all architectures (+12–35pp). AugMix's mixing of augmented images provides partial occlusion robustness.
- **Scrap paper:** ViT-Tiny shows largest gain (+42.7pp). Likely benefits from AugMix's patch-level diversity combined with ViT's patch-based attention.
- **Multi-object:** Minimal improvement for CNNs (~±1pp), small for ViTs (+0.5–5.8pp). Multi-object confusion is a **semantic** challenge that AugMix's pixel-level augmentations cannot address.
- **Shadow:** Moderate improvements for CNNs (+12–17pp) but small for ViTs (+1.6–7.3pp). Shadow remains the hardest corruption for ConvNeXt even with AugMix.

---

## Training Configuration

| Parameter | CNN Models (ResNet-18, EfficientNetV2-S) | ViT Models (ViT-Tiny, ConvNeXt-Tiny) |
|---|---|---|
| Optimizer | Adam | AdamW |
| Learning Rate | 1e-3 | 5e-4 |
| Weight Decay | 0 | 0.05 |
| Epochs | 20 (early stop patience=5) | 30 |
| Warmup | — | 5 epochs |
| Scheduler | ReduceLROnPlateau | Cosine Warmup |
| Pretrained | No (from scratch) | Yes (ImageNet) |
| AMP | No | Yes |
| Batch Size | 256 | 256 |
| Seed | 42 | 42 |

---

## Conclusion

AugMix provides a meaningful but **partial** robustness improvement on ScrewSet-C. The average corruption accuracy increases from 21.8% → 38.7% (still a **60.3pp gap** from clean accuracy). This supports the thesis that **ScrewSet-C corruptions are domain-specific, structurally challenging perturbations** that require targeted robustness strategies beyond generic data augmentation.

The corruption-specific analysis reveals that:
1. **Pixel-level corruptions** (reflection, shadow, occlusion) benefit most from AugMix
2. **Semantic corruptions** (multi_object) are largely unaffected
3. **ViTs generally benefit more** than CNNs from AugMix, with ViT-Tiny showing the largest average improvement (+20.4pp)
4. ConvNeXt-Tiny remains the most robust architecture overall (61.2% SS-C with AugMix)
