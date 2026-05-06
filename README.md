# ScrewSet: A Large-Scale Benchmark for Fine-Grained Industrial Screw Classification and Corruption Robustness

Code and evaluation scripts for the NeurIPS 2026 Datasets & Benchmarks paper.

---

## Datasets

**Zenodo:** <https://zenodo.org/uploads/20058871>

| Archive | Contents |
|---------|----------|
| `screwset_split.tar.gz` | 109,200 images · 40 screw classes · train / val / test split |
| `screwset_c.tar.gz` | 7,680 images · 6 physical corruptions (shadow, reflection, occlusion top-left, occlusion bottom-right, scrap paper, multi-object) |

For CIFAR-10, ImageNet-C, ImageNet-A, and ScrewSet-S download details see the paper appendix.

---

## Setup

```bash
conda create -n screwset python=3.11 -y && conda activate screwset
pip install torch torchvision timm open_clip_torch transformers peft tqdm numpy scipy matplotlib
```

Expected data layout after download:

```
data/
├── screwset_split/        # clean ScrewSet (train/val/test)
├── screwset_c/            # ScrewSet-C physical corruptions
├── screwset_s/            # ScrewSet-S simulated corruptions (.npy)
├── cifar10/
├── cifar10_c/
├── imagenet-val/
├── imagenet-c/
└── imagenet-a/
```

---

## Replication

### Phase 1 — CNN baselines (ResNet, ConvNeXt, MobileViT, EfficientFormer)
```bash
python scripts/phase1/phase1_baselines.py --data-root data/screwset_split
```

### Phase 2 — ViT / DeiT / Swin baselines
```bash
python scripts/phase2/phase2_vit_baselines.py --data-root data/screwset_split
```

### Phase 3 — CLIP / SigLIP / DINOv2 / VLM evaluation
```bash
python scripts/phase3/phase3_vlm_baselines.py --data-root data/screwset_split
```

### ScrewSet-S evaluation
```bash
python scripts/eval_screwset_s.py --data-root data/screwset_s
```

### Augmentation baselines
```bash
bash scripts/augmentation/run_augmentation_screwset_s.sh
```

Results are written to `results/`.

---

## Repository structure

```
src/           core library (datasets, evaluation, corruption, config)
scripts/
  phase1/      CNN training & evaluation
  phase2/      ViT training & evaluation
  phase3/      CLIP / VLM zero-shot & fine-tuning
  augmentation/ augmentation ablation
  analysis/    per-class and per-corruption breakdown
results/       JSON result files and summary tables
```

---

## License

See [LICENSE](LICENSE).
