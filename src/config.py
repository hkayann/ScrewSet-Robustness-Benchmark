"""
Global paths, seeds, and constant registries shared across all phases.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Dataset list ─────────────────────────────────────────────────────────────
ALL_DATASETS = [
    "cifar10", "screwset", "imagenet_a", "imagenet_val", "imagenet_c", "lens",
]

# ── CIFAR-10 class names ─────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── ImageNet-C corruption types ──────────────────────────────────────────────
IMAGENET_C_CORRUPTIONS_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise",           # noise
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",  # blur
    "snow", "frost", "fog", "brightness",                      # weather
    "contrast", "elastic_transform", "pixelate",               # digital
    "jpeg_compression",
]

IMAGENET_C_CORRUPTIONS_EXTRA = [
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

# ── CNN pretrained weight IDs (Phase 1) ──────────────────────────────────────
CNN_TIMM_PRETRAINED_IDS = {
    "mobilenetv4_conv_small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "efficientnetv2_rw_s": "efficientnetv2_rw_s.ra2_in1k",
    "ghostnetv2_100": "ghostnetv2_100.in1k",
    "convnextv2_atto": "convnextv2_atto.fcmae_ft_in1k",
}

# ── ViT pretrained weight IDs (Phase 2) ─────────────────────────────────────
VIT_TIMM_PRETRAINED_IDS = {
    "vit_tiny_patch16_224": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "deit_tiny_patch16_224": "deit_tiny_patch16_224.fb_in1k",
    "deit_small_patch16_224": "deit_small_patch16_224.fb_in1k",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
    "efficientformer_l1": "efficientformer_l1.snap_dist_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
}

# ── VLM model registry (Phase 3) ────────────────────────────────────────────
VLM_MODELS = {
    "clip_vit_b32": {
        "backend": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "family": "CLIP (OpenAI)",
        "params": "151M",
    },
    "clip_vit_b16": {
        "backend": "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "family": "CLIP (OpenAI)",
        "params": "150M",
    },
    "clip_vit_l14": {
        "backend": "open_clip",
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "family": "CLIP (OpenAI)",
        "params": "428M",
    },
    "openclip_vit_b16": {
        "backend": "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "family": "OpenCLIP (LAION-2B)",
        "params": "150M",
    },
    "siglip_vit_b16": {
        "backend": "open_clip",
        "model_name": "ViT-B-16-SigLIP",
        "pretrained": "webli",
        "family": "SigLIP (Google)",
        "params": "150M",
    },
    "eva02_clip_vit_b16": {
        "backend": "open_clip",
        "model_name": "EVA02-B-16",
        "pretrained": "merged2b_s8b_b131k",
        "family": "EVA-02-CLIP",
        "params": "150M",
    },
    "blip2": {
        "backend": "blip2",
        "model_id": "Salesforce/blip2-opt-2.7b",
        "family": "BLIP-2",
        "params": "3.7B",
    },
    "llava": {
        "backend": "llava",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "family": "LLaVA-1.5",
        "params": "7B",
    },
}
