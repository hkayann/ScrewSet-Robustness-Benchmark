#!/usr/bin/env python3
"""
Per-corruption analysis across all phases and datasets.

Produces:
  1. Heatmaps         -- model x corruption accuracy (ScrewSet-C, CIFAR-10-C)
  2. Radar charts     -- per-corruption profiles for representative models
  3. Correlation       -- cross-dataset robustness correlation scatter
  4. mCE tables       -- mean Corruption Error (ResNet-18 reference)
  5. Robustness gap   -- clean vs corrupt accuracy gap analysis
  6. AugMix comparison -- baseline vs AugMix per-corruption delta heatmap
  7. CSV exports       -- all tables saved for paper LaTeX import

Outputs saved to results/figures/
"""
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Model display names
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    # Phase 1 - CNNs
    "resnet18": "ResNet-18",
    "efficientnetv2_rw_s": "EfficientNetV2-S",
    "mobilenet_v3_large": "MobileNetV3-L",
    "mobilenetv4_conv_small": "MobileNetV4-S",
    "shufflenet_v2_x1_0": "ShuffleNetV2",
    "squeezenet1_1": "SqueezeNet",
    "ghostnetv2_100": "GhostNetV2",
    "convnextv2_atto": "ConvNeXtV2-A",
    # Phase 2 - ViTs
    "vit_tiny_patch16_224": "ViT-Tiny",
    "vit_small_patch16_224": "ViT-Small",
    "deit_tiny_patch16_224": "DeiT-Tiny",
    "deit_small_patch16_224": "DeiT-Small",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientformer_l1": "EfficientFormer",
    "mobilevit_s": "MobileViT-S",
    # Phase 3 - VLMs
    "clip_vit_b32": "CLIP-B/32",
    "clip_vit_b16": "CLIP-B/16",
    "clip_vit_l14": "CLIP-L/14",
    "openclip_vit_b16": "OpenCLIP-B/16",
    "siglip_vit_b16": "SigLIP-B/16",
    "eva02_clip_vit_b16": "EVA02-CLIP-B/16",
    "blip2": "BLIP-2",
    "llava": "LLaVA-1.5",
}

CORRUPTION_DISPLAY = {
    "screwset_multi_object": "Multi-object",
    "screwset_occlusion_bottom_right": "Occlusion BR",
    "screwset_occlusion_top_left": "Occlusion TL",
    "screwset_reflection": "Reflection",
    "screwset_scrap_paper": "Scrap paper",
    "screwset_shadow": "Shadow",
}

SCREWSET_CORRUPTIONS = list(CORRUPTION_DISPLAY.keys())

PHASE_LABELS = {
    "phase1": "CNN",
    "phase2": "ViT",
    "phase3": "VLM",
}

# Ordering for consistent display
PHASE1_MODELS = [
    "resnet18", "efficientnetv2_rw_s", "mobilenet_v3_large",
    "mobilenetv4_conv_small", "shufflenet_v2_x1_0", "squeezenet1_1",
    "ghostnetv2_100", "convnextv2_atto",
]
PHASE2_MODELS = [
    "vit_tiny_patch16_224", "vit_small_patch16_224",
    "deit_tiny_patch16_224", "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224", "convnext_tiny",
    "efficientformer_l1", "mobilevit_s",
]
PHASE3_MODELS = [
    "clip_vit_b32", "clip_vit_b16", "clip_vit_l14",
    "openclip_vit_b16", "siglip_vit_b16", "eva02_clip_vit_b16",
    "blip2", "llava",
]
ALL_MODELS = PHASE1_MODELS + PHASE2_MODELS + PHASE3_MODELS

# CIFAR-10-C standard 15 corruptions (matching RobustBench)
CIFAR10_CORRUPTIONS_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_jsons():
    """Load every non-stale JSON into a list of dicts with source metadata."""
    records = []
    for phase_dir in ["phase1", "phase2", "phase3", "augmix"]:
        pattern = str(RESULTS / phase_dir / "*.json")
        for fp in sorted(glob.glob(pattern)):
            if "/_stale/" in fp or "_stale" in os.path.basename(fp):
                continue
            if "vqa_ablation" in fp:
                continue
            with open(fp) as f:
                d = json.load(f)
            d["_source_file"] = fp
            d["_phase_dir"] = phase_dir
            records.append(d)
    return records


def build_screwset_table(records):
    """
    Build a DataFrame: rows = models, columns = [clean, corruption1, ..., mean_corrupt].
    Only ScrewSet results (screwset dataset, non-augmix).
    """
    rows = []
    for d in records:
        ds = d.get("dataset", "")
        if ds != "screwset":
            continue
        if d["_phase_dir"] == "augmix":
            continue
        model = d.get("model", "")
        clean = d.get("test_acc", 0.0)
        cr = d.get("corrupt_results", {})
        mean_c = d.get("mean_corrupt_acc", 0.0)
        row = {"model": model, "clean": clean, "mean_corrupt": mean_c}
        for c in SCREWSET_CORRUPTIONS:
            row[c] = cr.get(c, np.nan)
        phase = d["_phase_dir"]
        row["phase"] = phase
        rows.append(row)
    df = pd.DataFrame(rows)
    # Order by canonical model list
    model_order = {m: i for i, m in enumerate(ALL_MODELS)}
    df["_order"] = df["model"].map(model_order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    df["display_name"] = df["model"].map(MODEL_DISPLAY)
    return df


def build_cifar10_corruption_table(records):
    """
    Build a DataFrame: rows = models, columns = [clean, corruption1, ..., mean_corrupt].
    Only CIFAR-10 results (phases 1 & 2).
    """
    rows = []
    for d in records:
        ds = d.get("dataset", "")
        if ds != "cifar10":
            continue
        if d["_phase_dir"] == "augmix":
            continue
        model = d.get("model", "")
        clean = d.get("test_acc", 0.0)
        cr = d.get("corrupt_results", {})
        mean_c = d.get("mean_corrupt_acc", 0.0)
        row = {"model": model, "clean": clean, "mean_corrupt": mean_c}
        for c in CIFAR10_CORRUPTIONS_15:
            val = cr.get(c, np.nan)
            # Handle nested dict format: {"severity_accs": [...], "mean": float}
            if isinstance(val, dict):
                val = val.get("mean", np.nan)
            row[c] = val
        phase = d["_phase_dir"]
        row["phase"] = phase
        rows.append(row)
    df = pd.DataFrame(rows)
    model_order = {m: i for i, m in enumerate(ALL_MODELS)}
    df["_order"] = df["model"].map(model_order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    df["display_name"] = df["model"].map(MODEL_DISPLAY)
    return df


def build_augmix_table(records):
    """Build DataFrame for AugMix results."""
    rows = []
    for d in records:
        if d["_phase_dir"] != "augmix":
            continue
        model = d.get("model", "")
        clean = d.get("test_acc", 0.0)
        cr = d.get("corrupt_results", {})
        mean_c = d.get("mean_corrupt_acc", 0.0)
        row = {"model": model, "clean": clean, "mean_corrupt": mean_c}
        for c in SCREWSET_CORRUPTIONS:
            row[c] = cr.get(c, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["display_name"] = df["model"].map(MODEL_DISPLAY)
    return df


def build_cross_dataset_table(records):
    """
    Build a summary table: model x dataset -> (clean_acc, mean_corrupt_acc).
    For cross-dataset correlation analysis.
    """
    rows = []
    for d in records:
        if d["_phase_dir"] == "augmix":
            continue
        if "vqa_ablation" in d.get("_source_file", ""):
            continue
        model = d.get("model", "")
        ds = d.get("dataset", "")
        clean = d.get("test_acc")
        mean_c = d.get("mean_corrupt_acc")
        # ImageNet-C has different schema
        if ds == "imagenet_c":
            mean_c = d.get("mean_acc_15_std") or d.get("mean_acc_all")
            clean = None  # no clean acc in imagenet_c files
        rows.append({
            "model": model, "dataset": ds,
            "clean_acc": clean, "mean_corrupt_acc": mean_c,
            "phase": d["_phase_dir"],
        })
    df = pd.DataFrame(rows)
    df["display_name"] = df["model"].map(MODEL_DISPLAY)
    return df


# ---------------------------------------------------------------------------
# Figure 1: ScrewSet-C heatmap (model x corruption)
# ---------------------------------------------------------------------------

def fig_screwset_heatmap(df_ss):
    """Heatmap of per-corruption accuracy on ScrewSet-C, all 24 models."""
    corr_cols = SCREWSET_CORRUPTIONS
    disp_cols = [CORRUPTION_DISPLAY[c] for c in corr_cols]

    mat = df_ss[corr_cols].values * 100
    labels = df_ss["display_name"].tolist()

    # Add phase separators via color bar
    phases = df_ss["phase"].tolist()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = sns.heatmap(
        mat, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=disp_cols, yticklabels=labels,
        vmin=0, vmax=100, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_title("ScrewSet-C Per-Corruption Accuracy (%)")
    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("")

    # Draw horizontal separators between phases
    n_p1 = sum(1 for p in phases if p == "phase1")
    n_p2 = sum(1 for p in phases if p == "phase2")
    ax.axhline(n_p1, color="black", linewidth=2)
    ax.axhline(n_p1 + n_p2, color="black", linewidth=2)
    # Phase labels on the right
    ax.text(len(disp_cols) + 0.3, n_p1 / 2, "CNN", va="center", fontsize=10,
            fontweight="bold", rotation=-90)
    ax.text(len(disp_cols) + 0.3, n_p1 + n_p2 / 2, "ViT", va="center",
            fontsize=10, fontweight="bold", rotation=-90)
    ax.text(len(disp_cols) + 0.3, n_p1 + n_p2 + (len(phases) - n_p1 - n_p2) / 2,
            "VLM", va="center", fontsize=10, fontweight="bold", rotation=-90)

    plt.tight_layout()
    out = FIGURES / "screwset_c_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: CIFAR-10-C heatmap (model x corruption, phases 1+2 only)
# ---------------------------------------------------------------------------

def fig_cifar10_heatmap(df_c10):
    """Heatmap of per-corruption accuracy on CIFAR-10-C, 16 models."""
    corr_cols = CIFAR10_CORRUPTIONS_15
    disp_cols = [c.replace("_", " ").title() for c in corr_cols]

    mat = df_c10[corr_cols].values * 100
    labels = df_c10["display_name"].tolist()
    phases = df_c10["phase"].tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        mat, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=disp_cols, yticklabels=labels,
        vmin=0, vmax=100, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_title("CIFAR-10-C Per-Corruption Accuracy (%)")
    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("")

    n_p1 = sum(1 for p in phases if p == "phase1")
    ax.axhline(n_p1, color="black", linewidth=2)
    ax.text(len(disp_cols) + 0.3, n_p1 / 2, "CNN", va="center", fontsize=10,
            fontweight="bold", rotation=-90)
    ax.text(len(disp_cols) + 0.3, n_p1 + (len(phases) - n_p1) / 2, "ViT",
            va="center", fontsize=10, fontweight="bold", rotation=-90)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = FIGURES / "cifar10_c_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 3: Radar charts -- per-corruption profiles
# ---------------------------------------------------------------------------

def fig_radar_screwset(df_ss):
    """Radar chart comparing CNN / ViT / VLM representative models on ScrewSet-C."""
    # Pick representative models: best per phase + worst per phase
    representatives = {
        "CNN": ["resnet18", "convnextv2_atto", "efficientnetv2_rw_s"],
        "ViT": ["vit_tiny_patch16_224", "convnext_tiny", "swin_tiny_patch4_window7_224"],
        "VLM": ["clip_vit_l14", "siglip_vit_b16", "eva02_clip_vit_b16"],
    }

    corr_cols = SCREWSET_CORRUPTIONS
    labels_c = [CORRUPTION_DISPLAY[c] for c in corr_cols]
    N = len(corr_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors_per_phase = {"CNN": "tab:blue", "ViT": "tab:orange", "VLM": "tab:green"}
    linestyles = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for phase_name, models in representatives.items():
        for i, m in enumerate(models):
            row = df_ss[df_ss["model"] == m]
            if row.empty:
                continue
            vals = (row[corr_cols].values[0] * 100).tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linestyles[i],
                    color=colors_per_phase[phase_name], linewidth=1.8,
                    label=f"{MODEL_DISPLAY.get(m, m)} ({phase_name})")
            ax.fill(angles, vals, alpha=0.05, color=colors_per_phase[phase_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_c, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], size=8)
    ax.set_title("ScrewSet-C Per-Corruption Profiles\n(Selected Models)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    out = FIGURES / "screwset_c_radar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_radar_cnn_vs_vit(df_ss):
    """Radar chart: all CNNs vs all ViTs (phase means) on ScrewSet-C."""
    corr_cols = SCREWSET_CORRUPTIONS
    labels_c = [CORRUPTION_DISPLAY[c] for c in corr_cols]
    N = len(corr_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    phase_means = {}
    for phase_name, models in [("CNN", PHASE1_MODELS), ("ViT", PHASE2_MODELS), ("VLM", PHASE3_MODELS)]:
        sub = df_ss[df_ss["model"].isin(models)]
        if sub.empty:
            continue
        phase_means[phase_name] = (sub[corr_cols].mean().values * 100).tolist()

    colors = {"CNN": "#1f77b4", "ViT": "#ff7f0e", "VLM": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for phase_name, vals in phase_means.items():
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, "-o", color=colors[phase_name],
                linewidth=2, markersize=5, label=f"{phase_name} mean")
        ax.fill(angles, vals_closed, alpha=0.1, color=colors[phase_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_c, size=9)
    ax.set_ylim(0, 80)
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%"], size=8)
    ax.set_title("ScrewSet-C: Phase-Averaged Corruption Profiles", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    out = FIGURES / "screwset_c_radar_phase_mean.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 4: Robustness gap bar chart
# ---------------------------------------------------------------------------

def fig_robustness_gap(df_ss):
    """Grouped bar chart: clean acc vs mean corrupt acc per model."""
    labels = df_ss["display_name"].tolist()
    clean = df_ss["clean"].values * 100
    corrupt = df_ss["mean_corrupt"].values * 100
    gap = clean - corrupt

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - w / 2, clean, w, label="Clean Accuracy", color="#4c72b0")
    bars2 = ax.bar(x + w / 2, corrupt, w, label="Mean SS-C Accuracy", color="#dd8452")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ScrewSet: Clean vs Corruption Accuracy Gap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 105)

    # Annotate gap
    for i in range(len(labels)):
        ax.annotate(
            f"{gap[i]:.0f}pp",
            xy=(x[i], max(clean[i], corrupt[i]) + 1),
            ha="center", fontsize=6, color="gray",
        )

    # Phase separators
    phases = df_ss["phase"].tolist()
    n_p1 = sum(1 for p in phases if p == "phase1")
    n_p2 = sum(1 for p in phases if p == "phase2")
    ax.axvline(n_p1 - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(n_p1 + n_p2 - 0.5, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    out = FIGURES / "robustness_gap_bar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 5: Cross-dataset correlation scatter
# ---------------------------------------------------------------------------

def fig_cross_dataset_correlation(df_cross):
    """
    Scatter: ScrewSet-C mean acc vs CIFAR-10-C / ImageNet-C mean acc.
    Shows whether robustness transfers across domains.
    """
    # Pivot to get one row per model with columns for each dataset
    pivot = df_cross.pivot_table(
        index="model", columns="dataset",
        values="mean_corrupt_acc", aggfunc="first"
    )
    pivot["display_name"] = pivot.index.map(MODEL_DISPLAY)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- ScrewSet-C vs CIFAR-10-C ---
    ax = axes[0]
    x_col, y_col = "screwset", "cifar10"
    if x_col in pivot.columns and y_col in pivot.columns:
        sub = pivot.dropna(subset=[x_col, y_col])
        x_vals = sub[x_col].values * 100
        y_vals = sub[y_col].values * 100

        # Color by phase
        for m_list, color, label in [
            (PHASE1_MODELS, "#1f77b4", "CNN"),
            (PHASE2_MODELS, "#ff7f0e", "ViT"),
            (PHASE3_MODELS, "#2ca02c", "VLM"),
        ]:
            mask = sub.index.isin(m_list)
            ax.scatter(x_vals[mask], y_vals[mask], c=color, s=50, label=label,
                       edgecolors="black", linewidths=0.5, zorder=3)
            for idx in sub.index[mask]:
                ax.annotate(MODEL_DISPLAY.get(idx, idx),
                            (sub.loc[idx, x_col] * 100, sub.loc[idx, y_col] * 100),
                            fontsize=6, textcoords="offset points", xytext=(4, 4))

        if len(x_vals) > 2:
            r, p_val = stats.pearsonr(x_vals, y_vals)
            # Fit line
            m_fit, b_fit = np.polyfit(x_vals, y_vals, 1)
            xline = np.linspace(x_vals.min() - 2, x_vals.max() + 2, 100)
            ax.plot(xline, m_fit * xline + b_fit, "--", color="gray", linewidth=1)
            ax.set_title(f"SS-C vs CIFAR-10-C  (r={r:.3f}, p={p_val:.3e})")
        else:
            ax.set_title("SS-C vs CIFAR-10-C")

        ax.set_xlabel("ScrewSet-C Mean Accuracy (%)")
        ax.set_ylabel("CIFAR-10-C Mean Accuracy (%)")
        ax.legend(fontsize=8)

    # --- ScrewSet-C vs ImageNet-C ---
    ax = axes[1]
    y_col = "imagenet_c"
    if x_col in pivot.columns and y_col in pivot.columns:
        sub = pivot.dropna(subset=[x_col, y_col])
        x_vals = sub[x_col].values * 100
        y_vals = sub[y_col].values * 100

        for m_list, color, label in [
            (PHASE1_MODELS, "#1f77b4", "CNN"),
            (PHASE2_MODELS, "#ff7f0e", "ViT"),
        ]:
            mask = sub.index.isin(m_list)
            ax.scatter(x_vals[mask], y_vals[mask], c=color, s=50, label=label,
                       edgecolors="black", linewidths=0.5, zorder=3)
            for idx in sub.index[mask]:
                ax.annotate(MODEL_DISPLAY.get(idx, idx),
                            (sub.loc[idx, x_col] * 100, sub.loc[idx, y_col] * 100),
                            fontsize=6, textcoords="offset points", xytext=(4, 4))

        if len(x_vals) > 2:
            r, p_val = stats.pearsonr(x_vals, y_vals)
            m_fit, b_fit = np.polyfit(x_vals, y_vals, 1)
            xline = np.linspace(x_vals.min() - 2, x_vals.max() + 2, 100)
            ax.plot(xline, m_fit * xline + b_fit, "--", color="gray", linewidth=1)
            ax.set_title(f"SS-C vs ImageNet-C  (r={r:.3f}, p={p_val:.3e})")
        else:
            ax.set_title("SS-C vs ImageNet-C")

        ax.set_xlabel("ScrewSet-C Mean Accuracy (%)")
        ax.set_ylabel("ImageNet-C Mean Accuracy (%)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES / "cross_dataset_correlation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 6: Model-corruption correlation matrix (ScrewSet-C)
# ---------------------------------------------------------------------------

def fig_corruption_correlation(df_ss):
    """Correlation matrix between corruption types across all models."""
    corr_cols = SCREWSET_CORRUPTIONS
    disp_cols = [CORRUPTION_DISPLAY[c] for c in corr_cols]

    corr_mat = df_ss[corr_cols].corr()
    corr_mat.columns = disp_cols
    corr_mat.index = disp_cols

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, mask=mask,
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("ScrewSet-C: Corruption-Corruption Correlation\n(across 24 models)")
    plt.tight_layout()
    out = FIGURES / "screwset_c_corruption_correlation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 7: Model-model correlation matrix (ScrewSet-C)
# ---------------------------------------------------------------------------

def fig_model_correlation(df_ss):
    """Correlation matrix between models across ScrewSet-C corruptions."""
    corr_cols = SCREWSET_CORRUPTIONS
    mat = df_ss.set_index("display_name")[corr_cols].T
    corr_mat = mat.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, mask=mask,
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Pearson r"},
        annot_kws={"size": 7},
    )
    ax.set_title("ScrewSet-C: Model-Model Correlation\n(across 6 corruption types)")
    plt.tight_layout()
    out = FIGURES / "screwset_c_model_correlation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 8: AugMix delta heatmap
# ---------------------------------------------------------------------------

def fig_augmix_delta(df_ss, df_aug):
    """Heatmap showing per-corruption improvement from AugMix."""
    if df_aug.empty:
        print("  [SKIP] No AugMix results found.")
        return

    augmix_models = df_aug["model"].tolist()
    corr_cols = SCREWSET_CORRUPTIONS
    disp_cols = [CORRUPTION_DISPLAY[c] for c in corr_cols]

    rows_delta = []
    rows_base = []
    rows_aug = []
    labels = []
    for m in augmix_models:
        base_row = df_ss[df_ss["model"] == m]
        aug_row = df_aug[df_aug["model"] == m]
        if base_row.empty or aug_row.empty:
            continue
        base_vals = base_row[corr_cols].values[0] * 100
        aug_vals = aug_row[corr_cols].values[0] * 100
        delta = aug_vals - base_vals
        rows_delta.append(delta)
        rows_base.append(base_vals)
        rows_aug.append(aug_vals)
        labels.append(MODEL_DISPLAY.get(m, m))

    if not rows_delta:
        print("  [SKIP] No matching AugMix baselines.")
        return

    mat_delta = np.array(rows_delta)
    mat_base = np.array(rows_base)
    mat_aug = np.array(rows_aug)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Baseline
    sns.heatmap(mat_base, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=disp_cols, yticklabels=labels,
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                ax=axes[0], cbar_kws={"label": "%"})
    axes[0].set_title("Baseline Accuracy (%)")

    # AugMix
    sns.heatmap(mat_aug, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=disp_cols, yticklabels=labels,
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                ax=axes[1], cbar_kws={"label": "%"})
    axes[1].set_title("AugMix Accuracy (%)")

    # Delta
    sns.heatmap(mat_delta, annot=True, fmt="+.1f", cmap="RdYlGn",
                xticklabels=disp_cols, yticklabels=labels,
                vmin=-5, vmax=45, linewidths=0.5, linecolor="white",
                ax=axes[2], cbar_kws={"label": "pp"})
    axes[2].set_title("AugMix Improvement (pp)")

    plt.suptitle("AugMix Effect on ScrewSet-C Per-Corruption Accuracy", fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIGURES / "augmix_delta_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 9: mCE bar chart (ScrewSet-C)
# ---------------------------------------------------------------------------

def compute_mce(df_ss, reference_model="resnet18"):
    """
    Compute mean Corruption Error (mCE) relative to a reference model.
    CE_c^f = (1 - acc_c^f) / (1 - acc_c^ref)
    mCE = mean over corruptions of CE_c^f
    """
    ref_row = df_ss[df_ss["model"] == reference_model]
    if ref_row.empty:
        print(f"  [WARN] Reference model {reference_model} not found. Skipping mCE.")
        return pd.DataFrame()

    corr_cols = SCREWSET_CORRUPTIONS
    ref_accs = ref_row[corr_cols].values[0]

    rows = []
    for _, r in df_ss.iterrows():
        model_accs = r[corr_cols].values.astype(float)
        ces = []
        for j, c in enumerate(corr_cols):
            ref_err = 1.0 - ref_accs[j]
            model_err = 1.0 - model_accs[j]
            if ref_err > 0:
                ces.append(model_err / ref_err)
            else:
                ces.append(0.0)
        mce = np.mean(ces)
        per_corruption = {CORRUPTION_DISPLAY[c]: ces[j] for j, c in enumerate(corr_cols)}
        rows.append({
            "model": r["model"],
            "display_name": r["display_name"],
            "phase": r["phase"],
            "mCE": mce,
            **per_corruption,
        })
    df_mce = pd.DataFrame(rows).sort_values("mCE")
    return df_mce


def fig_mce_bar(df_mce):
    """Bar chart of mCE values."""
    if df_mce.empty:
        return

    colors_phase = {"phase1": "#1f77b4", "phase2": "#ff7f0e", "phase3": "#2ca02c"}
    bar_colors = [colors_phase.get(p, "gray") for p in df_mce["phase"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_mce))
    ax.barh(x, df_mce["mCE"].values, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(x)
    ax.set_yticklabels(df_mce["display_name"].tolist(), fontsize=8)
    ax.set_xlabel("mCE (relative to ResNet-18)")
    ax.set_title("ScrewSet-C: Mean Corruption Error (mCE)\nLower = More Robust")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="ResNet-18 baseline")
    ax.invert_yaxis()

    # Legend for phases
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="CNN"),
        Patch(facecolor="#ff7f0e", label="ViT"),
        Patch(facecolor="#2ca02c", label="VLM"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Reference (mCE=1.0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    out = FIGURES / "screwset_c_mce_bar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 10: Relative Corruption Accuracy (Clean-normalized) stacked
# ---------------------------------------------------------------------------

def fig_relative_drop(df_ss):
    """
    Stacked bar showing clean acc and per-corruption drop for each model.
    Visualizes which corruptions cause the most damage per model.
    """
    corr_cols = SCREWSET_CORRUPTIONS
    disp_cols = [CORRUPTION_DISPLAY[c] for c in corr_cols]

    # Compute relative accuracy retained = corrupt_acc / clean_acc
    labels = df_ss["display_name"].tolist()
    clean = df_ss["clean"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))

    bottom = np.zeros(len(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(corr_cols)))

    for j, c in enumerate(corr_cols):
        vals = df_ss[c].values
        # relative retained
        rel = np.where(clean > 0, vals / clean, 0) * 100
        ax.bar(x, rel, bottom=bottom, color=colors[j], label=disp_cols[j],
               edgecolor="white", linewidth=0.3)
        bottom += rel

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative Relative Accuracy Retained (%)")
    ax.set_title("ScrewSet-C: Per-Corruption Accuracy Retained (Relative to Clean)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # Phase separators
    phases = df_ss["phase"].tolist()
    n_p1 = sum(1 for p in phases if p == "phase1")
    n_p2 = sum(1 for p in phases if p == "phase2")
    ax.axvline(n_p1 - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(n_p1 + n_p2 - 0.5, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    out = FIGURES / "screwset_c_relative_drop.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def export_csvs(df_ss, df_c10, df_mce, df_aug, df_cross):
    """Export all tables as CSVs for LaTeX import."""
    csv_dir = FIGURES / "csv"
    csv_dir.mkdir(exist_ok=True)

    # ScrewSet-C per-corruption table
    cols = ["display_name", "phase", "clean", "mean_corrupt"] + SCREWSET_CORRUPTIONS
    df_out = df_ss[cols].copy()
    df_out.columns = (
        ["Model", "Phase", "Clean", "Mean SS-C"]
        + [CORRUPTION_DISPLAY[c] for c in SCREWSET_CORRUPTIONS]
    )
    # Convert to percentages
    for c in df_out.columns[2:]:
        df_out[c] = (df_out[c] * 100).round(2)
    df_out.to_csv(csv_dir / "screwset_c_per_corruption.csv", index=False)

    # CIFAR-10-C per-corruption table
    if not df_c10.empty:
        cols_c10 = ["display_name", "phase", "clean", "mean_corrupt"] + CIFAR10_CORRUPTIONS_15
        df_c10_out = df_c10[cols_c10].copy()
        rename_map = {c: c.replace("_", " ").title() for c in CIFAR10_CORRUPTIONS_15}
        rename_map["display_name"] = "Model"
        rename_map["phase"] = "Phase"
        rename_map["clean"] = "Clean"
        rename_map["mean_corrupt"] = "Mean C10-C"
        df_c10_out = df_c10_out.rename(columns=rename_map)
        for c in df_c10_out.columns[2:]:
            df_c10_out[c] = (df_c10_out[c] * 100).round(2)
        df_c10_out.to_csv(csv_dir / "cifar10_c_per_corruption.csv", index=False)

    # mCE table
    if not df_mce.empty:
        df_mce_out = df_mce.copy()
        df_mce_out["mCE"] = df_mce_out["mCE"].round(4)
        for c in [CORRUPTION_DISPLAY[cc] for cc in SCREWSET_CORRUPTIONS]:
            if c in df_mce_out.columns:
                df_mce_out[c] = df_mce_out[c].round(4)
        df_mce_out.to_csv(csv_dir / "screwset_c_mce.csv", index=False)

    # AugMix comparison
    if not df_aug.empty:
        aug_rows = []
        for _, aug_r in df_aug.iterrows():
            base = df_ss[df_ss["model"] == aug_r["model"]]
            if base.empty:
                continue
            row = {"Model": MODEL_DISPLAY.get(aug_r["model"], aug_r["model"])}
            row["Clean (Base)"] = round(base["clean"].values[0] * 100, 2)
            row["Clean (AugMix)"] = round(aug_r["clean"] * 100, 2)
            row["SS-C (Base)"] = round(base["mean_corrupt"].values[0] * 100, 2)
            row["SS-C (AugMix)"] = round(aug_r["mean_corrupt"] * 100, 2)
            row["Delta SS-C"] = round(
                (aug_r["mean_corrupt"] - base["mean_corrupt"].values[0]) * 100, 2
            )
            for c in SCREWSET_CORRUPTIONS:
                cn = CORRUPTION_DISPLAY[c]
                row[f"{cn} (Base)"] = round(base[c].values[0] * 100, 2)
                row[f"{cn} (AugMix)"] = round(aug_r[c] * 100, 2)
                row[f"{cn} (Delta)"] = round(
                    (aug_r[c] - base[c].values[0]) * 100, 2
                )
            aug_rows.append(row)
        pd.DataFrame(aug_rows).to_csv(csv_dir / "augmix_comparison.csv", index=False)

    # Cross-dataset summary
    pivot = df_cross.pivot_table(
        index="model", columns="dataset",
        values=["clean_acc", "mean_corrupt_acc"], aggfunc="first"
    )
    pivot.columns = [f"{v}_{d}" for v, d in pivot.columns]
    pivot["display_name"] = pivot.index.map(MODEL_DISPLAY)
    pivot = pivot.reset_index()
    # Round
    for c in pivot.columns:
        if pivot[c].dtype in [np.float64, np.float32]:
            pivot[c] = (pivot[c] * 100).round(2)
    pivot.to_csv(csv_dir / "cross_dataset_summary.csv", index=False)

    print(f"  CSVs saved to {csv_dir}/")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df_ss, df_mce):
    """Print key findings to stdout."""
    print("\n" + "=" * 72)
    print("PER-CORRUPTION ANALYSIS SUMMARY")
    print("=" * 72)

    # Phase-level averages
    print("\n--- Phase-Averaged ScrewSet-C Accuracy ---")
    for phase, models in [("CNN", PHASE1_MODELS), ("ViT", PHASE2_MODELS), ("VLM", PHASE3_MODELS)]:
        sub = df_ss[df_ss["model"].isin(models)]
        if sub.empty:
            continue
        print(f"  {phase:4s}:  clean={sub['clean'].mean()*100:.1f}%  "
              f"mean_SS-C={sub['mean_corrupt'].mean()*100:.1f}%  "
              f"gap={((sub['clean']-sub['mean_corrupt']).mean())*100:.1f}pp")

    # Hardest / easiest corruptions
    print("\n--- Corruption Difficulty Ranking (mean across all 24 models) ---")
    corr_means = {}
    for c in SCREWSET_CORRUPTIONS:
        corr_means[CORRUPTION_DISPLAY[c]] = df_ss[c].mean() * 100
    for name, val in sorted(corr_means.items(), key=lambda x: x[1]):
        print(f"  {name:20s}:  {val:.1f}%")

    # Top 5 most/least robust models
    print("\n--- Most Robust Models (ScrewSet-C) ---")
    top5 = df_ss.nlargest(5, "mean_corrupt")
    for _, r in top5.iterrows():
        print(f"  {r['display_name']:20s}:  SS-C={r['mean_corrupt']*100:.1f}%  "
              f"clean={r['clean']*100:.1f}%")

    print("\n--- Least Robust Models (ScrewSet-C) ---")
    bot5 = df_ss.nsmallest(5, "mean_corrupt")
    for _, r in bot5.iterrows():
        print(f"  {r['display_name']:20s}:  SS-C={r['mean_corrupt']*100:.1f}%  "
              f"clean={r['clean']*100:.1f}%")

    # mCE summary
    if not df_mce.empty:
        print("\n--- mCE Rankings (lower = more robust, ref = ResNet-18) ---")
        for _, r in df_mce.iterrows():
            marker = " <-- reference" if r["mCE"] == 1.0 else ""
            print(f"  {r['display_name']:20s}:  mCE = {r['mCE']:.3f}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading all JSON results...")
    records = load_all_jsons()
    print(f"  Loaded {len(records)} result files.\n")

    print("Building data tables...")
    df_ss = build_screwset_table(records)
    df_c10 = build_cifar10_corruption_table(records)
    df_aug = build_augmix_table(records)
    df_cross = build_cross_dataset_table(records)

    print(f"  ScrewSet table:  {len(df_ss)} models x {len(SCREWSET_CORRUPTIONS)} corruptions")
    print(f"  CIFAR-10 table:  {len(df_c10)} models x {len(CIFAR10_CORRUPTIONS_15)} corruptions")
    print(f"  AugMix table:    {len(df_aug)} models")
    print(f"  Cross-dataset:   {len(df_cross)} records\n")

    # Compute mCE
    df_mce = compute_mce(df_ss, reference_model="resnet18")

    # Generate all figures
    print("Generating figures...")
    fig_screwset_heatmap(df_ss)
    fig_cifar10_heatmap(df_c10)
    fig_radar_screwset(df_ss)
    fig_radar_cnn_vs_vit(df_ss)
    fig_robustness_gap(df_ss)
    fig_cross_dataset_correlation(df_cross)
    fig_corruption_correlation(df_ss)
    fig_model_correlation(df_ss)
    fig_augmix_delta(df_ss, df_aug)
    fig_mce_bar(df_mce)
    fig_relative_drop(df_ss)

    # Export CSVs
    print("\nExporting CSVs...")
    export_csvs(df_ss, df_c10, df_mce, df_aug, df_cross)

    # Print summary
    print_summary(df_ss, df_mce)

    print(f"\nAll outputs saved to: {FIGURES}/")
    print("Done.")


if __name__ == "__main__":
    main()
