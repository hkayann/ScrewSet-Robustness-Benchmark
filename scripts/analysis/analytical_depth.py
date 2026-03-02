#!/usr/bin/env python3
"""
Analytical depth figures for the ScrewSet paper.

A. Parameter count vs corruption robustness (regression)
B. Clean accuracy vs corruption accuracy scatter (predictivity analysis)
C. Corruption taxonomy analysis (spatial / appearance / semantic grouping)

Outputs saved to results/figures/
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

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
# Model metadata
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "resnet18": "ResNet-18",
    "efficientnetv2_rw_s": "EffNetV2-S",
    "mobilenet_v3_large": "MobNetV3-L",
    "mobilenetv4_conv_small": "MobNetV4-S",
    "shufflenet_v2_x1_0": "ShuffleV2",
    "squeezenet1_1": "SqueezeNet",
    "ghostnetv2_100": "GhostV2",
    "convnextv2_atto": "CNXv2-A",
    "vit_tiny_patch16_224": "ViT-Ti",
    "vit_small_patch16_224": "ViT-S",
    "deit_tiny_patch16_224": "DeiT-Ti",
    "deit_small_patch16_224": "DeiT-S",
    "swin_tiny_patch4_window7_224": "Swin-Ti",
    "convnext_tiny": "CNX-Ti",
    "efficientformer_l1": "EffFormer",
    "mobilevit_s": "MobViT-S",
    "clip_vit_b32": "CLIP-B/32",
    "clip_vit_b16": "CLIP-B/16",
    "clip_vit_l14": "CLIP-L/14",
    "openclip_vit_b16": "OpenCLIP",
    "siglip_vit_b16": "SigLIP",
    "eva02_clip_vit_b16": "EVA02",
    "blip2": "BLIP-2",
    "llava": "LLaVA",
}

# Parameter counts (computed from timm/torchvision, num_classes=40)
PARAM_COUNTS = {
    "resnet18": 11_197_032,
    "efficientnetv2_rw_s": 22_220_016,
    "mobilenet_v3_large": 4_253_272,
    "mobilenetv4_conv_small": 2_544_264,
    "shufflenet_v2_x1_0": 1_294_604,
    "squeezenet1_1": 743_016,
    "ghostnetv2_100": 4_927_148,
    "convnextv2_atto": 3_400_240,
    "vit_tiny_patch16_224": 5_532_136,
    "vit_small_patch16_224": 21_681_064,
    "deit_tiny_patch16_224": 5_532_136,
    "deit_small_patch16_224": 21_681_064,
    "swin_tiny_patch4_window7_224": 27_550_114,
    "convnext_tiny": 27_850_888,
    "efficientformer_l1": 11_427_848,
    "mobilevit_s": 4_963_272,
    # VLMs -- total params (published values)
    "clip_vit_b32": 151_000_000,
    "clip_vit_b16": 150_000_000,
    "clip_vit_l14": 428_000_000,
    "openclip_vit_b16": 150_000_000,
    "siglip_vit_b16": 93_000_000,
    "eva02_clip_vit_b16": 87_000_000,
    "blip2": 3_800_000_000,
    "llava": 7_000_000_000,
}

PHASE_MAP = {
    "resnet18": "CNN", "efficientnetv2_rw_s": "CNN", "mobilenet_v3_large": "CNN",
    "mobilenetv4_conv_small": "CNN", "shufflenet_v2_x1_0": "CNN",
    "squeezenet1_1": "CNN", "ghostnetv2_100": "CNN", "convnextv2_atto": "CNN",
    "vit_tiny_patch16_224": "ViT", "vit_small_patch16_224": "ViT",
    "deit_tiny_patch16_224": "ViT", "deit_small_patch16_224": "ViT",
    "swin_tiny_patch4_window7_224": "ViT", "convnext_tiny": "ViT",
    "efficientformer_l1": "ViT", "mobilevit_s": "ViT",
    "clip_vit_b32": "VLM", "clip_vit_b16": "VLM", "clip_vit_l14": "VLM",
    "openclip_vit_b16": "VLM", "siglip_vit_b16": "VLM",
    "eva02_clip_vit_b16": "VLM", "blip2": "VLM", "llava": "VLM",
}

PHASE_COLORS = {"CNN": "#1f77b4", "ViT": "#ff7f0e", "VLM": "#2ca02c"}
PHASE_MARKERS = {"CNN": "o", "ViT": "s", "VLM": "^"}

SCREWSET_CORRUPTIONS = [
    "screwset_multi_object", "screwset_occlusion_bottom_right",
    "screwset_occlusion_top_left", "screwset_reflection",
    "screwset_scrap_paper", "screwset_shadow",
]

CORRUPTION_DISPLAY = {
    "screwset_multi_object": "Multi-object",
    "screwset_occlusion_bottom_right": "Occlusion BR",
    "screwset_occlusion_top_left": "Occlusion TL",
    "screwset_reflection": "Reflection",
    "screwset_scrap_paper": "Scrap paper",
    "screwset_shadow": "Shadow",
}

# Corruption taxonomy
TAXONOMY = {
    "Spatial\n(Occlusion)": ["screwset_occlusion_bottom_right", "screwset_occlusion_top_left"],
    "Appearance\n(Lighting)": ["screwset_reflection", "screwset_shadow"],
    "Semantic\n(Context)": ["screwset_multi_object", "screwset_scrap_paper"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_screwset_data():
    """Load ScrewSet baseline results for all 24 models."""
    rows = []
    for phase_dir in ["phase1", "phase2", "phase3"]:
        for fp in sorted(glob.glob(str(RESULTS / phase_dir / "*_screwset_baselines.json"))):
            d = json.load(open(fp))
            model = d["model"]
            cr = d.get("corrupt_results", {})
            row = {
                "model": model,
                "display": MODEL_DISPLAY.get(model, model),
                "phase": PHASE_MAP.get(model, "?"),
                "params": PARAM_COUNTS.get(model, np.nan),
                "params_M": PARAM_COUNTS.get(model, 0) / 1e6,
                "clean": d.get("test_acc", 0.0),
                "mean_corrupt": d.get("mean_corrupt_acc", 0.0),
            }
            for c in SCREWSET_CORRUPTIONS:
                row[c] = cr.get(c, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A. Parameter count vs corruption robustness
# ---------------------------------------------------------------------------

def fig_A_param_vs_robustness(df):
    """
    Scatter: log(params) vs SS-C accuracy, with regression per phase
    and overall regression.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- A1: All models, log scale ---
    ax = axes[0]
    for phase in ["CNN", "ViT", "VLM"]:
        sub = df[df["phase"] == phase]
        ax.scatter(
            sub["params_M"], sub["mean_corrupt"] * 100,
            c=PHASE_COLORS[phase], marker=PHASE_MARKERS[phase],
            s=70, label=phase, edgecolors="black", linewidths=0.5, zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["display"], (r["params_M"], r["mean_corrupt"] * 100),
                fontsize=6, textcoords="offset points", xytext=(4, 4),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("ScrewSet-C Mean Accuracy (%)")
    ax.set_title("A1. Parameter Count vs Corruption Robustness")
    ax.legend()

    # Overall regression (log params vs corrupt acc)
    x_log = np.log10(df["params_M"].values)
    y = df["mean_corrupt"].values * 100
    mask = np.isfinite(x_log) & np.isfinite(y)
    r_all, p_all = stats.pearsonr(x_log[mask], y[mask])
    rho_all, p_rho = stats.spearmanr(x_log[mask], y[mask])
    ax.text(0.02, 0.97,
            f"Pearson r = {r_all:.3f} (p={p_all:.2e})\n"
            f"Spearman rho = {rho_all:.3f} (p={p_rho:.2e})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # --- A2: CNN + ViT only (supervised), log scale ---
    ax = axes[1]
    df_sup = df[df["phase"].isin(["CNN", "ViT"])].copy()
    for phase in ["CNN", "ViT"]:
        sub = df_sup[df_sup["phase"] == phase]
        ax.scatter(
            sub["params_M"], sub["mean_corrupt"] * 100,
            c=PHASE_COLORS[phase], marker=PHASE_MARKERS[phase],
            s=70, label=phase, edgecolors="black", linewidths=0.5, zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["display"], (r["params_M"], r["mean_corrupt"] * 100),
                fontsize=6, textcoords="offset points", xytext=(4, 4),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("ScrewSet-C Mean Accuracy (%)")
    ax.set_title("A2. Supervised Models Only")

    # Per-phase regressions
    for phase, color in [("CNN", "#1f77b4"), ("ViT", "#ff7f0e")]:
        sub = df_sup[df_sup["phase"] == phase]
        x_log_p = np.log10(sub["params_M"].values)
        y_p = sub["mean_corrupt"].values * 100
        if len(x_log_p) > 2:
            r_p, p_p = stats.pearsonr(x_log_p, y_p)
            m_fit, b_fit = np.polyfit(x_log_p, y_p, 1)
            xline = np.linspace(x_log_p.min() - 0.1, x_log_p.max() + 0.1, 50)
            ax.plot(10**xline, m_fit * xline + b_fit, "--", color=color, linewidth=1.2,
                    alpha=0.7)

    # Overall supervised regression
    x_log_s = np.log10(df_sup["params_M"].values)
    y_s = df_sup["mean_corrupt"].values * 100
    r_s, p_s = stats.pearsonr(x_log_s, y_s)
    rho_s, p_rho_s = stats.spearmanr(x_log_s, y_s)
    m_fit, b_fit = np.polyfit(x_log_s, y_s, 1)
    xline = np.linspace(x_log_s.min() - 0.1, x_log_s.max() + 0.1, 50)
    ax.plot(10**xline, m_fit * xline + b_fit, "-", color="gray", linewidth=1.5,
            alpha=0.5, label="Overall fit")
    ax.text(0.02, 0.97,
            f"Pearson r = {r_s:.3f} (p={p_s:.2e})\n"
            f"Spearman rho = {rho_s:.3f} (p={p_rho_s:.2e})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    ax.legend()

    plt.tight_layout()
    out = FIGURES / "param_vs_robustness.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  [A] Saved: {out.name}")

    # Print key stats
    print(f"      All models:       Pearson r={r_all:.3f} p={p_all:.2e}, "
          f"Spearman rho={rho_all:.3f} p={p_rho:.2e}")
    print(f"      Supervised only:  Pearson r={r_s:.3f} p={p_s:.2e}, "
          f"Spearman rho={rho_s:.3f} p={p_rho_s:.2e}")


# ---------------------------------------------------------------------------
# B. Clean accuracy vs corruption accuracy scatter
# ---------------------------------------------------------------------------

def fig_B_clean_vs_corrupt(df):
    """
    Scatter: clean accuracy vs SS-C accuracy per model.
    Shows whether clean performance predicts robustness.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- B1: All 24 models ---
    ax = axes[0]
    for phase in ["CNN", "ViT", "VLM"]:
        sub = df[df["phase"] == phase]
        ax.scatter(
            sub["clean"] * 100, sub["mean_corrupt"] * 100,
            c=PHASE_COLORS[phase], marker=PHASE_MARKERS[phase],
            s=70, label=phase, edgecolors="black", linewidths=0.5, zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["display"], (r["clean"] * 100, r["mean_corrupt"] * 100),
                fontsize=6, textcoords="offset points", xytext=(4, 4),
            )

    x_all = df["clean"].values * 100
    y_all = df["mean_corrupt"].values * 100
    r_all, p_all = stats.pearsonr(x_all, y_all)

    # Diagonal (perfect robustness)
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3, label="y=x (no degradation)")

    ax.set_xlabel("Clean Test Accuracy (%)")
    ax.set_ylabel("ScrewSet-C Mean Accuracy (%)")
    ax.set_title("B1. Clean vs Corrupt: All Models")
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 55)
    ax.text(0.02, 0.97,
            f"Pearson r = {r_all:.3f} (p={p_all:.2e})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    ax.legend(fontsize=8)

    # --- B2: Supervised only (CNN + ViT) ---
    ax = axes[1]
    df_sup = df[df["phase"].isin(["CNN", "ViT"])]
    for phase in ["CNN", "ViT"]:
        sub = df_sup[df_sup["phase"] == phase]
        ax.scatter(
            sub["clean"] * 100, sub["mean_corrupt"] * 100,
            c=PHASE_COLORS[phase], marker=PHASE_MARKERS[phase],
            s=70, label=phase, edgecolors="black", linewidths=0.5, zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["display"], (r["clean"] * 100, r["mean_corrupt"] * 100),
                fontsize=6, textcoords="offset points", xytext=(4, 4),
            )

    x_sup = df_sup["clean"].values * 100
    y_sup = df_sup["mean_corrupt"].values * 100
    r_sup, p_sup = stats.pearsonr(x_sup, y_sup)

    # Regression line
    m_fit, b_fit = np.polyfit(x_sup, y_sup, 1)
    xline = np.linspace(x_sup.min() - 2, x_sup.max() + 2, 100)
    ax.plot(xline, m_fit * xline + b_fit, "--", color="gray", linewidth=1.2)
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3, label="y=x")

    ax.set_xlabel("Clean Test Accuracy (%)")
    ax.set_ylabel("ScrewSet-C Mean Accuracy (%)")
    ax.set_title("B2. Supervised Models: Clean vs Corrupt")
    ax.set_xlim(92, 101)
    ax.set_ylim(0, 52)
    ax.text(0.02, 0.97,
            f"Pearson r = {r_sup:.3f} (p={p_sup:.2e})\n"
            f"Slope = {m_fit:.2f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES / "clean_vs_corrupt.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  [B] Saved: {out.name}")

    # Compute gap stats
    df_sup_gap = df_sup.copy()
    df_sup_gap["gap"] = (df_sup_gap["clean"] - df_sup_gap["mean_corrupt"]) * 100
    print(f"      All models:       Pearson r={r_all:.3f} p={p_all:.2e}")
    print(f"      Supervised only:  Pearson r={r_sup:.3f} p={p_sup:.2e}")
    print(f"      Supervised gap:   min={df_sup_gap['gap'].min():.1f}pp, "
          f"max={df_sup_gap['gap'].max():.1f}pp, "
          f"mean={df_sup_gap['gap'].mean():.1f}pp")


# ---------------------------------------------------------------------------
# C. Corruption taxonomy analysis
# ---------------------------------------------------------------------------

def fig_C_corruption_taxonomy(df):
    """
    Grouped analysis by corruption category:
      - Spatial (occlusion BR, occlusion TL)
      - Appearance (reflection, shadow)
      - Semantic (multi-object, scrap paper)

    Produces:
      C1: Bar chart of mean acc per taxonomy group per phase
      C2: Heatmap of per-model per-taxonomy-group accuracy
      C3: Scatter showing within-category correlation
    """
    # Compute taxonomy group means per model
    for tax_name, tax_corruptions in TAXONOMY.items():
        col_name = f"tax_{tax_name.split(chr(10))[0].lower()}"
        df[col_name] = df[tax_corruptions].mean(axis=1) * 100
    tax_cols = ["tax_spatial", "tax_appearance", "tax_semantic"]
    tax_labels = ["Spatial\n(Occlusion)", "Appearance\n(Lighting)", "Semantic\n(Context)"]

    # ----- C1: Grouped bar chart by phase -----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    phase_means = {}
    for phase in ["CNN", "ViT", "VLM"]:
        sub = df[df["phase"] == phase]
        means = [sub[tc].mean() for tc in tax_cols]
        stds = [sub[tc].std() for tc in tax_cols]
        phase_means[phase] = (means, stds)

    x = np.arange(len(tax_labels))
    w = 0.25
    for i, (phase, (means, stds)) in enumerate(phase_means.items()):
        ax.bar(x + i * w, means, w, yerr=stds, label=phase,
               color=PHASE_COLORS[phase], capsize=3, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + w)
    ax.set_xticklabels(tax_labels, fontsize=9)
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("C1. Corruption Taxonomy: Phase Comparison")
    ax.legend()
    ax.set_ylim(0, max(max(m) for m, _ in phase_means.values()) * 1.3)

    # ----- C2: Heatmap per model per taxonomy group -----
    ax = axes[1]
    # Only supervised models for clearer heatmap
    df_sup = df[df["phase"].isin(["CNN", "ViT"])].copy()
    mat = df_sup[tax_cols].values
    labels_y = df_sup["display"].tolist()

    sns.heatmap(
        mat, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=["Spatial", "Appearance", "Semantic"],
        yticklabels=labels_y,
        vmin=0, vmax=70, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    n_cnn = sum(1 for _, r in df_sup.iterrows() if r["phase"] == "CNN")
    ax.axhline(n_cnn, color="black", linewidth=2)
    ax.set_title("C2. Per-Model Taxonomy Accuracy")

    # ----- C3: Scatter — Spatial vs Semantic robustness -----
    ax = axes[2]
    for phase in ["CNN", "ViT"]:
        sub = df_sup[df_sup["phase"] == phase]
        ax.scatter(
            sub["tax_spatial"], sub["tax_semantic"],
            c=PHASE_COLORS[phase], marker=PHASE_MARKERS[phase],
            s=70, label=phase, edgecolors="black", linewidths=0.5, zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["display"],
                (r["tax_spatial"], r["tax_semantic"]),
                fontsize=6, textcoords="offset points", xytext=(4, 4),
            )

    x_s = df_sup["tax_spatial"].values
    y_s = df_sup["tax_semantic"].values
    r_val, p_val = stats.pearsonr(x_s, y_s)
    m_fit, b_fit = np.polyfit(x_s, y_s, 1)
    xline = np.linspace(x_s.min() - 2, x_s.max() + 2, 50)
    ax.plot(xline, m_fit * xline + b_fit, "--", color="gray", linewidth=1)

    ax.set_xlabel("Spatial (Occlusion) Accuracy (%)")
    ax.set_ylabel("Semantic (Context) Accuracy (%)")
    ax.set_title("C3. Spatial vs Semantic Robustness")
    ax.text(0.02, 0.97,
            f"r = {r_val:.3f} (p={p_val:.2e})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES / "corruption_taxonomy.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  [C1-C3] Saved: {out.name}")

    # ----- C4: Relative vulnerability radar (CNN vs ViT per taxonomy) -----
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    tax_names_short = ["Spatial\n(Occlusion)", "Appearance\n(Lighting)", "Semantic\n(Context)"]
    N = len(tax_names_short)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Compute "relative robustness" = corrupt_acc / clean_acc per taxonomy
    for phase in ["CNN", "ViT"]:
        sub = df[df["phase"] == phase]
        clean_mean = sub["clean"].mean() * 100
        rel_vals = []
        for tc in tax_cols:
            rel_vals.append(sub[tc].mean() / clean_mean * 100)
        rel_vals += rel_vals[:1]
        ax.plot(angles, rel_vals, "-o", color=PHASE_COLORS[phase],
                linewidth=2, markersize=6, label=f"{phase} (rel. retained)")
        ax.fill(angles, rel_vals, alpha=0.1, color=PHASE_COLORS[phase])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tax_names_short, size=9)
    ax.set_ylim(0, 55)
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_yticklabels(["10%", "20%", "30%", "40%", "50%"], size=8)
    ax.set_title("C4. Relative Robustness Retained\nby Corruption Category", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    out = FIGURES / "corruption_taxonomy_radar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  [C4] Saved: {out.name}")

    # Print taxonomy stats
    print("\n      --- Corruption Taxonomy Summary ---")
    for phase in ["CNN", "ViT", "VLM"]:
        sub = df[df["phase"] == phase]
        print(f"      {phase}:")
        for tc, tn in zip(tax_cols, ["Spatial", "Appearance", "Semantic"]):
            print(f"        {tn:12s}: {sub[tc].mean():.1f}% +/- {sub[tc].std():.1f}%")

    # Key insight: which category is hardest?
    print("\n      --- Hardest Category per Phase (supervised) ---")
    for phase in ["CNN", "ViT"]:
        sub = df[df["phase"] == phase]
        vals = {tn: sub[tc].mean() for tc, tn in zip(tax_cols, ["Spatial", "Appearance", "Semantic"])}
        hardest = min(vals, key=vals.get)
        easiest = max(vals, key=vals.get)
        print(f"      {phase}: Hardest = {hardest} ({vals[hardest]:.1f}%), "
              f"Easiest = {easiest} ({vals[easiest]:.1f}%)")


# ---------------------------------------------------------------------------
# D. Supplementary: Robustness gap decomposition
# ---------------------------------------------------------------------------

def fig_D_gap_decomposition(df):
    """
    Waterfall-style chart showing how each corruption contributes to
    the total accuracy drop from clean to corrupt, per phase.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    phases_sup = ["CNN", "ViT"]
    colors_corr = plt.cm.Set2(np.linspace(0, 1, len(SCREWSET_CORRUPTIONS)))

    for idx, phase in enumerate(phases_sup):
        ax = axes[idx]
        sub = df[df["phase"] == phase]
        clean_mean = sub["clean"].mean() * 100

        # Per-corruption mean accuracy
        corr_means = []
        corr_names = []
        for c in SCREWSET_CORRUPTIONS:
            corr_means.append(sub[c].mean() * 100)
            corr_names.append(CORRUPTION_DISPLAY[c])

        # Sort by accuracy (hardest first)
        order = np.argsort(corr_means)
        corr_means = [corr_means[i] for i in order]
        corr_names = [corr_names[i] for i in order]
        colors_sorted = [colors_corr[i] for i in order]

        x = np.arange(len(corr_names))
        bars = ax.bar(x, corr_means, color=colors_sorted, edgecolor="black", linewidth=0.3)

        # Clean reference line
        ax.axhline(clean_mean, color="red", linestyle="--", linewidth=1.2,
                    label=f"Clean: {clean_mean:.1f}%")

        # Annotate drop
        for i, (val, name) in enumerate(zip(corr_means, corr_names)):
            drop = clean_mean - val
            ax.annotate(f"-{drop:.0f}pp",
                        xy=(i, val + 0.5), ha="center", fontsize=7, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(corr_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{phase}: Per-Corruption Accuracy\n(sorted by difficulty)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, clean_mean * 1.15)

    plt.tight_layout()
    out = FIGURES / "gap_decomposition.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  [D] Saved: {out.name}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(df):
    csv_dir = FIGURES / "csv"
    csv_dir.mkdir(exist_ok=True)

    # Param + robustness table
    out_df = df[["display", "phase", "params_M", "clean", "mean_corrupt"]].copy()
    out_df.columns = ["Model", "Phase", "Params (M)", "Clean (%)", "SS-C (%)"]
    out_df["Clean (%)"] = (out_df["Clean (%)"] * 100).round(2)
    out_df["SS-C (%)"] = (out_df["SS-C (%)"] * 100).round(2)
    out_df["Params (M)"] = out_df["Params (M)"].round(2)
    out_df["Gap (pp)"] = (out_df["Clean (%)"] - out_df["SS-C (%)"]).round(2)

    # Add taxonomy scores
    for tax_name, tax_corruptions in TAXONOMY.items():
        col_short = tax_name.split("\n")[0]
        out_df[f"{col_short} (%)"] = (df[tax_corruptions].mean(axis=1) * 100).round(2).values

    out_df.to_csv(csv_dir / "analytical_depth_table.csv", index=False)
    print(f"  CSV saved to {csv_dir}/analytical_depth_table.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading ScrewSet baseline data...")
    df = load_screwset_data()
    print(f"  Loaded {len(df)} models.\n")

    print("Generating Figure A: Parameter count vs robustness...")
    fig_A_param_vs_robustness(df)

    print("\nGenerating Figure B: Clean vs corrupt accuracy...")
    fig_B_clean_vs_corrupt(df)

    print("\nGenerating Figure C: Corruption taxonomy analysis...")
    fig_C_corruption_taxonomy(df)

    print("\nGenerating Figure D: Gap decomposition...")
    fig_D_gap_decomposition(df)

    print("\nExporting CSV...")
    export_csv(df)

    print("\nDone. All outputs in:", FIGURES)


if __name__ == "__main__":
    main()
