"""
eval/correlation.py

Loads results/raw_predictions.csv, computes per-example instability scores,
runs Spearman rank correlation against the iaa_score column, prints the result,
and saves a scatterplot to results/instability_vs_iaa.png.

Usage
-----
    python eval/correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.instability import per_example_instability

PREDICTIONS_PATH = PROJECT_ROOT / "results" / "raw_predictions.csv"
OUTPUT_FIG = PROJECT_ROOT / "results" / "instability_vs_iaa.png"


def load_and_score(predictions_path: Path) -> pd.DataFrame:
    """
    Load raw predictions, compute per-example instability, and return a
    summary DataFrame with one row per example.

    Columns: example_id, iaa_score, instability.
    """
    df = pd.read_csv(predictions_path)

    # Compute overall instability (across all 15 variants) per example
    instability = per_example_instability(df).rename("instability")

    # One iaa_score per example — grab from any row
    iaa = df.groupby("example_id")["iaa_score"].first()
    label = df.groupby("example_id")["label"].first()

    summary = pd.concat([iaa, instability, label], axis=1).reset_index()
    return summary


def compute_correlation(summary: pd.DataFrame) -> tuple[float, float]:
    """
    Compute Spearman correlation between instability and iaa_score.

    Returns (rho, p_value).
    """
    valid = summary.dropna(subset=["instability", "iaa_score"])
    rho, p_value = spearmanr(valid["iaa_score"], valid["instability"])
    return float(rho), float(p_value)


def plot_scatter(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Produce and save a scatter plot of instability vs. IAA score, with a
    Lowess regression line and per-label colouring.
    """
    valid = summary.dropna(subset=["instability", "iaa_score"])
    rho, p_value = compute_correlation(valid)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    palette = {0: "#e05c5c", 1: "#5c8ee0"}
    label_names = {0: "Negative", 1: "Positive"}
    for lbl, grp in valid.groupby("label"):
        ax.scatter(
            grp["iaa_score"],
            grp["instability"],
            label=label_names[lbl],
            color=palette[lbl],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            s=55,
        )

    # Lowess trend line
    from statsmodels.nonparametric.smoothers_lowess import lowess  # optional dep
    try:
        sorted_iaa = valid.sort_values("iaa_score")
        smoothed = lowess(sorted_iaa["instability"], sorted_iaa["iaa_score"], frac=0.4)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=1.8,
                linestyle="--", label="LOWESS trend")
    except ImportError:
        # statsmodels not installed — skip trend line
        pass

    significance = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.set_xlabel("IAA Score (annotator agreement proxy)", fontsize=12)
    ax.set_ylabel("Prompt Instability Score", fontsize=12)
    ax.set_title(
        f"Prompt Instability vs. Inter-Annotator Agreement\n"
        f"Spearman ρ = {rho:.3f}  ({significance})",
        fontsize=13,
    )
    ax.legend(title="Ground-truth label", framealpha=0.9)
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(-0.05, 0.55)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {output_path}")


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        print("Run eval/run_eval.py first to generate predictions.")
        sys.exit(1)

    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    summary = load_and_score(PREDICTIONS_PATH)
    print(f"Loaded {len(summary)} examples.\n")

    rho, p_value = compute_correlation(summary)
    significance = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
    print("=" * 50)
    print(f"  Spearman ρ (instability ~ IAA) = {rho:+.4f}")
    print(f"  {significance}")
    print("=" * 50)

    if rho < 0:
        print("\nInterpretation: Higher annotator agreement is associated with "
              "lower prompt instability — examples humans find unambiguous "
              "are also more consistently classified across prompt variants.")
    else:
        print("\nInterpretation: Higher annotator agreement is not associated "
              "with lower prompt instability in this sample.")

    print()

    # Per-dimension correlation summary
    df_full = pd.read_csv(PREDICTIONS_PATH)
    print("Per-dimension instability means:")
    for dim, grp in df_full.groupby("dimension"):
        from eval.instability import instability_score
        dim_instability = (
            grp.groupby("example_id")["parsed_label"]
            .apply(lambda s: instability_score(s.tolist()))
        )
        print(f"  {dim:20s}  mean={dim_instability.mean():.4f}  "
              f"std={dim_instability.std():.4f}")

    print()
    plot_scatter(summary, OUTPUT_FIG)
    print("\nCorrelation analysis complete.")


if __name__ == "__main__":
    main()
