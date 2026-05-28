"""
eval/tradeoff.py

For each prompt variant, computes:
  - Accuracy against ground-truth labels
  - Mean instability (how often this variant disagrees with other variants
    in its dimension for the same example)

Then saves a tradeoff scatterplot to results/accuracy_vs_instability.png.

The tradeoff analysis answers: "Do more stable prompts tend to be more accurate,
or is there a cost to stability?"

Usage
-----
    python eval/tradeoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.instability import instability_score

PREDICTIONS_PATH = PROJECT_ROOT / "results" / "raw_predictions.csv"
OUTPUT_FIG = PROJECT_ROOT / "results" / "accuracy_vs_instability.png"

# Color palette, one per dimension
DIMENSION_COLORS = {
    "verbosity":        "#4C72B0",
    "label_position":   "#DD8452",
    "label_vocab":      "#55A868",
    "instruction_verb": "#C44E52",
}


def compute_per_variant_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each (dimension, variant_name) pair.

    Accuracy = fraction of examples where parsed_label == label.
    Rows where parsed_label is None are treated as incorrect.
    """
    rows = []
    for (dim, variant), group in df.groupby(["dimension", "variant_name"]):
        valid = group.dropna(subset=["parsed_label"])
        if len(valid) == 0:
            accuracy = np.nan
        else:
            accuracy = (valid["parsed_label"] == valid["label"]).mean()
        n_total = len(group)
        n_valid = len(valid)
        rows.append({
            "dimension":    dim,
            "variant_name": variant,
            "accuracy":     accuracy,
            "n_total":      n_total,
            "n_valid":      n_valid,
            "parse_rate":   n_valid / n_total if n_total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def compute_per_variant_instability(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each variant, compute mean instability across all examples.

    Instability for a single example within a dimension = instability_score()
    applied to all variant predictions within that dimension for that example.

    We then compute the *contribution* of each variant by looking at how
    its prediction relates to the overall distribution within its dimension.
    Specifically, for each (example, dimension) we compute the dimension-level
    instability, then attribute it equally to all variants in that dimension.
    """
    rows = []
    for (dim, variant), group in df.groupby(["dimension", "variant_name"]):
        # Per-example instability across this entire dimension
        dim_df = df[df["dimension"] == dim]
        example_instabilities = (
            dim_df.groupby("example_id")["parsed_label"]
            .apply(lambda s: instability_score(s.tolist()))
        )
        mean_dim_instability = example_instabilities.mean()

        # Flip rate for this specific variant vs. majority vote in dimension
        flip_rates = []
        for example_id, ex_group in dim_df.groupby("example_id"):
            all_preds = [p for p in ex_group["parsed_label"].tolist() if p is not None]
            if not all_preds:
                continue
            majority = 1 if sum(all_preds) > len(all_preds) / 2 else 0
            variant_pred_rows = group[group["example_id"] == example_id]["parsed_label"]
            if len(variant_pred_rows) == 0 or variant_pred_rows.isna().all():
                continue
            variant_pred = variant_pred_rows.iloc[0]
            if variant_pred is not None:
                flip_rates.append(int(variant_pred != majority))

        variant_flip_rate = np.mean(flip_rates) if flip_rates else np.nan

        rows.append({
            "dimension":           dim,
            "variant_name":        variant,
            "mean_dim_instability": mean_dim_instability,
            "variant_flip_rate":   variant_flip_rate,
        })

    return pd.DataFrame(rows)


def plot_tradeoff(
    accuracy_df: pd.DataFrame,
    instability_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Produce and save the accuracy–instability tradeoff scatter plot.
    """
    merged = accuracy_df.merge(instability_df, on=["dimension", "variant_name"])
    merged = merged.dropna(subset=["accuracy", "mean_dim_instability"])

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in merged.iterrows():
        color = DIMENSION_COLORS.get(row["dimension"], "gray")
        ax.scatter(
            row["mean_dim_instability"],
            row["accuracy"],
            color=color,
            s=90,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            row["variant_name"],
            xy=(row["mean_dim_instability"], row["accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )

    # Reference lines
    ax.axhline(y=merged["accuracy"].mean(), color="gray", linewidth=0.8,
               linestyle=":", alpha=0.7, label=f"Mean accuracy ({merged['accuracy'].mean():.2f})")

    # Legend for dimensions
    patches = [
        mpatches.Patch(color=color, label=dim.replace("_", " ").title())
        for dim, color in DIMENSION_COLORS.items()
    ]
    ax.legend(handles=patches, title="Prompt dimension",
              loc="lower left", framealpha=0.9, fontsize=9)

    ax.set_xlabel("Mean Instability (within dimension)", fontsize=12)
    ax.set_ylabel("Accuracy (vs. ground truth)", fontsize=12)
    ax.set_title("Accuracy–Instability Tradeoff by Prompt Variant", fontsize=13)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved tradeoff plot to {output_path}")


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        print("Run eval/run_eval.py first to generate predictions.")
        sys.exit(1)

    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded {len(df)} rows ({df['example_id'].nunique()} examples, "
          f"{df['variant_name'].nunique()} variants).\n")

    print("Computing per-variant accuracy...")
    accuracy_df = compute_per_variant_accuracy(df)

    print("Computing per-variant instability...")
    instability_df = compute_per_variant_instability(df)

    # Print combined table
    merged = accuracy_df.merge(instability_df, on=["dimension", "variant_name"])
    merged = merged.sort_values(["dimension", "accuracy"], ascending=[True, False])
    print("\nPer-variant accuracy and instability:")
    print(
        merged[["dimension", "variant_name", "accuracy", "mean_dim_instability",
                "variant_flip_rate", "parse_rate"]]
        .to_string(index=False, float_format="{:.4f}".format)
    )

    # Per-dimension summaries
    print("\nPer-dimension summary:")
    for dim, grp in merged.groupby("dimension"):
        print(f"  {dim:20s}  "
              f"acc={grp['accuracy'].mean():.4f}±{grp['accuracy'].std():.4f}  "
              f"instability={grp['mean_dim_instability'].mean():.4f}±"
              f"{grp['mean_dim_instability'].std():.4f}")

    print()
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    plot_tradeoff(accuracy_df, instability_df, OUTPUT_FIG)
    print("\nTradeoff analysis complete.")


if __name__ == "__main__":
    main()
