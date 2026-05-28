"""
eval/instability.py

Core instability scoring logic.

Instability score for a single example evaluated on K prompt variants:

    instability = 1 - max(count_positive / K, count_negative / K)

  - 0.0  → all K variants agreed (fully stable)
  - 0.5  → maximum disagreement (half positive, half negative)

Also provides helpers for:
  - Aggregating instability per prompt dimension
  - Normalising raw model outputs to a canonical {0, 1, None} label
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd


# Map surface forms produced by different label-vocab variants to a canonical
# binary label: 1 = positive, 0 = negative, None = unparseable.
_POSITIVE_SURFACE = {
    "positive", "pos", "good", "favorable", "favourable", "1",
}
_NEGATIVE_SURFACE = {
    "negative", "neg", "bad", "unfavorable", "unfavourable", "0",
}


def parse_label(response: str) -> int | None:
    """
    Extract a canonical label from a raw model response string.

    Returns 1 (positive), 0 (negative), or None if unparseable.
    The function is intentionally tolerant: it strips punctuation and looks
    for the first matching token so partial responses still parse correctly.
    """
    cleaned = response.strip().lower()
    # Remove surrounding punctuation / quotation marks
    cleaned = re.sub(r"[\"'.,!?;:]", "", cleaned)
    # Check the first token, then the whole string
    first_token = cleaned.split()[0] if cleaned.split() else ""
    for candidate in (first_token, cleaned):
        if candidate in _POSITIVE_SURFACE:
            return 1
        if candidate in _NEGATIVE_SURFACE:
            return 0
    return None


def instability_score(predictions: list[int | None]) -> float:
    """
    Compute the instability score for one example given K predictions.

    Predictions containing None (failed parses) are excluded from the
    denominator so that parse failures do not artificially inflate instability.

    Parameters
    ----------
    predictions:
        List of K labels, each 0, 1, or None.

    Returns
    -------
    float in [0.0, 0.5], or np.nan if fewer than 2 valid predictions exist.
    """
    valid = [p for p in predictions if p is not None]
    k = len(valid)
    if k < 2:
        return np.nan
    count_pos = sum(valid)
    count_neg = k - count_pos
    return 1.0 - max(count_pos / k, count_neg / k)


def dimension_instability(
    df_preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean instability broken down by prompt dimension.

    Parameters
    ----------
    df_preds:
        DataFrame with columns: example_id, dimension, variant_name,
        parsed_label.  One row per (example, variant) pair.

    Returns
    -------
    DataFrame with columns: dimension, variant_name, mean_instability.
    """
    records = []
    for (dim, variant), group in df_preds.groupby(["dimension", "variant_name"]):
        # For each example within this dimension×variant, we need to compare
        # its prediction against the predictions from all other variants in
        # the same dimension to compute per-example instability contribution.
        # Here we compute a simpler aggregate: flip rate (fraction of examples
        # where this variant disagrees with the majority across all variants).
        pass  # filled below

    # Compute per-example instability across all variants in each dimension
    result_rows = []
    for dim, dim_group in df_preds.groupby("dimension"):
        for example_id, ex_group in dim_group.groupby("example_id"):
            preds = ex_group["parsed_label"].tolist()
            score = instability_score(preds)
            result_rows.append({
                "dimension": dim,
                "example_id": example_id,
                "instability": score,
            })

    result_df = pd.DataFrame(result_rows)
    summary = (
        result_df.groupby("dimension")["instability"]
        .agg(mean_instability="mean", std_instability="std", n_examples="count")
        .reset_index()
    )
    return summary


def per_example_instability(df_preds: pd.DataFrame) -> pd.Series:
    """
    Compute overall instability for each example, aggregated across all variants.

    Parameters
    ----------
    df_preds:
        DataFrame with columns: example_id, parsed_label (plus any others).

    Returns
    -------
    pd.Series indexed by example_id.
    """
    return (
        df_preds.groupby("example_id")["parsed_label"]
        .apply(lambda s: instability_score(s.tolist()))
    )


if __name__ == "__main__":
    # Smoke-test with synthetic data
    test_cases = [
        ([1, 1, 1, 1], 0.0),    # all agree positive
        ([0, 0, 0, 0], 0.0),    # all agree negative
        ([1, 0, 1, 0], 0.5),    # perfect split
        ([1, 1, 1, 0], 0.25),   # 3/4 positive
        ([1, None, 1, None], 0.0),  # valid predictions all agree
        ([None, None], float("nan")),
    ]
    for preds, expected in test_cases:
        score = instability_score(preds)
        status = "OK" if (np.isnan(score) and np.isnan(expected)) or np.isclose(score, expected) else "FAIL"
        print(f"preds={preds!r:30s}  score={score:.4f}  expected={expected}  [{status}]")

    print("\nparse_label tests:")
    parse_tests = [
        ("positive", 1), ("Negative.", 0), ("GOOD", 1), ("unfavorable", 0),
        ("1", 1), ("0", 0), ("  bad  ", 0), ("garbage", None),
    ]
    for raw, expected in parse_tests:
        result = parse_label(raw)
        status = "OK" if result == expected else "FAIL"
        print(f"  {raw!r:25s} -> {result}  expected {expected}  [{status}]")
