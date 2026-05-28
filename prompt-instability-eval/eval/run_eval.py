"""
eval/run_eval.py

Main evaluation loop.

For each example in data/sst2_sample.csv, runs every prompt variant through
the Anthropic API (claude-haiku-4-5-20251001, temperature=0) and saves the
raw responses and parsed labels to results/raw_predictions.csv.

Usage
-----
    python eval/run_eval.py                  # full run
    python eval/run_eval.py --dry-run        # print prompts, no API calls
    python eval/run_eval.py --limit 5        # only process first N examples

Environment
-----------
    ANTHROPIC_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.prompt_matrix import ALL_VARIANTS
from eval.instability import parse_label

DATA_PATH = PROJECT_ROOT / "data" / "sst2_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "raw_predictions.csv"

MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0
MAX_TOKENS = 16  # We only need a single word response

# Seconds to wait between API calls to stay within rate limits
RATE_LIMIT_DELAY = 0.25


def call_anthropic(client, prompt: str, dry_run: bool = False) -> str:
    """
    Send a single prompt to the Anthropic API and return the text response.
    In dry-run mode, returns a placeholder without making an API call.
    """
    if dry_run:
        return "[DRY RUN]"

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def run_evaluation(dry_run: bool = False, limit: int | None = None) -> pd.DataFrame:
    """
    Core evaluation loop.

    Returns a DataFrame with one row per (example_id, dimension, variant_name).
    Columns: example_id, text, label, iaa_score, dimension, variant_name,
             prompt, raw_response, parsed_label.
    """
    import anthropic  # imported here so --dry-run doesn't require the key

    client = anthropic.Anthropic() if not dry_run else None

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    if limit is not None:
        df = df.head(limit)
        print(f"Limiting to first {limit} examples.")

    n_examples = len(df)
    n_variants = len(ALL_VARIANTS)
    total_calls = n_examples * n_variants
    print(f"Examples: {n_examples} | Variants: {n_variants} | Total API calls: {total_calls}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    completed = 0

    for idx, row in df.iterrows():
        example_id = int(idx)
        text = row["text"]
        label = int(row["label"])
        iaa_score = float(row["iaa_score"])

        for dim, variant_name, prompt_fn in ALL_VARIANTS:
            prompt = prompt_fn(text)

            if dry_run:
                print(f"\n[example {example_id}] [{dim}] [{variant_name}]")
                print(prompt[:200] + ("..." if len(prompt) > 200 else ""))
                raw_response = "[DRY RUN]"
                parsed = None
            else:
                try:
                    raw_response = call_anthropic(client, prompt, dry_run=False)
                    parsed = parse_label(raw_response)
                    time.sleep(RATE_LIMIT_DELAY)
                except Exception as e:
                    print(f"  WARNING: API error for example {example_id}, variant {variant_name}: {e}")
                    raw_response = f"ERROR: {e}"
                    parsed = None

            records.append({
                "example_id":    example_id,
                "text":          text,
                "label":         label,
                "iaa_score":     iaa_score,
                "dimension":     dim,
                "variant_name":  variant_name,
                "prompt":        prompt,
                "raw_response":  raw_response,
                "parsed_label":  parsed,
            })

            completed += 1
            if completed % 50 == 0:
                pct = 100 * completed / total_calls
                print(f"  Progress: {completed}/{total_calls} ({pct:.1f}%)")

    result_df = pd.DataFrame(records)

    if not dry_run:
        result_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(result_df)} rows to {OUTPUT_PATH}")
        _print_summary(result_df)
    else:
        print(f"\n[DRY RUN] Would save {len(result_df)} rows to {OUTPUT_PATH}")

    return result_df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of parse success rates after evaluation."""
    total = len(df)
    parsed_ok = df["parsed_label"].notna().sum()
    parse_rate = 100 * parsed_ok / total
    print(f"\nParse success: {parsed_ok}/{total} ({parse_rate:.1f}%)")
    failed = df[df["parsed_label"].isna()][["variant_name", "raw_response"]].head(10)
    if not failed.empty:
        print("Sample unparsed responses:")
        print(failed.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt variant evaluation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without making API calls.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N examples.")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    run_evaluation(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
