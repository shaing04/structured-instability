# Prompt Sensitivity Evaluation for LLM Classification Instability

This repository contains the evaluation pipeline for a _draft_ paper studying how
prompt phrasing variation affects classification consistency in large language models.
We use SST-2 sentiment classification as a benchmark task and measure _instability_
— the rate at which different prompt phrasings produce different labels for the same
input — and how that instability correlates with human annotator disagreement.

## Research Questions

1. How much does prompt phrasing variation drive label instability in LLM classifiers?
2. Do examples with high human annotator disagreement (low IAA) exhibit more prompt-induced instability?
3. Is there a measurable accuracy–instability tradeoff across prompt dimensions?

## Repository Structure

```
prompt-instability-eval/
├── data/
│   ├── sst2_sample.py          # Downloads and saves 100-example SST-2 sample
│   └── sst2_sample.csv         # Generated dataset (created by running the script)
├── prompts/
│   └── prompt_matrix.py        # All prompt variants across 4 dimensions
├── eval/
│   ├── instability.py          # Instability scoring functions
│   ├── run_eval.py             # Main evaluation loop (calls Anthropic API)
│   ├── correlation.py          # Spearman correlation: instability vs. IAA
│   └── tradeoff.py             # Accuracy vs. instability tradeoff plots
├── results/                    # Output CSVs and figures (created at runtime)
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
cd prompt-instability-eval
pip install -r requirements.txt
```

### 2. Set your API key

You will need an API key to run the evaluation. Below is an example how to set your API key using Anthropic.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"   # Linux/macOS
$env:ANTHROPIC_API_KEY="your-api-key-here"     # Windows PowerShell
```

### 3. Create the dataset

```bash
cd data
python sst2_sample.py
```

This downloads 100 examples from the SST-2 validation split via Hugging Face
`datasets` and writes `data/sst2_sample.csv` with columns:

- `text` — the review sentence
- `label` — ground-truth sentiment (0 = negative, 1 = positive)
- `iaa_score` — inter-annotator agreement proxy (float 0–1)

## Running the Evaluation

### Step 1 — Run all prompt variants through the API

```bash
cd eval
python run_eval.py
```

This iterates over all 15 prompt variants (4 verbosity × 3 label-position × 4
label-vocab × 4 instruction-verb = 192 total, de-duplicated to unique
representative variants per dimension) for each of the 100 examples, calls
`claude-haiku-4-5-20251001` at `temperature=0`, and writes
`results/raw_predictions.csv`.

**Note:** This makes ~1,500 API calls. Estimated cost ≈ $0.05–$0.20 at
Haiku pricing. A `--dry-run` flag is available to preview without API calls.

### Step 2 — Correlation analysis

```bash
python correlation.py
```

Computes per-example instability scores, runs Spearman correlation against
`iaa_score`, prints the result, and saves `results/instability_vs_iaa.png`.

### Step 3 — Accuracy–instability tradeoff

```bash
python tradeoff.py
```

Computes per-variant accuracy and mean instability, then saves
`results/accuracy_vs_instability.png`.

## Prompt Dimensions

| Dimension        | Variants | Description                                             |
| ---------------- | -------- | ------------------------------------------------------- |
| Verbosity        | 4        | Minimal → verbose instruction framing                   |
| Label position   | 3        | Where label options appear in the prompt                |
| Label vocabulary | 4        | positive/negative, good/bad, favorable/unfavorable, 1/0 |
| Instruction verb | 4        | classify, identify, determine, label                    |

## Instability Score

For a single example evaluated on K prompt variants:

```
instability = 1 - max(count_positive / K, count_negative / K)
```

A score of 0 means all K variants agreed; 0.5 means maximum disagreement.

## Citation

If you use this code, please cite:

```bibtex
@misc{prompt-instability-eval,
  title  = {Structured Instability: Prompt Sensitivity in LLM Classification},
  year   = {2026},
  url    = {https://github.com/shaing04/structured-instability}
}
```
