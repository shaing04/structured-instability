"""
data/sst2_sample.py

Downloads 100 examples from the SST-2 validation split via Hugging Face `datasets`
and writes sst2_sample.csv with columns:
  - text        : the review sentence
  - label       : ground-truth sentiment (0 = negative, 1 = positive)
  - iaa_score   : inter-annotator agreement proxy (float 0–1)

Real IAA scores are not publicly distributed for SST-2, so we simulate them:
  - Start from a base confidence derived from the Naive-Bayes word polarity
    of the sentence (positive/negative word ratio).
  - Add Gaussian noise so the distribution is continuous and plausible.
  - Clip to [0.5, 1.0] — agreement is always at least 50% by definition.

Run:
    python data/sst2_sample.py
"""

import os
import numpy as np
import pandas as pd

SEED = 42
SAMPLE_SIZE = 100
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sst2_sample.csv")

# Simple positive/negative word lexicons for the IAA proxy signal
POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "brilliant", "superb", "love", "best", "beautiful", "perfect",
    "enjoyable", "delightful", "charming", "fun", "engaging", "impressive",
    "powerful", "moving", "touching", "compelling", "funny", "sweet",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "dull", "boring", "worst",
    "disappointing", "poor", "weak", "slow", "stupid", "ugly", "painful",
    "tedious", "flat", "predictable", "forgettable", "mess", "waste",
    "annoying", "laughable", "incoherent", "pretentious",
}


def polarity_confidence(text: str) -> float:
    """
    Returns a crude [0,1] confidence that the text has a clear sentiment.
    High value = strong lexical signal (easy example, high IAA proxy).
    Low value  = weak or mixed signal (ambiguous example, low IAA proxy).
    """
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t.strip(".,!?") in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t.strip(".,!?") in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.5  # no signal → maximum ambiguity
    # Agreement proxy: how lopsided the pos/neg balance is
    dominant = max(pos, neg)
    return dominant / total


def simulate_iaa_scores(texts: list[str], rng: np.random.Generator) -> np.ndarray:
    """
    Simulate inter-annotator agreement scores for a list of texts.

    The score represents the fraction of annotators who chose the majority
    label, i.e. 1.0 = unanimous, 0.5 = maximum disagreement.
    """
    base_confidence = np.array([polarity_confidence(t) for t in texts])
    noise = rng.normal(loc=0.0, scale=0.08, size=len(texts))
    iaa = np.clip(base_confidence + noise, 0.5, 1.0)
    return iaa.round(4)


def main() -> None:
    from datasets import load_dataset  # imported here so the rest of the module
                                       # is importable even without `datasets`

    print("Loading SST-2 validation split from Hugging Face...")
    dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=SAMPLE_SIZE, replace=False)
    sample = dataset.select(indices.tolist())

    texts = sample["sentence"]
    labels = sample["label"]

    iaa_scores = simulate_iaa_scores(texts, rng)

    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "iaa_score": iaa_scores,
    })

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} examples to {OUTPUT_PATH}")
    print(df.head())
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nIAA score stats:\n{df['iaa_score'].describe().to_string()}")


if __name__ == "__main__":
    main()
