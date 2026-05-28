"""
prompts/prompt_matrix.py

Defines all prompt variants across four dimensions as a structured dictionary.
Each variant is a callable that accepts a review text string and returns a
fully formatted prompt string ready to send to the model.

Dimensions
----------
verbosity       : 4 variants  — how much instruction framing surrounds the task
label_position  : 3 variants  — where the label options appear in the prompt
label_vocab     : 4 variants  — what words are used to name the two classes
instruction_verb: 4 variants  — the action verb used in the instruction

The module also exports:
  ALL_VARIANTS   — flat list of (dimension, variant_name, prompt_fn) tuples
  PROMPT_MATRIX  — nested dict  dimension -> variant_name -> prompt_fn
"""

from __future__ import annotations
from typing import Callable

PromptFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# Dimension 1: Verbosity
# ---------------------------------------------------------------------------

def verbosity_minimal(text: str) -> str:
    return (
        f'Sentiment of: "{text}"\n'
        f'Answer: positive or negative.'
    )


def verbosity_brief(text: str) -> str:
    return (
        f'Classify the sentiment of the following movie review.\n\n'
        f'Review: {text}\n\n'
        f'Respond with exactly one word: positive or negative.'
    )


def verbosity_standard(text: str) -> str:
    return (
        f'You are a sentiment analysis assistant.\n'
        f'Your task is to classify the sentiment of movie reviews as either '
        f'positive or negative.\n\n'
        f'Movie review:\n"{text}"\n\n'
        f'Provide only the sentiment label as your answer.'
    )


def verbosity_verbose(text: str) -> str:
    return (
        f'You are an expert sentiment analysis system specializing in movie reviews.\n'
        f'Read the following movie review carefully and determine whether the '
        f'overall sentiment expressed by the reviewer is positive or negative.\n'
        f'Consider word choice, tone, and overall opinion expressed.\n\n'
        f'Movie review to analyze:\n"{text}"\n\n'
        f'Instructions: Respond with a single word only — either "positive" or '
        f'"negative" — representing the overall sentiment of the review.\n'
        f'Your answer:'
    )


VERBOSITY_VARIANTS: dict[str, PromptFn] = {
    "minimal":  verbosity_minimal,
    "brief":    verbosity_brief,
    "standard": verbosity_standard,
    "verbose":  verbosity_verbose,
}


# ---------------------------------------------------------------------------
# Dimension 2: Label position
# ---------------------------------------------------------------------------

def label_position_first(text: str) -> str:
    return (
        f'Labels: positive, negative\n\n'
        f'Classify the sentiment of this movie review:\n'
        f'"{text}"\n\n'
        f'Answer:'
    )


def label_position_last(text: str) -> str:
    return (
        f'Classify the sentiment of this movie review:\n'
        f'"{text}"\n\n'
        f'Choose one label: positive or negative.\n'
        f'Answer:'
    )


def label_position_embedded(text: str) -> str:
    return (
        f'Is the sentiment of the following movie review positive or negative?\n\n'
        f'Review: "{text}"\n\n'
        f'Answer:'
    )


LABEL_POSITION_VARIANTS: dict[str, PromptFn] = {
    "label_first":    label_position_first,
    "label_last":     label_position_last,
    "label_embedded": label_position_embedded,
}


# ---------------------------------------------------------------------------
# Dimension 3: Label vocabulary
# ---------------------------------------------------------------------------

def label_vocab_posneg(text: str) -> str:
    return (
        f'Classify the sentiment of this movie review.\n'
        f'"{text}"\n\n'
        f'Answer with exactly one word: positive or negative.'
    )


def label_vocab_goodbad(text: str) -> str:
    return (
        f'Classify the sentiment of this movie review.\n'
        f'"{text}"\n\n'
        f'Answer with exactly one word: good or bad.'
    )


def label_vocab_favorable(text: str) -> str:
    return (
        f'Classify the sentiment of this movie review.\n'
        f'"{text}"\n\n'
        f'Answer with exactly one word: favorable or unfavorable.'
    )


def label_vocab_binary(text: str) -> str:
    return (
        f'Classify the sentiment of this movie review.\n'
        f'"{text}"\n\n'
        f'Answer with exactly one digit: 1 (positive) or 0 (negative).'
    )


LABEL_VOCAB_VARIANTS: dict[str, PromptFn] = {
    "positive_negative":    label_vocab_posneg,
    "good_bad":             label_vocab_goodbad,
    "favorable_unfavorable": label_vocab_favorable,
    "binary_1_0":           label_vocab_binary,
}


# ---------------------------------------------------------------------------
# Dimension 4: Instruction verb
# ---------------------------------------------------------------------------

def verb_classify(text: str) -> str:
    return (
        f'Classify the sentiment of the following movie review as positive or negative.\n\n'
        f'Review: "{text}"\n\n'
        f'Answer:'
    )


def verb_identify(text: str) -> str:
    return (
        f'Identify the sentiment of the following movie review as positive or negative.\n\n'
        f'Review: "{text}"\n\n'
        f'Answer:'
    )


def verb_determine(text: str) -> str:
    return (
        f'Determine whether the sentiment of the following movie review is '
        f'positive or negative.\n\n'
        f'Review: "{text}"\n\n'
        f'Answer:'
    )


def verb_label(text: str) -> str:
    return (
        f'Label the sentiment of the following movie review as positive or negative.\n\n'
        f'Review: "{text}"\n\n'
        f'Answer:'
    )


INSTRUCTION_VERB_VARIANTS: dict[str, PromptFn] = {
    "classify":  verb_classify,
    "identify":  verb_identify,
    "determine": verb_determine,
    "label":     verb_label,
}


# ---------------------------------------------------------------------------
# Consolidated exports
# ---------------------------------------------------------------------------

PROMPT_MATRIX: dict[str, dict[str, PromptFn]] = {
    "verbosity":        VERBOSITY_VARIANTS,
    "label_position":   LABEL_POSITION_VARIANTS,
    "label_vocab":      LABEL_VOCAB_VARIANTS,
    "instruction_verb": INSTRUCTION_VERB_VARIANTS,
}

# Flat list of (dimension, variant_name, prompt_fn) — convenient for iteration
ALL_VARIANTS: list[tuple[str, str, PromptFn]] = [
    (dim, name, fn)
    for dim, variants in PROMPT_MATRIX.items()
    for name, fn in variants.items()
]


if __name__ == "__main__":
    # Quick smoke-test: print all variants for a sample sentence
    sample = "This film is a masterpiece of modern cinema."
    print(f"Sample text: {sample!r}\n")
    print(f"Total variants: {len(ALL_VARIANTS)}\n")
    print("=" * 70)
    for dim, name, fn in ALL_VARIANTS:
        print(f"[{dim}] {name}")
        print("-" * 50)
        print(fn(sample))
        print()
