"""
Comprehensive Narrative & Memorization Evaluator.
"""

from src.narrative.evaluator import (
    tokenize_words,
    compute_distinct_n,
    compute_repetition_rate,
    compute_self_bleu,
    compute_dataset_ngram_overlap,
    compute_perplexity,
    compute_profile_keyword_alignment,
    evaluate_story_comprehensive
)

__all__ = [
    "tokenize_words",
    "compute_distinct_n",
    "compute_repetition_rate",
    "compute_self_bleu",
    "compute_dataset_ngram_overlap",
    "compute_perplexity",
    "compute_profile_keyword_alignment",
    "evaluate_story_comprehensive"
]
