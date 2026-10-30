"""
Narrative evaluation module for measuring coherence, diversity, profile alignment,
and dataset memorization (n-gram overlap) in generated children's stories.
"""

import math
import re
import torch
from collections import Counter
from typing import List, Dict, Any, Set, Tuple


def tokenize_words(text: str) -> List[str]:
    """
    Simple word tokenizer for Indonesian text (lowercased, alphanumeric).
    """
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


def compute_distinct_n(tokens: List[str], n: int = 1) -> float:
    """
    Compute Distinct-n metric: ratio of unique n-grams to total n-grams.
    Higher value indicates greater lexical diversity / creativity.
    """
    if len(tokens) < n or len(tokens) == 0:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_rate(tokens: List[str], n: int = 3) -> float:
    """
    Compute repetition rate: ratio of duplicate n-grams to total n-grams.
    Lower value indicates less repetitive text.
    """
    if len(tokens) < n or len(tokens) == 0:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(ngrams)


def compute_self_bleu(samples: List[str], max_n: int = 2) -> float:
    """
    Compute simplified Self-BLEU metric across a list of generated samples.
    Lower Self-BLEU signifies higher diversity across generations.
    """
    if len(samples) < 2:
        return 0.0

    tokenized_samples = [tokenize_words(s) for s in samples]
    bleu_scores = []

    for i, hyp in enumerate(tokenized_samples):
        refs = [tokenized_samples[j] for j in range(len(tokenized_samples)) if j != i]
        if not hyp or not refs:
            continue
        
        # Calculate precision for n-grams up to max_n
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter([tuple(hyp[k:k+n]) for k in range(len(hyp) - n + 1)])
            if not hyp_ngrams:
                precisions.append(1e-9)
                continue
            
            # Max count in any reference
            max_ref_counts = Counter()
            for ref in refs:
                ref_ngrams = Counter([tuple(ref[k:k+n]) for k in range(len(ref) - n + 1)])
                for ng in hyp_ngrams:
                    max_ref_counts[ng] = max(max_ref_counts[ng], ref_ngrams[ng])
            
            clipped = sum(min(count, max_ref_counts[ng]) for ng, count in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            precisions.append((clipped / total) if total > 0 else 1e-9)

        # Geometric mean
        log_prec_sum = sum(math.log(p if p > 0 else 1e-9) for p in precisions) / max_n
        bleu = math.exp(log_prec_sum)
        bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def extract_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """
    Extract set of n-grams from a token list.
    """
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_dataset_ngram_overlap(
    generated_text: str,
    dataset_texts: List[str],
    n_values: List[int] = [4, 6]
) -> Dict[str, float]:
    """
    Check if the model creates new narrative or copies training data:
    computes the percentage of generated n-grams that appear in the training dataset.

    Returns:
        dict: e.g. {'overlap_4gram_pct': 5.2, 'overlap_6gram_pct': 0.0, 'is_novel': True}
    """
    gen_tokens = tokenize_words(generated_text)
    results = {}

    for n in n_values:
        gen_ngrams = extract_ngrams(gen_tokens, n)
        if not gen_ngrams:
            results[f"overlap_{n}gram_pct"] = 0.0
            continue

        # Collect all n-grams from dataset
        dataset_ngrams: Set[Tuple[str, ...]] = set()
        for doc in dataset_texts:
            doc_tokens = tokenize_words(doc)
            dataset_ngrams.update(extract_ngrams(doc_tokens, n))

        matched = gen_ngrams.intersection(dataset_ngrams)
        overlap_pct = (len(matched) / len(gen_ngrams)) * 100.0
        results[f"overlap_{n}gram_pct"] = round(overlap_pct, 2)
        results[f"matched_{n}gram_count"] = len(matched)
        results[f"total_gen_{n}grams"] = len(gen_ngrams)

    # Consider novel if 6-gram overlap is below 15% and 4-gram overlap is below 35%
    results["is_novel"] = (
        results.get("overlap_6gram_pct", 0.0) < 15.0 and
        results.get("overlap_4gram_pct", 0.0) < 35.0
    )
    return results


def compute_profile_keyword_alignment(
    story_text: str,
    profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculates alignment between the child's input profile and the generated story.
    Checks presence of child's name, genre keywords, and thematic concepts.
    """
    story_lower = story_text.lower()
    name = str(profile.get("name", "")).lower()
    genre = str(profile.get("genre", "")).lower()
    theme = str(profile.get("theme", "")).lower()
    culture = str(profile.get("cultural_background", "")).lower()

    name_matched = (name in story_lower) if name else False

    # Extract theme key terms
    theme_terms = [t for t in re.findall(r'\b\w{4,}\b', theme) if t not in ["dan", "yang", "pada"]]
    theme_matches = [term for term in theme_terms if term in story_lower]
    theme_match_ratio = len(theme_matches) / len(theme_terms) if theme_terms else 1.0

    genre_terms = [g for g in re.findall(r'\b\w{4,}\b', genre)]
    genre_matches = [term for term in genre_terms if term in story_lower]
    genre_match_ratio = len(genre_matches) / len(genre_terms) if genre_terms else 1.0

    culture_matched = (culture in story_lower) if culture and culture != "nasional" else True

    # Alignment score out of 100
    alignment_score = (
        (40.0 if name_matched else 0.0) +
        (30.0 * theme_match_ratio) +
        (20.0 * genre_match_ratio) +
        (10.0 if culture_matched else 0.0)
    )

    return {
        "alignment_score": round(alignment_score, 2),
        "name_matched": name_matched,
        "theme_matched_terms": theme_matches,
        "theme_match_ratio": round(theme_match_ratio, 2),
        "genre_matched_terms": genre_matches,
        "culture_matched": culture_matched
    }


def compute_perplexity(
    text: str,
    model,
    tokenizer,
    device: str = "cpu"
) -> float:
    """
    Computes text perplexity under the specified GPT-2 model.
    PPL = exp(loss). Lower perplexity indicates higher fluency according to the model.
    """
    if model is None or tokenizer is None:
        return 25.0  # nominal default for fallback mode

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    if input_ids.shape[1] < 2:
        return 1.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    return round(math.exp(loss), 3) if loss < 20 else 999.0


def evaluate_story_comprehensive(
    story_text: str,
    profile: Dict[str, Any],
    dataset_stories: List[str],
    model = None,
    tokenizer = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run full suite of automated metrics for a generated story.
    """
    tokens = tokenize_words(story_text)
    
    dist1 = compute_distinct_n(tokens, n=1)
    dist2 = compute_distinct_n(tokens, n=2)
    rep_rate = compute_repetition_rate(tokens, n=3)
    alignment = compute_profile_keyword_alignment(story_text, profile)
    ngram_overlap = compute_dataset_ngram_overlap(story_text, dataset_stories)
    ppl = compute_perplexity(story_text, model, tokenizer, device)

    return {
        "word_count": len(tokens),
        "distinct_1": round(dist1, 4),
        "distinct_2": round(dist2, 4),
        "repetition_rate": round(rep_rate, 4),
        "perplexity": ppl,
        "profile_alignment": alignment,
        "dataset_overlap": ngram_overlap,
        "manual_rubric_template": {
            "koherensi_alur (1-5)": None,
            "kesesuaian_profil_anak (1-5)": None,
            "kreativitas_narasi (1-5)": None,
            "nilai_moral_dan_edukasi (1-5)": None,
            "kesesuaian_usia (1-5)": None
        }
    }

