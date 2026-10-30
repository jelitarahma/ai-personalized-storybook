"""
Experiment runner comparing text decoding strategies (Greedy, Top-K, Top-p)
across multiple hyperparameters and seeds, producing outputs/decoding_experiments.csv.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from src.narrative.gpt2_generator import StoryGenerator
from src.narrative.evaluator import (
    tokenize_words,
    compute_distinct_n,
    compute_repetition_rate,
    compute_self_bleu,
    compute_dataset_ngram_overlap,
    compute_perplexity
)


def run_decoding_experiments(
    output_csv: str = "outputs/decoding_experiments.csv",
    model_name_or_path: str = "cahya/gpt2-small-indonesian-522M",
    prompt: str = "Di sebuah desa lereng bukit Nusantara, hiduplah seekor naga kecil bernama Gono. Gono sangat ingin menjadi pahlawan yang menolong sesama.",
    seeds: List[int] = [42, 100, 2024],
    k_values: List[int] = [10, 50, 100],
    p_values: List[float] = [0.7, 0.9, 0.95],
    max_new_tokens: int = 120
) -> pd.DataFrame:
    """
    Executes systematic decoding experiments and saves results to CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load dataset for memorization overlap check
    dataset_stories: List[str] = []
    dataset_path = "data/raw/indonesian_children_stories.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            dataset_stories = [item["story"] for item in raw_data]

    generator = StoryGenerator(model_name_or_path=model_name_or_path)

    configs: List[Dict[str, Any]] = [
        {"strategy": "greedy", "top_k": None, "top_p": None, "temp": 1.0}
    ]
    for k in k_values:
        configs.append({"strategy": "top_k", "top_k": k, "top_p": None, "temp": 0.8})
    for p in p_values:
        configs.append({"strategy": "top_p", "top_k": None, "top_p": p, "temp": 0.8})

    all_records: List[Dict[str, Any]] = []

    print(f"\nRunning decoding experiments across {len(configs)} configurations and {len(seeds)} seeds...")

    for cfg in configs:
        strategy = cfg["strategy"]
        top_k = cfg["top_k"]
        top_p = cfg["top_p"]
        temp = cfg["temp"]
        config_name = f"{strategy}_k{top_k or '-'}_p{top_p or '-'}"

        generated_samples_for_bleu: List[str] = []
        sample_metrics: List[Dict[str, Any]] = []

        for seed in seeds:
            print(f"  Testing {config_name} (seed={seed})...")
            res = generator.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                strategy=strategy,
                top_k=top_k,
                top_p=top_p,
                temperature=temp,
                seed=seed
            )
            story = res["story"]
            generated_samples_for_bleu.append(story)

            words = tokenize_words(story)
            dist1 = compute_distinct_n(words, n=1)
            dist2 = compute_distinct_n(words, n=2)
            rep_rate = compute_repetition_rate(words, n=3)
            ppl = compute_perplexity(story, generator.model, generator.tokenizer, generator.device)
            overlap = compute_dataset_ngram_overlap(story, dataset_stories, n_values=[4, 6])

            sample_metrics.append({
                "config_name": config_name,
                "strategy": strategy,
                "top_k": top_k if top_k is not None else -1,
                "top_p": top_p if top_p is not None else -1.0,
                "temperature": temp,
                "seed": seed,
                "total_words": len(words),
                "distinct_1": round(dist1, 4),
                "distinct_2": round(dist2, 4),
                "repetition_rate": round(rep_rate, 4),
                "perplexity": ppl,
                "overlap_4gram_pct": overlap.get("overlap_4gram_pct", 0.0),
                "overlap_6gram_pct": overlap.get("overlap_6gram_pct", 0.0),
                "is_novel": overlap.get("is_novel", True),
                "generated_story": story[:160] + "..."
            })

        # Calculate self-BLEU across the seeds for this configuration
        self_bleu = compute_self_bleu(generated_samples_for_bleu, max_n=2)
        for m in sample_metrics:
            m["self_bleu_across_seeds"] = round(self_bleu, 4)
            all_records.append(m)

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nExperiments completed successfully. Saved {len(df)} records to: {output_csv}")

    # Summary table
    summary = df.groupby("config_name")[["distinct_1", "distinct_2", "repetition_rate", "self_bleu_across_seeds", "perplexity", "overlap_4gram_pct"]].mean()
    print("\n--- Summary Table (Averages by Configuration) ---")
    print(summary.to_string())

    return df


if __name__ == "__main__":
    run_decoding_experiments()

