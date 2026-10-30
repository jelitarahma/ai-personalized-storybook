"""
Visual Evaluation Module.
Evaluates visual consistency, cross-story visual diversity, and logs translation obstacles.
Produces evaluation grids and structured metrics for Module B.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Any, Tuple


def compute_visual_diversity(image_paths: List[str]) -> Dict[str, float]:
    """
    Computes pairwise pixel-level and color histogram diversity across a set of images.
    Higher diversity score indicates that different stories yield distinctly unique visuals.
    """
    if len(image_paths) < 2:
        return {"mean_pairwise_difference": 0.0, "diversity_score": 0.0}

    images = []
    histograms = []
    for p in image_paths:
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((64, 64))
            arr = np.array(im, dtype=np.float32) / 255.0
            images.append(arr)
            # 3D color histogram
            hist, _ = np.histogramdd(arr.reshape(-1, 3), bins=(8, 8, 8), range=[(0, 1), (0, 1), (0, 1)])
            hist = hist / (np.sum(hist) + 1e-9)
            histograms.append(hist.flatten())

    if len(images) < 2:
        return {"mean_pairwise_difference": 0.0, "diversity_score": 0.0}

    diffs = []
    hist_dists = []
    n = len(images)
    for i in range(n):
        for j in range(i + 1, n):
            # Mean absolute pixel difference
            l1_diff = np.mean(np.abs(images[i] - images[j]))
            diffs.append(l1_diff)
            # Histogram distance (Euclidean)
            h_dist = np.linalg.norm(histograms[i] - histograms[j])
            hist_dists.append(h_dist)

    mean_diff = float(np.mean(diffs))
    mean_hist_dist = float(np.mean(hist_dists))
    # Normalized diversity score (0 - 100)
    diversity_score = round(min(mean_diff * 200.0, 100.0), 2)

    return {
        "mean_pixel_difference": round(mean_diff, 4),
        "mean_histogram_distance": round(mean_hist_dist, 4),
        "diversity_score": diversity_score
    }


def create_cross_story_visual_grid(
    scenarios: List[Dict[str, Any]],
    output_path: str = "outputs/visual_evaluations/cross_story_visual_grid.png"
) -> str:
    """
    Assembles a side-by-side comparison grid of character portraits and scene illustrations
    across multiple story scenarios to demonstrate visual variety and text-to-visual fidelity.
    """
    n = len(scenarios)
    if n == 0:
        return ""

    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for idx, sc in enumerate(scenarios):
        title = sc.get("title", f"Skenario {idx+1}")
        char_path = sc.get("char_path")
        scene_path = sc.get("scene_path")
        attrs = sc.get("attributes", {})

        # Plot Character
        if char_path and os.path.exists(char_path):
            char_img = Image.open(char_path)
            axes[idx, 0].imshow(char_img)
        axes[idx, 0].set_title(f"Karakter: {title}\n(Rambut: {attrs.get('character', {}).get('hair_color')}, Baju: {attrs.get('character', {}).get('clothing_color')})", fontsize=10)
        axes[idx, 0].axis("off")

        # Plot Scene
        if scene_path and os.path.exists(scene_path):
            scene_img = Image.open(scene_path)
            axes[idx, 1].imshow(scene_img)
        axes[idx, 1].set_title(f"Scene Setting: {attrs.get('scene', {}).get('location')}\n(Suasana: {attrs.get('scene', {}).get('lighting_atmosphere')})", fontsize=10)
        axes[idx, 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Cross-story visual grid saved: {output_path}")
    return output_path


def log_visual_translation_obstacles(
    all_extracted_attrs: List[Dict[str, Any]],
    output_log_path: str = "outputs/visual_evaluations/text_to_visual_obstacles.json"
) -> Dict[str, Any]:
    """
    Analyzes and logs recurring obstacles, missing attributes, and ambiguities
    encountered during the text-to-visual translation process across stories.
    """
    total_stories = len(all_extracted_attrs)
    missing_hair_count = 0
    missing_clothing_count = 0
    missing_expression_count = 0
    missing_location_count = 0
    ambiguity_log = []

    for idx, item in enumerate(all_extracted_attrs):
        log_info = item.get("detection_log", {})
        notes = log_info.get("notes", [])
        for note in notes:
            if "rambut" in note:
                missing_hair_count += 1
            if "pakaian" in note:
                missing_clothing_count += 1
            if "ekspresi" in note:
                missing_expression_count += 1
            if "lokasi" in note:
                missing_location_count += 1
            ambiguity_log.append({
                "story_index": idx + 1,
                "note": note
            })

    report = {
        "total_stories_analyzed": total_stories,
        "obstacle_frequencies": {
            "missing_hair_attribute": missing_hair_count,
            "missing_clothing_attribute": missing_clothing_count,
            "missing_expression_attribute": missing_expression_count,
            "missing_location_attribute": missing_location_count
        },
        "default_fallback_rate_pct": round(
            (missing_hair_count + missing_clothing_count + missing_expression_count + missing_location_count) /
            (total_stories * 4 + 1e-9) * 100.0,
            2
        ),
        "challenges_summary": [
            "Cerita anak kerap memfokuskan aksi naratif daripada deskripsi fisik mendetail (misal warna rambut sering kali implisit).",
            "Kata sifat warna kadang bersinggungan dengan elemen metafora (misal 'hati emas' bukan rambut emas).",
            "Transisi adegan dalam satu cerita (pagi ke malam, rumah ke hutan) menuntut penentuan 'key frame' latar utama."
        ],
        "detailed_notes": ambiguity_log
    }

    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    with open(output_log_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report

