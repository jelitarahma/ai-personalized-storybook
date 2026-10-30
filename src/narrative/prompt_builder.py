"""
Personalized prompt builder module.
Translates a child's profile into structured, age-adaptive, and culturally sensitive narrative prompts.
"""

from typing import Dict, Any


def get_age_guideline(age: int) -> Dict[str, Any]:
    """
    Returns linguistic and structural guidelines tailored to the child's developmental age.
    """
    if age <= 7:
        return {
            "sentence_length": "pendek dan berirama (5-10 kata per kalimat)",
            "vocabulary": "kosakata dasar, ceria, dan mudah dipahami",
            "plot_structure": "alur linear sederhana dengan pesan moral eksplisit",
            "tone": "hangat, ramah, dan penuh keajaiban",
            "target_tokens": 120
        }
    elif age <= 10:
        return {
            "sentence_length": "sedang (10-18 kata per kalimat)",
            "vocabulary": "kosakata deskriptif dengan pengenalan konsep alam dan kerja sama",
            "plot_structure": "alur petualangan dengan rintangan dan pemecahan masalah",
            "tone": "semangat petualangan, rasa ingin tahu yang tinggi, dan inspiratif",
            "target_tokens": 180
        }
    else:
        return {
            "sentence_length": "kompleks dan variatif (15-25 kata per kalimat)",
            "vocabulary": "kosakata kaya, metafora halus, dan eksplorasi karakter mendalam",
            "plot_structure": "alur multi-tahap dengan dilema moral dan kepemimpinan",
            "tone": "reflektif, penuh intrik positif, dan memberdayakan",
            "target_tokens": 240
        }


def build_personalized_prompt(
    profile: Dict[str, Any],
    format_type: str = "natural"
) -> str:
    """
    Build a tailored prompt from a child profile dictionary.

    Args:
        profile (dict): Contains 'name', 'age', 'genre', 'theme', and optional 'cultural_background', 'character_preference'.
        format_type (str): 'natural' for human-readable instruction prompt,
                           'tokenized' for fine-tuned prefix token structure.

    Returns:
        str: Prompt string.
    """
    name = profile.get("name", "Sahabat Cilik")
    age = int(profile.get("age", 7))
    genre = profile.get("genre", "Fantasi")
    theme = profile.get("theme", "Keberanian dan Kebaikan Hati")
    culture = profile.get("cultural_background", "Nusantara")
    character = profile.get("character_preference", "seorang petualang cilik")

    guidelines = get_age_guideline(age)

    if format_type == "tokenized":
        # Formatted prefix matching dataset special tokens
        prompt = (
            f"<|startoftext|> [GENRE: {genre}] [USIA: {age} tahun] [TEMA: {theme}] "
            f"<|story|> Di sebuah tempat yang indah bernuansa {culture}, {character} bernama {name}"
        )
    else:
        # Natural language instruction prompt for zero-shot or general GPT-2 prompting
        prompt = (
            f"Tuliskan sebuah cerita {genre} ramah anak untuk pembaca berusia {age} tahun. "
            f"Karakter utama bernama {name}, {character} dari latar budaya {culture}. "
            f"Tema cerita adalah {theme}. "
            f"Gunakan gaya bahasa {guidelines['tone']} dengan kalimat {guidelines['sentence_length']} "
            f"dan {guidelines['vocabulary']}. "
            f"Cerita dimulai: Pada suatu hari yang cerah,"
        )

    return prompt.strip()


def build_story_seed_prefix(profile: Dict[str, Any]) -> str:
    """
    Creates the introductory starter clause for autoregressive GPT-2 generation
    that primes the language model with the child's identity and setting.
    """
    name = profile.get("name", "Gono")
    age = int(profile.get("age", 7))
    genre = profile.get("genre", "Fantasi")
    theme = profile.get("theme", "Kebaikan Hati")
    culture = profile.get("cultural_background", "Indonesia")
    character = profile.get("character_preference", "anak pemberani")

    starter = f"Di sebuah desa yang damai di Nusantara, hiduplah {character} bernama {name}. {name} adalah anak berusia {age} tahun yang sangat ceria dan berhati mulia."
    return starter

