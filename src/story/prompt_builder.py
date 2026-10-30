"""
Contextual Prompt Builder for Multi-Page Indonesian Children's Stories.
Builds age-adaptive, page-specific generation prompts incorporating character registries,
previous page continuity, and world settings without enforcing rigid genre restrictions.
"""

from typing import Dict, Any, List, Optional
from src.story.character_registry import CharacterRegistry, Character
from src.story.story_arc import StoryArcPlanner


def get_age_linguistic_guidance(age: int) -> Dict[str, Any]:
    """
    Returns psycholinguistic parameters tailored to child reader age (Solanki & Khublani, 2024).
    """
    if age <= 7:
        return {
            "sentence_structure": "Kalimat pendek, berirama riang, struktur S-P-O sederhana.",
            "target_word_count": (35, 60),
            "vocab_level": "Kosakata konkret, akrab sehari-hari, penuh kehangatan.",
            "max_new_tokens": 85
        }
    elif age <= 10:
        return {
            "sentence_structure": "Kalimat majemuk sederhana, alur bertahap dengan rasa ingin tahu dan kerjasama.",
            "target_word_count": (50, 85),
            "vocab_level": "Kosakata deskriptif petualangan, dialog hangat, eksploratif.",
            "max_new_tokens": 115
        }
    else:
        return {
            "sentence_structure": "Kalimat deskriptif kaya alur sebab-akibat, pemecahan masalah, dan refleksi diri.",
            "target_word_count": (65, 110),
            "vocab_level": "Kosakata kaya imajinatif, nilai empati mendalam dan keberanian.",
            "max_new_tokens": 140
        }


class StoryPromptBuilder:
    """
    Constructs prompts for multi-page generation maintaining character and plot continuity.
    """

    @staticmethod
    def build_page_prompt(
        premise: str,
        age: int,
        character_reg: CharacterRegistry,
        page_idx: int,
        total_pages: int,
        world_settings: Optional[Dict[str, Any]] = None,
        previous_pages_summary: Optional[str] = None,
        moral_theme: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds page-specific prompt instructions for GPT-2 narrative generator.
        """
        world = world_settings or {}
        location = world.get("location", "alam terbuka yang indah")
        atmosphere = world.get("atmosphere", "hangat dan damai")
        weather = world.get("weather", "cerah menyenangkan")
        cultural_flavor = world.get("culture", "Nusantara ramah")

        guidance = get_age_linguistic_guidance(age)
        stage_info = StoryArcPlanner.get_stage_definition(page_idx, total_pages)

        main_char = character_reg.get_main_character()
        main_char_desc = main_char.get_physical_description() if main_char else "Seorang anak ramah"
        supporting_chars = character_reg.get_supporting_characters()
        supporting_desc = ", ".join([c.get_physical_description() for c in supporting_chars]) if supporting_chars else "Sahabat alam"

        # Structured prompt prefix
        prompt_lines = [
            f"<|startoftext|>",
            f"<|usia|> {age} tahun",
            f"<|tokoh|> {main_char.name if main_char else 'Sahabat'}",
            f"<|halaman|> {page_idx} dari {total_pages} ({stage_info['stage']})",
            f"<|latar|> {location}, suasana {atmosphere}, cuaca {weather}",
        ]

        if moral_theme and moral_theme.strip():
            prompt_lines.append(f"<|tema|> {moral_theme.strip()}")

        prompt_lines.append(f"<|fokus_adegan|> {stage_info['focus']}")

        if previous_pages_summary:
            prompt_lines.append(f"<|kejadian_sebelumnya|> {previous_pages_summary}")

        prompt_lines.append(f"<|cerita|>")

        # Starter sentence seed to guide the model directly into storytelling tone
        starter_seed = StoryPromptBuilder._create_starter_seed(
            page_idx=page_idx,
            total_pages=total_pages,
            main_name=main_char.name if main_char else "Sang Tokoh",
            location=location,
            premise=premise,
            previous_summary=previous_pages_summary
        )

        full_prompt = "\n".join(prompt_lines)

        return {
            "prompt_text": full_prompt,
            "starter_seed": starter_seed,
            "guidance": guidance,
            "stage_info": stage_info,
            "main_char_desc": main_char_desc,
            "supporting_desc": supporting_desc
        }

    @staticmethod
    def _create_starter_seed(
        page_idx: int,
        total_pages: int,
        main_name: str,
        location: str,
        premise: str,
        previous_summary: Optional[str]
    ) -> str:
        """
        Creates an evocative opening sentence starter tailored to the story stage.
        """
        if page_idx == 1:
            clean_premise = premise.strip().rstrip(".")
            if len(clean_premise) > 10 and not clean_premise.lower().startswith("tentang"):
                return f"Di {location}, {main_name} bersiap mengawali harinya karena {clean_premise.lower()}."
            else:
                return f"Di sebuah {location} yang indah, hiduplah {main_name} yang selalu bersemangat."
        elif page_idx == total_pages:
            return f"Akhirnya, dengan hati yang riang dan bersyukur, {main_name} tersenyum lebar."
        else:
            return f"Langkah {main_name} berlanjut dengan penuh rasa penasaran di antara {location}."

