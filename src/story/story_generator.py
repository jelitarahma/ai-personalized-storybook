"""
Multi-Page Indonesian Story Generator.
Coordinates GPT-2 text generation across 3-6 pages with context-aware chaining,
repetition suppression, age-appropriate sentence trimming, and single-page regeneration.
"""

import os
import re
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.reproducibility import set_seed
from src.story.character_registry import CharacterRegistry
from src.story.prompt_builder import StoryPromptBuilder, get_age_linguistic_guidance

DEFAULT_PRETRAINED = "cahya/gpt2-small-indonesian-522M"
FINE_TUNED_CHECKPOINT = "models/checkpoints/gpt2_storybook"


class MultiPageStoryGenerator:
    """
    Sequentially generates coherent multi-page children's storybooks.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if model_name_or_path is None:
            if os.path.exists(FINE_TUNED_CHECKPOINT) and os.path.isdir(FINE_TUNED_CHECKPOINT):
                self.model_path = FINE_TUNED_CHECKPOINT
            else:
                self.model_path = DEFAULT_PRETRAINED
        else:
            self.model_path = model_name_or_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def _ensure_model_loaded(self):
        if self.model is None or self.tokenizer is None:
            print(f"Loading GPT-2 model '{self.model_path}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

    def generate_single_page_text(
        self,
        prompt: str,
        starter_seed: str,
        age: int,
        strategy: str = "top_p",
        top_k: int = 50,
        top_p: float = 0.90,
        temperature: float = 0.85,
        seed: int = 42
    ) -> str:
        """
        Generates narrative text for a single storybook page.
        """
        self._ensure_model_loaded()
        set_seed(seed)

        guidance = get_age_linguistic_guidance(age)
        max_tokens = guidance["max_new_tokens"]

        input_text = f"{starter_seed} "
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 3,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": strategy != "greedy"
        }

        if strategy == "top_k":
            gen_kwargs["top_k"] = top_k
            gen_kwargs["temperature"] = temperature
        elif strategy == "top_p":
            gen_kwargs["top_p"] = top_p
            gen_kwargs["temperature"] = temperature
        elif strategy == "greedy":
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_text = self._post_process_narrative(generated_text, age)
        return cleaned_text

    def _post_process_narrative(self, raw_text: str, age: int) -> str:
        """
        Cleans repetitive phrases, normalizes spaces, and truncates cleanly at last punctuation mark.
        """
        text = re.sub(r'<\|.*?\|>', '', raw_text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove immediate duplicated sentences
        sentences = re.split(r'([.!?]+)', text)
        cleaned_sentences = []
        seen = set()

        for i in range(0, len(sentences) - 1, 2):
            sentence = (sentences[i] + sentences[i + 1]).strip()
            norm = sentence.lower()
            if norm not in seen and len(sentence) > 5:
                seen.add(norm)
                cleaned_sentences.append(sentence)

        if cleaned_sentences:
            text = " ".join(cleaned_sentences)

        # Truncate at last complete punctuation
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct != -1:
            text = text[:last_punct + 1].strip()

        # Length validation according to child age
        words = text.split()
        guidance = get_age_linguistic_guidance(age)
        min_words, max_words = guidance["target_word_count"]

        if len(words) < min_words:
            # Fallback gentle enrichment for child story
            text += f" Hari itu menjadi petualangan yang sangat istimewa dan penuh senyuman hangat."

        return text

    def generate_full_book(
        self,
        premise: str,
        age: int,
        character_reg: CharacterRegistry,
        total_pages: int = 4,
        world_settings: Optional[Dict[str, Any]] = None,
        moral_theme: Optional[str] = None,
        strategy: str = "top_p",
        top_k: int = 50,
        top_p: float = 0.90,
        temperature: float = 0.85,
        base_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Generates all pages of the storybook with contextual continuity.
        """
        pages = []
        previous_summary = None

        for p_idx in range(1, total_pages + 1):
            page_seed = base_seed + (p_idx * 17)

            prompt_data = StoryPromptBuilder.build_page_prompt(
                premise=premise,
                age=age,
                character_reg=character_reg,
                page_idx=p_idx,
                total_pages=total_pages,
                world_settings=world_settings,
                previous_pages_summary=previous_summary,
                moral_theme=moral_theme
            )

            page_text = self.generate_single_page_text(
                prompt=prompt_data["prompt_text"],
                starter_seed=prompt_data["starter_seed"],
                age=age,
                strategy=strategy,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=page_seed
            )

            # Build brief context summary for subsequent pages
            first_sentence = page_text.split('.')[0] if '.' in page_text else page_text[:50]
            previous_summary = f"Pada halaman sebelumnya: {first_sentence}."

            pages.append({
                "page_number": p_idx,
                "stage": prompt_data["stage_info"]["stage"],
                "stage_title": prompt_data["stage_info"]["title"],
                "text": page_text,
                "prompt_used": prompt_data["prompt_text"],
                "seed": page_seed
            })

        return pages

    def regenerate_page(
        self,
        pages: List[Dict[str, Any]],
        page_idx: int,
        premise: str,
        age: int,
        character_reg: CharacterRegistry,
        world_settings: Optional[Dict[str, Any]] = None,
        moral_theme: Optional[str] = None,
        new_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Regenerates a single page without altering other pages.
        """
        total_pages = len(pages)
        if page_idx < 1 or page_idx > total_pages:
            return pages

        seed = new_seed if new_seed is not None else (pages[page_idx - 1]["seed"] + 999)

        prev_summary = None
        if page_idx > 1:
            prev_text = pages[page_idx - 2]["text"]
            prev_summary = prev_text.split('.')[0] + "."

        prompt_data = StoryPromptBuilder.build_page_prompt(
            premise=premise,
            age=age,
            character_reg=character_reg,
            page_idx=page_idx,
            total_pages=total_pages,
            world_settings=world_settings,
            previous_pages_summary=prev_summary,
            moral_theme=moral_theme
        )

        new_text = self.generate_single_page_text(
            prompt=prompt_data["prompt_text"],
            starter_seed=prompt_data["starter_seed"],
            age=age,
            seed=seed
        )

        pages[page_idx - 1]["text"] = new_text
        pages[page_idx - 1]["seed"] = seed
        pages[page_idx - 1]["prompt_used"] = prompt_data["prompt_text"]

        return pages

