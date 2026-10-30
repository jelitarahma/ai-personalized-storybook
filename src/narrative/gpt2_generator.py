"""
GPT-2 Text Generation engine supporting Top-K, Top-p (Nucleus), Temperature,
and reproducible seed controls for Indonesian personalized children's stories.
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure project root is in sys.path for standalone script execution
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.reproducibility import set_seed

DEFAULT_MODEL_NAME = "cahya/gpt2-small-indonesian-522M"


class StoryGenerator:
    """
    Indonesian GPT-2 narrative generator for personalized children's storybooks.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """
        Loads the GPT-2 model and tokenizer from Hugging Face or local path.
        """
        print(f"Loading Indonesian GPT-2 model '{self.model_name}' on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"[Warning] Failed to load {self.model_name}: {e}")
            print("Falling back to standard 'gpt2' or lightweight initialized model.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                self.model.to(self.device)
                self.model.eval()
            except Exception as e2:
                print(f"[Warning] Fallback also failed: {e2}. Operating in template fallback mode.")
                self.model = None
                self.tokenizer = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        strategy: str = "top_p",
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generate story continuation given a prompt.

        Args:
            prompt (str): Starting text.
            max_new_tokens (int): Maximum tokens to generate.
            strategy (str): 'greedy', 'top_k', 'top_p', or 'hybrid'.
            top_k (int): Top-K filter threshold.
            top_p (float): Top-p nucleus cumulative probability.
            temperature (float): Randomness scale.
            repetition_penalty (float): Penalty for repeated tokens.
            seed (int): Reproducibility seed.

        Returns:
            dict: {
                'story': clean output story string,
                'raw_output': raw text,
                'prompt': input prompt,
                'strategy': strategy name,
                'params': { ... }
            }
        """
        set_seed(seed)

        # Build generation kwargs based on strategy
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": 3,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else 50256,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else 50256,
        }

        if strategy == "greedy":
            gen_kwargs["do_sample"] = False
        elif strategy == "top_k":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = top_k or 50
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["temperature"] = temperature
        elif strategy == "top_p":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = 0
            gen_kwargs["top_p"] = top_p or 0.9
            gen_kwargs["temperature"] = temperature
        elif strategy == "hybrid":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = top_k or 50
            gen_kwargs["top_p"] = top_p or 0.9
            gen_kwargs["temperature"] = temperature
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

            # Decode only the newly generated tokens
            generated_ids = output_ids[0][input_length:]
            continuation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # Fallback narrative synthesis for constrained testing
            continuation = (
                " Di tengah perjalanannya, ia menemukan pohon ajaib yang memancarkan cahaya keemasan. "
                "Dengan penuh rasa ingin tahu dan keberanian, ia menolong seekor burung kecil yang sayapnya terluka. "
                "Burung itu kemudian membantunya menemukan jalan pulang menuju desa dengan selamat."
            )
            full_raw = prompt + continuation

        # Clean up continuation text: cut at the last complete sentence
        clean_story = self._postprocess_story(prompt, continuation)

        return {
            "story": clean_story,
            "raw_output": full_raw,
            "prompt": prompt,
            "strategy": strategy,
            "params": {
                "max_new_tokens": max_new_tokens,
                "strategy": strategy,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "seed": seed
            }
        }

    def _postprocess_story(self, prompt: str, continuation: str) -> str:
        """
        Assembles and cleans the generated story, cutting cleanly at sentence ends.
        """
        combined = prompt.strip() + " " + continuation.strip()
        # Find last sentence-ending punctuation (., !, ?)
        match = re.search(r'^(.*[.!?])', combined, re.DOTALL)
        if match:
            clean = match.group(1).strip()
        else:
            clean = combined.strip() + "."
        
        # Remove consecutive duplicated words or phrases
        clean = re.sub(r'\b(\w+)( \1\b)+', r'\1', clean, flags=re.IGNORECASE)
        return clean

