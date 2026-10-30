"""
Latent Mapping with NumPy.
Constructs semantically conditioned latent vectors z in R^D using attribute direction vectors,
seed decoupling (identity seed vs page variation seed), and hypersphere normalization.
"""

import numpy as np
from typing import Dict, Any, Tuple


class SemanticLatentMapper:
    """
    Manages semantic latent vector synthesis and attribute steering using linear algebra.
    """

    def __init__(self, latent_dim: int = 32, base_seed: int = 42):
        self.latent_dim = latent_dim
        self.base_seed = base_seed
        self._init_attribute_directions()

    def _init_attribute_directions(self):
        """
        Initializes orthogonal-like semantic directional vectors for physical & scenic traits.
        """
        rng = np.random.RandomState(1337)
        self.directions = {}

        attributes = [
            # Character attributes
            "char_child", "char_animal", "char_cheerful", "char_curious", "char_brave",
            "hair_dark", "hair_blonde", "hair_brown", "clothing_warm", "clothing_adventurous",
            # Scenic attributes
            "scene_river", "scene_forest", "scene_hills", "scene_house", "scene_garden",
            # Weather & atmosphere
            "time_sunny_morning", "time_golden_dusk", "time_starlit_night",
            "mood_peaceful", "mood_joyful", "mood_mysterious"
        ]

        for attr in attributes:
            vec = rng.randn(self.latent_dim)
            vec = vec / np.linalg.norm(vec)
            self.directions[attr] = vec

    def map_scene_to_latent(
        self,
        scene_json: Dict[str, Any],
        identity_seed: int = 42,
        page_variation_seed: int = 101,
        alpha_scale: float = 0.35
    ) -> Dict[str, Any]:
        """
        Maps a Scene JSON to a conditioned latent vector z via vector arithmetic.
        Formula: z_cond = z_identity + z_page_noise + sum(alpha_k * v_k)
        """
        # 1. Base identity noise
        rng_id = np.random.RandomState(identity_seed)
        z_id = rng_id.randn(self.latent_dim)
        z_id = z_id / np.linalg.norm(z_id)

        # 2. Page variation noise
        rng_page = np.random.RandomState(page_variation_seed)
        z_page = rng_page.randn(self.latent_dim) * 0.25

        # 3. Accumulated attribute directions
        shift = np.zeros(self.latent_dim)
        active_attributes = []

        # Location cues
        loc_text = scene_json.get("location", "").lower()
        if "sungai" in loc_text or "air" in loc_text:
            shift += self.directions["scene_river"] * 0.8
            active_attributes.append("scene_river")
        elif "hutan" in loc_text or "pohon" in loc_text:
            shift += self.directions["scene_forest"] * 0.8
            active_attributes.append("scene_forest")
        elif "bukit" in loc_text:
            shift += self.directions["scene_hills"] * 0.8
            active_attributes.append("scene_hills")
        elif "rumah" in loc_text:
            shift += self.directions["scene_house"] * 0.8
            active_attributes.append("scene_house")

        # Weather cues
        weather_text = scene_json.get("time_and_weather", "").lower()
        if "senja" in weather_text or "sore" in weather_text:
            shift += self.directions["time_golden_dusk"] * 0.6
            active_attributes.append("time_golden_dusk")
        elif "malam" in weather_text:
            shift += self.directions["time_starlit_night"] * 0.6
            active_attributes.append("time_starlit_night")
        else:
            shift += self.directions["time_sunny_morning"] * 0.6
            active_attributes.append("time_sunny_morning")

        # Main character cues
        chars = scene_json.get("characters_present", [])
        if chars:
            main_c = chars[0]
            c_type = main_c.get("character_type", "anak")
            if c_type == "hewan":
                shift += self.directions["char_animal"] * 0.7
                active_attributes.append("char_animal")
            else:
                shift += self.directions["char_child"] * 0.7
                active_attributes.append("char_child")

            exp = main_c.get("expression", "").lower()
            if "ceria" in exp or "gembira" in exp:
                shift += self.directions["char_cheerful"] * 0.5
                active_attributes.append("char_cheerful")
            elif "penasaran" in exp:
                shift += self.directions["char_curious"] * 0.5
                active_attributes.append("char_curious")

        # Combine
        z_raw = z_id + z_page + (alpha_scale * shift)

        # Hypersphere normalization: maintain expected norm sqrt(D)
        expected_norm = np.sqrt(self.latent_dim)
        z_conditioned = (z_raw / np.linalg.norm(z_raw)) * expected_norm

        cosine_sim = float(np.dot(z_id, z_conditioned) / (np.linalg.norm(z_id) * np.linalg.norm(z_conditioned)))

        return {
            "latent_vector": z_conditioned.tolist(),
            "latent_dim": self.latent_dim,
            "norm": float(np.linalg.norm(z_conditioned)),
            "cosine_similarity_to_base": round(cosine_sim, 4),
            "active_attributes": active_attributes,
            "identity_seed": identity_seed,
            "page_seed": page_variation_seed
        }

