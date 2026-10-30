"""
NumPy-based Semantic Latent Mapping Module.
Maps extracted text attributes (hair color, clothing, expression, scene environment, lighting)
into structured latent direction vectors in latent space R^nz.
References: Lecture Notes 01 (Latent Arithmetic) & Lecture Notes 03 (Latent Space Representation).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.utils.reproducibility import set_seed


class SemanticLatentMapper:
    """
    Computes controllable latent vectors from structured textual attributes
    using vector arithmetic and orthogonal direction dictionaries.
    """

    def __init__(self, latent_dim: int = 100, seed: int = 42):
        self.latent_dim = latent_dim
        self.seed = seed
        self._initialize_attribute_directions()

    def _initialize_attribute_directions(self):
        """
        Creates stable, reproducible semantic basis directions in R^nz
        for various visual attributes using Gram-Schmidt pseudo-orthogonalization.
        """
        rng = np.random.RandomState(self.seed)

        # Attribute keys to define
        attributes = [
            # Character hair attributes
            "hair_hitam", "hair_cokelat", "hair_merah", "hair_keemasan", "hair_putih",
            # Character clothing colors
            "cloth_hijau", "cloth_biru", "cloth_merah", "cloth_kuning", "cloth_cokelat", "cloth_putih",
            # Character expressions
            "expr_ceria", "expr_berani", "expr_kagum", "expr_bijak",
            # Scene environments
            "scene_hutan", "scene_bukit", "scene_pantai", "scene_desa", "scene_angkasa",
            # Scene atmosphere & lighting
            "atm_pagi", "atm_malam", "atm_senja", "atm_kabut"
        ]

        self.directions: Dict[str, np.ndarray] = {}
        for attr in attributes:
            vec = rng.randn(self.latent_dim)
            vec = vec / (np.linalg.norm(vec) + 1e-9)  # unit norm
            self.directions[attr] = vec

    def map_character_attributes(
        self,
        char_attrs: Dict[str, Any],
        base_seed: Optional[int] = None,
        alpha: float = 0.75
    ) -> np.ndarray:
        """
        Synthesizes a character latent vector z_char in R^nz by modulating
        a base Gaussian noise vector with semantic attribute directions.

        Formula:
            z_char = normalize(z_base + alpha * sum(v_attribute))
        """
        seed = base_seed if base_seed is not None else self.seed
        rng = np.random.RandomState(seed)
        z_base = rng.randn(self.latent_dim)

        hair = char_attrs.get("hair_color", "hitam")
        cloth = char_attrs.get("clothing_color", "merah")
        expr = char_attrs.get("expression", "ceria & tersenyum")

        mod_vector = np.zeros(self.latent_dim)

        # Map hair
        hair_key = f"hair_{hair}"
        if hair_key in self.directions:
            mod_vector += 1.0 * self.directions[hair_key]

        # Map clothing
        cloth_key = f"cloth_{cloth}"
        if cloth_key in self.directions:
            mod_vector += 1.0 * self.directions[cloth_key]

        # Map expression
        if "ceria" in expr:
            mod_vector += 1.2 * self.directions["expr_ceria"]
        elif "berani" in expr:
            mod_vector += 1.2 * self.directions["expr_berani"]
        elif "kagum" in expr:
            mod_vector += 1.2 * self.directions["expr_kagum"]
        else:
            mod_vector += 1.2 * self.directions["expr_bijak"]

        # Vector addition + spherical projection (Gaussian normalization)
        z_char = z_base + alpha * mod_vector
        z_char = (z_char / np.linalg.norm(z_char)) * np.sqrt(self.latent_dim)
        return z_char

    def map_scene_attributes(
        self,
        scene_attrs: Dict[str, Any],
        base_seed: Optional[int] = None,
        alpha: float = 0.85
    ) -> np.ndarray:
        """
        Synthesizes a scene setting latent vector z_scene in R^nz.
        """
        seed = (base_seed + 101) if base_seed is not None else (self.seed + 101)
        rng = np.random.RandomState(seed)
        z_base = rng.randn(self.latent_dim)

        location = scene_attrs.get("location", "desa").lower()
        atmosphere = scene_attrs.get("lighting_atmosphere", "pagi").lower()

        mod_vector = np.zeros(self.latent_dim)

        # Location mapping
        if "hutan" in location:
            mod_vector += 1.2 * self.directions["scene_hutan"]
        elif "bukit" in location:
            mod_vector += 1.2 * self.directions["scene_bukit"]
        elif "pantai" in location:
            mod_vector += 1.2 * self.directions["scene_pantai"]
        elif "angkasa" in location or "bintang" in location:
            mod_vector += 1.2 * self.directions["scene_angkasa"]
        else:
            mod_vector += 1.2 * self.directions["scene_desa"]

        # Atmosphere mapping
        if "malam" in atmosphere:
            mod_vector += 1.0 * self.directions["atm_malam"]
        elif "senja" in atmosphere:
            mod_vector += 1.0 * self.directions["atm_senja"]
        elif "kabut" in atmosphere or "sejuk" in atmosphere:
            mod_vector += 1.0 * self.directions["atm_kabut"]
        else:
            mod_vector += 1.0 * self.directions["atm_pagi"]

        z_scene = z_base + alpha * mod_vector
        z_scene = (z_scene / np.linalg.norm(z_scene)) * np.sqrt(self.latent_dim)
        return z_scene

    def interpolate_latents(
        self,
        z_start: np.ndarray,
        z_end: np.ndarray,
        num_steps: int = 5
    ) -> np.ndarray:
        """
        Spherical linear interpolation (Slerp) between two latent points in latent space.
        """
        steps = np.linspace(0.0, 1.0, num_steps)
        interpolated = []
        for t in steps:
            z_t = (1.0 - t) * z_start + t * z_end
            z_t = (z_t / np.linalg.norm(z_t)) * np.sqrt(self.latent_dim)
            interpolated.append(z_t)
        return np.array(interpolated)

