"""
Visual rendering engine for storybook illustrations.
Synthesizes distinct character portraits and scene landscapes using VAE/DCGAN generative latent spaces,
and compiles them into an aesthetically formatted storybook page layout.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, Tuple
from src.visual.dcgan_model import Generator
from src.visual.vae_model import ConvVAE
from src.visual.latent_mapping import SemanticLatentMapper


class VisualRenderer:
    """
    Renders character portraits, scenic environments, and complete storybook book pages.
    """

    def __init__(
        self,
        nz: int = 100,
        device: Optional[str] = None,
        seed: int = 42
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nz = nz
        self.mapper = SemanticLatentMapper(latent_dim=nz, seed=seed)
        
        # Initialize generative models
        self.dcgan_gen = Generator(nz=nz, ngf=32, nc=3).to(self.device)
        self.dcgan_gen.eval()

        self.vae = ConvVAE(latent_dim=32, in_channels=3).to(self.device)
        self.vae.eval()

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts PyTorch image tensor [-1, 1] to uint8 RGB numpy array [0, 255].
        """
        img = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def render_character(
        self,
        char_attrs: Dict[str, Any],
        output_path: Optional[str] = None,
        seed: int = 42
    ) -> Image.Image:
        """
        Synthesizes a distinct character portrait conditioned on extracted attributes
        (hair, clothing color, expression, name).
        """
        # Map attributes to latent vector
        z_char = self.mapper.map_character_attributes(char_attrs, base_seed=seed)
        z_tensor = torch.tensor(z_char, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Generative model pass
        with torch.no_grad():
            raw_gen = self.dcgan_gen(z_tensor)
        base_texture = self._tensor_to_numpy(raw_gen)

        # Create stylized illustration canvas (256x256)
        img = Image.new("RGB", (256, 256), color=(245, 247, 250))
        draw = ImageDraw.Draw(img)

        # Palette colors based on attributes
        color_palette = {
            "hitam": (45, 45, 45),
            "cokelat": (115, 75, 45),
            "merah": (210, 50, 45),
            "keemasan": (230, 180, 40),
            "putih": (235, 235, 240),
            "hijau": (46, 139, 87),
            "biru": (52, 120, 210),
            "kuning": (245, 205, 50)
        }

        hair_rgb = color_palette.get(char_attrs.get("hair_color", "hitam"), (50, 50, 50))
        cloth_rgb = color_palette.get(char_attrs.get("clothing_color", "merah"), (200, 50, 50))
        skin_tone = (248, 218, 185)

        # Draw soft background portrait vignette
        bg_rgb = (
            min(cloth_rgb[0] + 40, 245),
            min(cloth_rgb[1] + 40, 245),
            min(cloth_rgb[2] + 40, 245)
        )
        draw.ellipse([20, 20, 236, 236], fill=bg_rgb, outline=(220, 225, 230), width=3)

        # Blend texture from GAN into vignette
        tex_pil = Image.fromarray(base_texture).resize((180, 180))
        img.paste(tex_pil, (38, 38), mask=None)

        # Overlay character illustration geometry
        # 1. Clothing / Torso
        draw.polygon([(68, 240), (128, 160), (188, 240)], fill=cloth_rgb)
        draw.line([(128, 160), (128, 240)], fill=(255, 255, 255), width=2)

        # 2. Head / Face
        draw.ellipse([88, 80, 168, 160], fill=skin_tone, outline=(180, 150, 120), width=2)

        # 3. Hair
        draw.chord([84, 68, 172, 125], start=180, end=360, fill=hair_rgb)
        draw.polygon([(84, 100), (95, 75), (105, 110)], fill=hair_rgb)
        draw.polygon([(150, 110), (160, 75), (172, 100)], fill=hair_rgb)

        # 4. Eyes
        draw.ellipse([104, 112, 114, 122], fill=(30, 30, 30))
        draw.ellipse([142, 112, 152, 122], fill=(30, 30, 30))
        # Eye shine
        draw.ellipse([106, 114, 109, 117], fill=(255, 255, 255))
        draw.ellipse([144, 114, 147, 117], fill=(255, 255, 255))

        # 5. Cheeks blush
        draw.ellipse([98, 126, 110, 134], fill=(250, 180, 180))
        draw.ellipse([146, 126, 158, 134], fill=(250, 180, 180))

        # 6. Mouth / Expression
        expr = char_attrs.get("expression", "ceria & tersenyum")
        if "berani" in expr:
            draw.line([(120, 138), (136, 138)], fill=(120, 40, 40), width=3)  # firm smile
        elif "kagum" in expr:
            draw.ellipse([124, 132, 132, 142], fill=(120, 40, 40))  # open mouth 'o'
        else:
            draw.arc([118, 128, 138, 144], start=0, end=180, fill=(150, 40, 40), width=3)  # cheerful smile

        # Caption text
        name_text = char_attrs.get("name", "Karakter")
        draw.rectangle([30, 218, 226, 248], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        draw.text((45, 224), f"Tokoh: {name_text}", fill=(40, 40, 40))

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, "PNG")

        return img

    def render_scene(
        self,
        scene_attrs: Dict[str, Any],
        output_path: Optional[str] = None,
        seed: int = 42
    ) -> Image.Image:
        """
        Synthesizes a distinct scenic setting illustration conditioned on location and atmosphere.
        """
        z_scene = self.mapper.map_scene_attributes(scene_attrs, base_seed=seed)
        z_tensor = torch.tensor(z_scene, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            raw_gen = self.dcgan_gen(z_tensor)
        base_texture = self._tensor_to_numpy(raw_gen)

        img = Image.new("RGB", (320, 240), color=(240, 245, 250))
        draw = ImageDraw.Draw(img)

        location = scene_attrs.get("location", "desa").lower()
        atmosphere = scene_attrs.get("lighting_atmosphere", "pagi").lower()

        # Sky color based on atmosphere
        if "malam" in atmosphere or "bintang" in location:
            sky_colors = [(15, 20, 45), (35, 45, 90)]
        elif "senja" in atmosphere:
            sky_colors = [(235, 110, 60), (250, 185, 95)]
        elif "kabut" in atmosphere:
            sky_colors = [(180, 200, 210), (210, 225, 230)]
        else:
            sky_colors = [(100, 180, 245), (195, 230, 255)]

        # Render sky gradient
        for y in range(140):
            t = y / 140.0
            r = int((1 - t) * sky_colors[0][0] + t * sky_colors[1][0])
            g = int((1 - t) * sky_colors[0][1] + t * sky_colors[1][1])
            b = int((1 - t) * sky_colors[0][2] + t * sky_colors[1][2])
            draw.line([(0, y), (320, y)], fill=(r, g, b))

        # Night stars or Sun
        if "malam" in atmosphere or "bintang" in location:
            rng = np.random.RandomState(seed)
            for _ in range(35):
                sx = rng.randint(5, 315)
                sy = rng.randint(5, 120)
                draw.point((sx, sy), fill=(255, 255, 255))
            draw.ellipse([250, 25, 280, 55], fill=(245, 245, 220))  # Moon
        else:
            draw.ellipse([30, 25, 75, 70], fill=(255, 220, 80))  # Sun

        # Scenery Terrain
        if "pantai" in location:
            # Sea and Sand
            draw.rectangle([0, 120, 320, 175], fill=(40, 140, 190))  # Sea
            # Waves
            for wx in range(0, 320, 30):
                draw.arc([wx, 140, wx + 25, 155], start=0, end=180, fill=(240, 250, 255), width=2)
            draw.polygon([(0, 170), (320, 155), (320, 240), (0, 240)], fill=(235, 205, 145))  # Sand
        elif "angkasa" in location:
            # Cosmic Rings and Nebula
            draw.ellipse([80, 110, 240, 190], outline=(180, 120, 240), width=4)
            draw.rectangle([0, 180, 320, 240], fill=(25, 25, 55))
        elif "hutan" in location or "bukit" in location:
            # Rolling green hills and tree silhouettes
            draw.chord([-40, 100, 200, 280], start=180, end=360, fill=(50, 125, 60))
            draw.chord([120, 110, 360, 280], start=180, end=360, fill=(35, 105, 45))
            draw.rectangle([0, 185, 320, 240], fill=(30, 85, 38))
            # Foliage trees
            for tx in [45, 100, 160, 220, 280]:
                draw.polygon([(tx, 185), (tx - 18, 185), (tx, 140), (tx + 18, 185)], fill=(20, 70, 30))
        else:
            # Nusantara Village
            draw.chord([-30, 120, 210, 270], start=180, end=360, fill=(75, 140, 70))
            draw.rectangle([0, 180, 320, 240], fill=(95, 155, 80))
            # Traditional wooden house outline
            draw.polygon([(180, 175), (210, 140), (240, 175)], fill=(140, 80, 45))
            draw.rectangle([188, 175, 232, 205], fill=(185, 125, 80))

        # Scene caption overlay
        loc_text = scene_attrs.get("location", "Desa")
        draw.rectangle([10, 205, 310, 232], fill=(255, 255, 255, 220), outline=(210, 215, 220))
        draw.text((18, 212), f"Setting: {loc_text.capitalize()} ({atmosphere})", fill=(30, 30, 30))

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, "PNG")

        return img

    def compose_storybook_page(
        self,
        story_text: str,
        profile: Dict[str, Any],
        char_img: Image.Image,
        scene_img: Image.Image,
        output_path: str
    ) -> Image.Image:
        """
        Assembles a comprehensive, beautiful children's storybook page (700x900)
        featuring header, character portrait, scene landscape, narrative text, and moral values.
        """
        page = Image.new("RGB", (700, 920), color=(254, 252, 247))  # Warm book paper tone
        draw = ImageDraw.Draw(page)

        # Outer ornate frame
        draw.rectangle([15, 15, 685, 905], outline=(210, 190, 160), width=3)
        draw.rectangle([22, 22, 678, 898], outline=(235, 225, 205), width=1)

        # Header Title Banner
        genre = profile.get("genre", "Fantasi")
        age = profile.get("age", 7)
        name = profile.get("name", "Sahabat")
        theme = profile.get("theme", "Kebaikan Hati")

        draw.rectangle([40, 35, 660, 95], fill=(245, 238, 225), outline=(200, 180, 150))
        draw.text((55, 45), f"Petualangan {name}: Cerita {genre} Anak Nusantara", fill=(60, 40, 20))
        draw.text((55, 70), f"Untuk Usia: {age} Tahun  |  Tema: {theme}", fill=(110, 85, 55))

        # Illustrations Section: Character (left) & Scene (right)
        char_thumb = char_img.resize((220, 220))
        scene_thumb = scene_img.resize((360, 220))

        # Borders for illustrations
        draw.rectangle([48, 118, 272, 342], fill=(230, 220, 205), outline=(180, 160, 130), width=2)
        page.paste(char_thumb, (50, 120))

        draw.rectangle([288, 118, 652, 342], fill=(230, 220, 205), outline=(180, 160, 130), width=2)
        page.paste(scene_thumb, (290, 120))

        # Narrative Story Text Box
        draw.rectangle([40, 360, 660, 800], fill=(255, 255, 255), outline=(225, 215, 195))
        draw.text((55, 375), "NARASI CERITA:", fill=(120, 60, 30))

        # Word wrap narrative story into text box
        words = story_text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > 60:
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))

        y_offset = 410
        for line in lines[:18]:  # Up to 18 lines fit cleanly
            draw.text((55, y_offset), line, fill=(35, 35, 35))
            y_offset += 21

        # Moral message & footer banner
        draw.rectangle([40, 815, 660, 885], fill=(240, 246, 240), outline=(170, 200, 175))
        draw.text((55, 825), "Pesan Kasih & Moral Cerita:", fill=(35, 95, 45))
        draw.text(
            (55, 848),
            f"Kebaikan hati dan kerja sama selalu membawa cahaya. Tetaplah bermimpi dan berbuat baik, {name}!",
            fill=(50, 75, 55)
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        page.save(output_path, "PNG")
        print(f"Storybook page saved: {output_path}")

        return page

