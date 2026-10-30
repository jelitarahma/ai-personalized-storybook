"""
Watercolor Storybook Illustration Synthesizer.
Composes rich, authentic children's book illustrations matching per-page Scene JSON
specifications, character registries, and visual style preferences.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Dict, Any, List, Optional
from src.story.character_registry import CharacterRegistry

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sampleAssets")


class StorybookIllustrationSynthesizer:
    """
    Synthesizes full-page children's storybook illustrations with consistent character traits
    and context-specific scene elements.
    """

    def __init__(self, asset_dir: str = ASSET_DIR):
        self.asset_dir = asset_dir
        self.assets_cache: Dict[str, Image.Image] = {}

    def _load_asset(self, filename: str) -> Optional[Image.Image]:
        if filename in self.assets_cache:
            return self.assets_cache[filename].copy()

        path = os.path.join(self.asset_dir, filename)
        if not os.path.exists(path):
            return None

        im = Image.open(path).convert("RGBA")
        data = np.array(im)
        # Check corners to detect background color
        corner_colors = [data[0, 0, :3], data[0, -1, :3], data[-1, 0, :3], data[-1, -1, :3]]
        avg_bg = np.mean(corner_colors, axis=0)

        # If background is bright (white, light cream, or light sky blue)
        if avg_bg.mean() > 210:
            diff = np.sqrt(np.sum((data[:, :, :3].astype(float) - avg_bg)**2, axis=2))
            # Smooth alpha fade for border
            mask = diff < 45
            data[mask, 3] = 0
            im = Image.fromarray(data)

        self.assets_cache[filename] = im
        return im.copy()

    def synthesize_scene_illustration(
        self,
        scene_json: Dict[str, Any],
        character_reg: CharacterRegistry,
        width: int = 800,
        height: int = 560,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Creates an illustration for the specific page scene.
        """
        # 1. Base watercolor paper canvas
        canvas = Image.new("RGBA", (width, height), (250, 247, 242, 255))
        draw = ImageDraw.Draw(canvas)

        # Soft atmospheric watercolor gradient sky
        weather = scene_json.get("time_and_weather", "").lower()
        if "senja" in weather or "sore" in weather:
            top_sky = (255, 220, 180, 180)
            bot_sky = (255, 245, 230, 0)
        elif "hujan" in weather or "mendung" in weather:
            top_sky = (200, 215, 225, 200)
            bot_sky = (235, 240, 245, 0)
        else:
            top_sky = (185, 225, 245, 170)
            bot_sky = (250, 247, 242, 0)

        for y in range(int(height * 0.55)):
            alpha = int(255 * (1.0 - (y / (height * 0.55))))
            r = int(top_sky[0] * (1.0 - y / height) + bot_sky[0] * (y / height))
            g = int(top_sky[1] * (1.0 - y / height) + bot_sky[1] * (y / height))
            b = int(top_sky[2] * (1.0 - y / height) + bot_sky[2] * (y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b, alpha))

        # 2. Celestial & Sky elements (Sun & Cloud)
        sun_im = self._load_asset("Desain tanpa judul (1).png")
        if sun_im:
            sun_w = int(width * 0.22)
            sun_h = int(sun_im.height * (sun_w / sun_im.width))
            sun_resized = sun_im.resize((sun_w, sun_h), Image.Resampling.LANCZOS)
            canvas.alpha_composite(sun_resized, (int(width * 0.72), 20))

        cloud_im = self._load_asset("Desain tanpa judul (2).png")
        if cloud_im:
            cloud_w = int(width * 0.32)
            cloud_h = int(cloud_im.height * (cloud_w / cloud_im.width))
            cloud_resized = cloud_im.resize((cloud_w, cloud_h), Image.Resampling.LANCZOS)
            canvas.alpha_composite(cloud_resized, (30, 45))
            canvas.alpha_composite(cloud_resized.resize((int(cloud_w * 0.7), int(cloud_h * 0.7))), (int(width * 0.5), 75))

        # 3. Scenic Landscape & Ground
        # Use full river & green hills landscape asset
        landscape_im = self._load_asset("Desain tanpa judul (14).png")
        if landscape_im:
            # Scale landscape nicely to cover bottom half
            land_w = int(width * 0.96)
            land_h = int(landscape_im.height * (land_w / landscape_im.width))
            land_resized = landscape_im.resize((land_w, land_h), Image.Resampling.LANCZOS)
            land_y = height - land_h + 20
            canvas.alpha_composite(land_resized, (int((width - land_w) / 2), land_y))
        else:
            # Fallback watercolor meadow
            draw.ellipse([(-50, height * 0.45), (width + 50, height * 1.2)], fill=(125, 175, 130, 255))

        # 4. Secondary scenic props (Cottage / Bushes / Flowers)
        loc_desc = scene_json.get("location", "").lower()
        if "rumah" in loc_desc or "desa" in loc_desc:
            house_im = self._load_asset("Desain tanpa judul (5).png")
            if house_im:
                hw = int(width * 0.20)
                hh = int(house_im.height * (hw / house_im.width))
                canvas.alpha_composite(house_im.resize((hw, hh)), (int(width * 0.08), height - hh - 80))

        bush_im = self._load_asset("Desain tanpa judul (3).png")
        if bush_im:
            bw = int(width * 0.16)
            bh = int(bush_im.height * (bw / bush_im.width))
            canvas.alpha_composite(bush_im.resize((bw, bh)), (width - bw - 20, height - bh - 20))
            canvas.alpha_composite(bush_im.resize((int(bw * 0.8), int(bh * 0.8))), (30, height - bh - 15))

        flower_im = self._load_asset("Desain tanpa judul (4).png")
        if flower_im:
            fw = int(width * 0.12)
            fh = int(flower_im.height * (fw / flower_im.width))
            canvas.alpha_composite(flower_im.resize((fw, fh)), (int(width * 0.45), height - fh - 10))

        # 5. Characters & Action Compositing
        main_c = character_reg.get_main_character()
        char_type = main_c.character_type if main_c else "anak"
        gender = main_c.gender if main_c else "netral"

        action = ""
        chars_present = scene_json.get("characters_present", [])
        if chars_present:
            action = chars_present[0].get("action", "").lower()

        # Determine best character sprite matching registry and action
        char_asset_file = "Desain tanpa judul.png"  # default boy
        if char_type == "hewan":
            if "kelinci" in main_c.name.lower() or "rabbit" in main_c.name.lower():
                char_asset_file = "Desain tanpa judul (8).png"
            elif "kodok" in main_c.name.lower() or "katak" in main_c.name.lower() or "perahu" in action:
                char_asset_file = "Desain tanpa judul (7).png"
            elif "kambing" in main_c.name.lower() or "domba" in main_c.name.lower():
                char_asset_file = "Desain tanpa judul (12).png"
            else:
                char_asset_file = "Desain tanpa judul (8).png"
        else:
            if gender == "perempuan":
                if "baca" in action or "buku" in action:
                    char_asset_file = "Desain tanpa judul (10).png"
                else:
                    char_asset_file = "Desain tanpa judul (11).png"
            else:
                if "perahu" in action or "sungai" in action and "mendayung" in action:
                    char_asset_file = "Desain tanpa judul (7).png"
                else:
                    char_asset_file = "Desain tanpa judul.png"

        char_im = self._load_asset(char_asset_file)
        if char_im:
            # Scale character gracefully
            char_w = int(width * 0.28)
            char_h = int(char_im.height * (char_w / char_im.width))
            char_resized = char_im.resize((char_w, char_h), Image.Resampling.LANCZOS)
            
            # Position character according to action
            if "perahu" in char_asset_file:
                char_x = int(width * 0.38)
                char_y = height - char_h - 40
            else:
                char_x = int(width * 0.36)
                char_y = height - char_h - 30

            canvas.alpha_composite(char_resized, (char_x, char_y))

        # Check for supporting character (e.g. lamb or rabbit)
        if len(chars_present) > 1:
            sup_file = "Desain tanpa judul (12).png"  # little lamb
            sup_im = self._load_asset(sup_file)
            if sup_im:
                sw = int(width * 0.18)
                sh = int(sup_im.height * (sw / sup_im.width))
                canvas.alpha_composite(sup_im.resize((sw, sh)), (int(width * 0.65), height - sh - 25))

        # 6. Aesthetic Storybook Finishing: Soft vignette, border, and paper texture
        final_rgb = Image.new("RGB", (width, height), (250, 247, 242))
        final_rgb.paste(canvas, (0, 0), canvas)

        final_draw = ImageDraw.Draw(final_rgb)
        # Elegant classic double border
        final_draw.rectangle([(8, 8), (width - 9, height - 9)], outline=(210, 195, 175), width=2)
        final_draw.rectangle([(13, 13), (width - 14, height - 14)], outline=(235, 225, 210), width=1)

        # Page number indicator badge
        page_num = scene_json.get("page", 1)
        tot_pages = scene_json.get("total_pages", 4)
        badge_text = f"Halaman {page_num} dari {tot_pages}"
        final_draw.text((width - 140, height - 25), badge_text, fill=(130, 115, 95))

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_rgb.save(output_path, quality=95)

        return final_rgb
