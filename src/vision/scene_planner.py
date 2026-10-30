"""
Scene Planner module.
Analyzes each story page's text along with the Character Registry to produce a comprehensive
Scene JSON including location, weather, characters present, individual actions, and props.
"""

import re
from typing import Dict, Any, List, Optional
from src.story.character_registry import CharacterRegistry, Character


class ScenePlanner:
    """
    Translates page narrative text and character registry into an actionable visual scene specification.
    """

    LOCATION_PATTERNS = {
        "sungai": ("aliran sungai jernih dengan pepohonan willow", "sungai"),
        "hutan": ("hutan rimba hijau yang asri dan rindang", "hutan"),
        "bukit": ("hamparan perbukitan hijau berkabut lembut", "bukit"),
        "danau": ("tepian danau tenang memantulkan cahaya mentari", "danau"),
        "kebun": ("kebun bunga liar warna-warni dan semak rimbun", "kebun"),
        "desa": ("jalan setapak desa yang damai dan asri", "desa"),
        "rumah": ("halaman depan rumah klasik bercerobong hangat", "rumah"),
        "langit": ("angkasa biru dengan awan-awan seputih kapas", "langit"),
        "perahu": ("sungai tenang dengan perahu kayu kecil", "perahu")
    }

    ACTION_PATTERNS = {
        "mendayung": ("mendayung perahu kayu kecil di atas air tenang", "fokus dan riang"),
        "berlayar": ("berlayar menyeberangi riak air jernih", "bersemangat"),
        "membaca": ("membaca buku dongeng dengan senyuman lebar", "penasaran dan tenang"),
        "berjalan": ("melangkah perlahan mengamati alam sekitar", "santai dan gembira"),
        "berlari": ("berlari riang di antara rerumputan hijau", "penuh energi"),
        "melihat": ("menatap keindahan alam sekitar dengan takjub", "terpesona"),
        "menatap": ("memandang cakrawala penuh harapan", "teduh"),
        "bertemu": ("menyapa sahabat dengan lambaian tangan hangat", "ramah dan ceria"),
        "tersenyum": ("tersenyum manis menikmati suasana indah", "hangat dan bahagia"),
        "menemukan": ("menemukan hal istimewa di balik semak bunga", "terkejut gembira")
    }

    @classmethod
    def plan_scene_for_page(
        cls,
        page_idx: int,
        total_pages: int,
        page_text: str,
        character_reg: CharacterRegistry,
        world_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates a structured Scene JSON specification for a single page.
        """
        world = world_settings or {}
        text_lower = page_text.lower()
        uncertainty_notes = []

        # 1. Determine Location
        detected_location = None
        for key, (loc_desc, loc_id) in cls.LOCATION_PATTERNS.items():
            if key in text_lower:
                detected_location = loc_desc
                break
        if not detected_location:
            detected_location = world.get("location", "alam perbukitan hijau dan sungai yang tenang")
            uncertainty_notes.append("Lokasi spesifik tidak eksplisit di teks; menggunakan pengaturan latar dunia.")

        # 2. Time & Weather
        weather = world.get("weather", "cerah menyenangkan")
        if "hujan" in text_lower:
            weather = "hujan gerimis sejuk"
        elif "senja" in text_lower or "sore" in text_lower:
            weather = "senja keemasan temaram"
        elif "malam" in text_lower:
            weather = "malam bertabur bintang"

        # 3. Atmosphere
        atmosphere = world.get("atmosphere", "damai dan hangat")
        if "kejutan" in text_lower or "tiba-tiba" in text_lower:
            atmosphere = "penuh rasa ingin tahu dan mendebarkan"
        elif "bahagia" in text_lower or "gembira" in text_lower:
            atmosphere = "ceria dan penuh sukacita"

        # 4. Characters Present & Individual Actions
        main_char = character_reg.get_main_character()
        all_chars = character_reg.get_all_characters()
        characters_present = []

        # Detect action for main character
        main_action = "mengamati alam dengan pandangan berbinar"
        main_expression = main_char.default_expression if main_char else "ceria"

        for act_key, (act_desc, exp_desc) in cls.ACTION_PATTERNS.items():
            if act_key in text_lower:
                main_action = act_desc
                main_expression = exp_desc
                break

        if main_char:
            characters_present.append({
                "name": main_char.name,
                "role": "tokoh_utama",
                "character_type": main_char.character_type,
                "action": main_action,
                "expression": main_expression,
                "physical_attributes": {
                    "fur_or_skin": main_char.fur_or_skin,
                    "hair_style": main_char.hair_style,
                    "hair_color": main_char.hair_color,
                    "clothing": main_char.clothing,
                    "accessories": main_char.accessories,
                    "unique_trait": main_char.unique_trait
                }
            })

        # Check for supporting characters presence
        for sup in character_reg.get_supporting_characters():
            if sup.name.lower() in text_lower or page_idx in [2, 3]:
                characters_present.append({
                    "name": sup.name,
                    "role": "pendukung",
                    "character_type": sup.character_type,
                    "action": "menemani sahabat dengan setia",
                    "expression": sup.default_expression,
                    "physical_attributes": {
                        "fur_or_skin": sup.fur_or_skin,
                        "clothing": sup.clothing,
                        "accessories": sup.accessories,
                        "unique_trait": sup.unique_trait
                    }
                })

        # 5. Key Props
        key_props = []
        if "perahu" in text_lower:
            key_props.append("perahu kayu cokelat")
        if "buku" in text_lower:
            key_props.append("buku cerita bersampul kuning")
        if "bunga" in text_lower:
            key_props.append("buket bunga liar")
        if "topi" in text_lower or "beret" in text_lower:
            key_props.append("topi pelindung kepala")
        if "kereta" in text_lower or "mobil" in text_lower:
            key_props.append("kereta antik merah")
        if not key_props:
            key_props.append("elemen alam sekitar (batu sungai & dedaunan)")

        return {
            "page": page_idx,
            "total_pages": total_pages,
            "location": detected_location,
            "time_and_weather": weather,
            "atmosphere": atmosphere,
            "characters_present": characters_present,
            "key_props": key_props,
            "uncertainty_notes": uncertainty_notes,
            "visual_style": world.get("visual_style", "Watercolor Storybook Ilustrasi Halus")
        }

