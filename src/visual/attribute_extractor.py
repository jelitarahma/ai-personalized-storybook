"""
Text-to-Visual Attribute Extraction Module.
Parses generated story narratives to extract visual attributes for main character and scene setting.
Outputs structured JSON with robust defaults and ambiguity logging.
"""

import re
from typing import Dict, Any, List, Tuple


# Lexicons for attribute parsing in Indonesian
HAIR_COLORS = {
    "hitam": ["hitam", "gelap", "kelam"],
    "cokelat": ["cokelat", "pirang gelap", "kayu"],
    "merah": ["merah", "tembaga", "jingga"],
    "keemasan": ["emas", "keemasan", "kuning terang"],
    "putih": ["putih", "perak", "kelabu"]
}

CLOTHING_COLORS = {
    "hijau": ["hijau", "zamrud", "daun"],
    "biru": ["biru", "laut", "langit", "aqua"],
    "merah": ["merah", "marun", "crimson"],
    "kuning": ["kuning", "emas", "cerah"],
    "cokelat": ["cokelat", "tanah", "krem"],
    "putih": ["putih", "bersih", "salju"]
}

CLOTHING_STYLES = {
    "rompi petualang": ["rompi", "ransel", "bot", "petualang"],
    "jubah pahlawan": ["jubah", "selendang", "mahkota", "pahlawan", "sayap"],
    "pakaian tradisional nusantara": ["batik", "sarung", "baju adat", "tenun", "songket"],
    "kaos santai": ["kaos", "baju santai", "celana pendek", "sepatu"]
}

EXPRESSIONS = {
    "ceria & tersenyum": ["senyum", "gembira", "tertawa", "ceria", "bahagia", "suka"],
    "berani & fokus": ["berani", "mantap", "gagah", "waspada", "tegas", "percaya diri"],
    "kagum & penasaran": ["kagum", "berbinar", "penasaran", "terpana", "heran", "ingin tahu"],
    "bijaksana & tenang": ["bijak", "tenang", "lembut", "damai", "teduh"]
}

LOCATIONS = {
    "hutan tropis nusantara": ["hutan", "rimba", "pepohonan", "belantara", "meranti", "kanopi"],
    "bukit hijau berbunga": ["bukit", "gunung", "lereng", "kebun", "taman", "lembah"],
    "pantai & pesisir samudra": ["pantai", "laut", "ombak", "pasir", "dermaga", "sanur"],
    "desa nusantara damai": ["desa", "kampung", "rumah gadang", "los pasar", "beranda"],
    "angkasa luar & bintang": ["bintang", "galaksi", "luar angkasa", "langit malam", "orbit", "komet"]
}

ATMOSPHERES = {
    "mentari pagi cerah": ["pagi", "cerah", "sinar", "hangat", "terang benderang"],
    "malam bertabur bintang": ["malam", "bintang", "bulan", "gelap", "konstelasi"],
    "senja keemasan": ["senja", "sore", "jingga", "matahari terbenam"],
    "kabut magis & sejuk": ["kabut", "sejuk", "rintik", "hujan", "embun"]
}


def extract_visual_attributes(
    story_text: str,
    profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extracts structured character and scene attributes from the narrative,
    falling back to profile-derived defaults when text does not explicitly mention them.

    Returns:
        dict: {
            'character': { 'name', 'hair_color', 'clothing_color', 'clothing_style', 'expression' },
            'scene': { 'location', 'background_elements', 'lighting_atmosphere' },
            'detection_log': { 'detected_count', 'defaulted_count', 'notes': [] }
        }
    """
    story_lower = story_text.lower()
    notes: List[str] = []
    detected_count = 0
    defaulted_count = 0

    # 1. Character Name
    char_name = profile.get("name", "Sahabat Cilik")

    # 2. Hair Color
    detected_hair = None
    for color, syns in HAIR_COLORS.items():
        if any(f"rambut {s}" in story_lower or f"berambut {s}" in story_lower or f"sisik {s}" in story_lower or f"bulu {s}" in story_lower for s in syns):
            detected_hair = color
            detected_count += 1
            break
    if not detected_hair:
        detected_hair = "hitam"  # standard natural Indonesian default
        defaulted_count += 1
        notes.append("Warna rambut tidak terdeteksi eksplisit; menggunakan default 'hitam'.")

    # 3. Clothing Color
    detected_clothing_color = None
    for color, syns in CLOTHING_COLORS.items():
        if any(f"baju {s}" in story_lower or f"rompi {s}" in story_lower or f"pakaian {s}" in story_lower or f"kain {s}" in story_lower for s in syns):
            detected_clothing_color = color
            detected_count += 1
            break
    if not detected_clothing_color:
        genre = profile.get("genre", "Fantasi").lower()
        if "fabel" in genre:
            detected_clothing_color = "hijau"
        elif "sains" in genre:
            detected_clothing_color = "biru"
        elif "petualang" in genre:
            detected_clothing_color = "cokelat"
        else:
            detected_clothing_color = "merah"
        defaulted_count += 1
        notes.append(f"Warna pakaian tidak terdeteksi di teks; menggunakan default genre '{detected_clothing_color}'.")

    # 4. Clothing Style
    detected_clothing_style = None
    for style, syns in CLOTHING_STYLES.items():
        if any(s in story_lower for s in syns):
            detected_clothing_style = style
            detected_count += 1
            break
    if not detected_clothing_style:
        detected_clothing_style = "rompi petualang"
        defaulted_count += 1
        notes.append("Gaya pakaian tidak terdeteksi; menggunakan default 'rompi petualang'.")

    # 5. Expression / Emotion
    detected_expression = None
    for expr, syns in EXPRESSIONS.items():
        if any(s in story_lower for s in syns):
            detected_expression = expr
            detected_count += 1
            break
    if not detected_expression:
        detected_expression = "ceria & tersenyum"
        defaulted_count += 1
        notes.append("Ekspresi tidak terdeteksi; menggunakan default ramah anak 'ceria & tersenyum'.")

    # 6. Scene Location
    detected_location = None
    for loc, syns in LOCATIONS.items():
        if any(s in story_lower for s in syns):
            detected_location = loc
            detected_count += 1
            break
    if not detected_location:
        detected_location = "desa nusantara damai"
        defaulted_count += 1
        notes.append("Lokasi adegan tidak terdeteksi; menggunakan default 'desa nusantara damai'.")

    # 7. Scene Atmosphere / Lighting
    detected_atmosphere = None
    for atm, syns in ATMOSPHERES.items():
        if any(s in story_lower for s in syns):
            detected_atmosphere = atm
            detected_count += 1
            break
    if not detected_atmosphere:
        detected_atmosphere = "mentari pagi cerah"
        defaulted_count += 1
        notes.append("Suasana pencahayaan tidak terdeteksi; menggunakan default 'mentari pagi cerah'.")

    # Background elements
    elements_map = {
        "hutan tropis nusantara": ["kanopi hijau lebat", "sulur tanaman hutan", "aliran air jernih"],
        "bukit hijau berbunga": ["lereng bukit hijau", "bunga warna-warni", "awan putih lembut"],
        "pantai & pesisir samudra": ["pasir pantai keemasan", "deburan ombak biru", "pohon kelapa nyiur"],
        "desa nusantara damai": ["rumah adat beratap kayu", "jalan setapak rindang", "pekarangan asri"],
        "angkasa luar & bintang": ["gugusan galaksi spiral", "bintang kejora berkilau", "langit biru kosmik"]
    }
    bg_elements = elements_map.get(detected_location, ["pemandangan alam asri", "langit cerah"])

    result = {
        "character": {
            "name": char_name,
            "hair_color": detected_hair,
            "clothing_color": detected_clothing_color,
            "clothing_style": detected_clothing_style,
            "expression": detected_expression
        },
        "scene": {
            "location": detected_location,
            "background_elements": bg_elements,
            "lighting_atmosphere": detected_atmosphere
        },
        "detection_log": {
            "detected_count": detected_count,
            "defaulted_count": defaulted_count,
            "detection_ratio": round(detected_count / (detected_count + defaulted_count), 2),
            "notes": notes
        }
    }
    return result

