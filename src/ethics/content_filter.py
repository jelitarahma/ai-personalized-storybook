"""
Child-Friendly Content Filtering and Safety Moderation Module.
Scans prompts and model generations for violence, profanity, dark themes, and fear triggers.
Applies automated child-safe text sanitization.
References: Lecture Notes 10 (Responsible AI & Content Moderation).
"""

import re
from typing import Dict, Any, List, Tuple


# Inappropriate or scary word filters for young readers
UNSAFE_TERMS_LEXICON = [
    # Violence & weapons
    "bunuh", "darah", "senjata", "tembak", "tikam", "mati konyol", "racun", "mayat",
    # Intense horror & nightmare triggers
    "hantu seram", "setan", "monster jahat mematikan", "kutukan berdarah", "neraka",
    # Profanity & vulgarity
    "bodoh", "tolol", "anjing", "babi", "brengsek", "sialan",
    # Adult/Inappropriate
    "narkoba", "rokok", "miras", "mabuk"
]

SAFE_CHILD_REPLACEMENTS = {
    "bunuh": "kalahkan",
    "darah": "luka kecil",
    "mati": "pergi",
    "monster jahat mematikan": "makhluk misterius",
    "hantu seram": "bayangan lucu",
    "bodoh": "belum tahu",
    "tolol": "kurang teliti",
    "sialan": "aduhai",
    "racun": "ramuan pahit"
}


class ChildContentFilter:
    """
    Ensures that stories and prompts strictly follow G-rated, child-safe standards.
    """

    def __init__(self):
        self.unsafe_terms = UNSAFE_TERMS_LEXICON
        self.replacements = SAFE_CHILD_REPLACEMENTS

    def screen_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Screens text for presence of unsafe or scary terms.
        Returns: (is_safe, list_of_detected_violations)
        """
        text_lower = text.lower()
        violations = []
        for term in self.unsafe_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                violations.append(term)
        return len(violations) == 0, violations

    def sanitize_text(self, text: str) -> Tuple[str, bool, List[str]]:
        """
        Sanitizes text by replacing sensitive terms with child-safe pedagogical alternatives.
        Returns: (sanitized_text, was_modified, list_of_replaced_terms)
        """
        sanitized = text
        replaced = []
        for bad_term, safe_sub in self.replacements.items():
            pattern = re.compile(r'\b' + re.escape(bad_term) + r'\b', re.IGNORECASE)
            if pattern.search(sanitized):
                sanitized = pattern.sub(safe_sub, sanitized)
                replaced.append(bad_term)

        was_modified = len(replaced) > 0
        return sanitized, was_modified, replaced

