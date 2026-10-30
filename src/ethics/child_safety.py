"""
Child Data Privacy & Safety Module.
Implements data minimization, pseudonymization, and privacy-preserving in-memory processing
aligned with COPPA and GDPR-Kids guidelines.
References: Lecture Notes 10 (Ethical and Societal Implications) & Solanki & Khublani (2024).
"""

import hashlib
import re
from typing import Dict, Any, Tuple


DISALLOWED_PII_PATTERNS = [
    r'\b\d{10,16}\b',             # Phone numbers or ID numbers
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
    r'\bjalan\s+[a-z0-9\s]+\b',   # Street addresses
    r'\brt\s*\d+\s*/\s*rw\s*\d+\b' # Indonesian RT/RW residence addresses
]

CHILD_FRIENDLY_PSEUDONYMS = [
    "Petualang Cilik", "Bintang Kejora", "Pelangi Senja", "Garuda Muda",
    "Sahabat Hutan", "Pena Ajaib", "Angin Sejuk", "Embun Pagi"
]


class ChildDataProtector:
    """
    Manages privacy protections for children's data:
    - Validates data minimization (rejection of extraneous personal data)
    - Generates cryptographic or friendly pseudonyms
    - Sanitizes logs so real PII is never permanently stored
    """

    def __init__(self, salt: str = "ai_storybook_salt_2026"):
        self.salt = salt

    def validate_and_minimize(self, raw_input: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Ensures input adheres to data minimization: only name, age, genre, theme are accepted.
        Scans and rejects any accidental PII (phone numbers, addresses, emails).
        """
        combined_text = " ".join(str(v) for v in raw_input.values()).lower()
        for pattern in DISALLOWED_PII_PATTERNS:
            if re.search(pattern, combined_text):
                return False, "Input mengandung data sensitif pribadi (PII) yang tidak diizinkan.", {}

        # Allow only safe, minimized profile attributes
        minimized = {
            "name": str(raw_input.get("name", "Sahabat Cilik")).strip()[:30],
            "age": int(raw_input.get("age", 7)),
            "genre": str(raw_input.get("genre", "Fantasi")).strip(),
            "theme": str(raw_input.get("theme", "Kebaikan Hati")).strip(),
            "cultural_background": str(raw_input.get("cultural_background", "Nusantara")).strip(),
            "character_preference": str(raw_input.get("character_preference", "anak ceria")).strip()
        }

        # Age bounds checking
        if minimized["age"] < 3 or minimized["age"] > 16:
            return False, f"Usia {minimized['age']} di luar rentang sasaran buku cerita anak (3-16 tahun).", {}

        return True, "Validasi privasi lolos (Data Minimization OK).", minimized

    def pseudonymize_profile(self, profile: Dict[str, Any], use_cute_alias: bool = True) -> Tuple[Dict[str, Any], str]:
        """
        Replaces the child's real name with a privacy-safe pseudonym for public display or logs.
        """
        real_name = profile.get("name", "Anak")
        # Deterministic hash token
        hash_digest = hashlib.sha256(f"{real_name}_{self.salt}".encode()).hexdigest()[:8]

        if use_cute_alias:
            idx = int(hash_digest, 16) % len(CHILD_FRIENDLY_PSEUDONYMS)
            alias = f"{CHILD_FRIENDLY_PSEUDONYMS[idx]}_{hash_digest[:4]}"
        else:
            alias = f"AnonChild_{hash_digest}"

        sanitized_profile = dict(profile)
        sanitized_profile["original_name_masked"] = f"{real_name[0]}***"
        sanitized_profile["name"] = alias
        sanitized_profile["is_pseudonymized"] = True

        return sanitized_profile, alias

    def sanitize_log_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips identifiable attributes from internal debug logs.
        """
        safe_copy = dict(record)
        if "profile" in safe_copy and "name" in safe_copy["profile"]:
            safe_copy["profile"] = dict(safe_copy["profile"])
            safe_copy["profile"]["name"] = "[ANONYMIZED_CHILD_NAME]"
        return safe_copy

