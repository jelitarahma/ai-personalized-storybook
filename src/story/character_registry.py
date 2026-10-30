"""
Character Registry module.
Maintains physical consistency of main and supporting characters across all story pages.
Acts as the single source of truth for character visual traits.
"""

from typing import List, Dict, Any, Optional


class Character:
    """
    Representation of an individual character (main or supporting).
    """

    def __init__(
        self,
        name: str,
        role: str = "tokoh_utama",
        character_type: str = "anak",
        gender: str = "netral",
        age_approx: int = 7,
        fur_or_skin: str = "cerah",
        hair_style: str = "pendek rapi",
        hair_color: str = "hitam",
        clothing: str = "baju petualang nyaman warna cokelat muda",
        accessories: str = "tas selempang kecil",
        default_expression: str = "ceria dan penuh rasa ingin tahu",
        unique_trait: str = "senyuman hangat dan mata berbinar"
    ):
        self.name = name.strip() or "Sahabat Kecil"
        self.role = role  # "tokoh_utama" or "pendukung"
        self.character_type = character_type  # "anak", "hewan", "robot", "fantasi", "lainnya"
        self.gender = gender  # "laki-laki", "perempuan", "netral"
        self.age_approx = age_approx
        self.fur_or_skin = fur_or_skin
        self.hair_style = hair_style
        self.hair_color = hair_color
        self.clothing = clothing
        self.accessories = accessories
        self.default_expression = default_expression
        self.unique_trait = unique_trait

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "character_type": self.character_type,
            "gender": self.gender,
            "age_approx": self.age_approx,
            "fur_or_skin": self.fur_or_skin,
            "hair_style": self.hair_style,
            "hair_color": self.hair_color,
            "clothing": self.clothing,
            "accessories": self.accessories,
            "default_expression": self.default_expression,
            "unique_trait": self.unique_trait
        }

    def get_physical_description(self) -> str:
        """
        Returns a concise physical description to guarantee visual consistency.
        """
        if self.character_type == "hewan":
            return (
                f"{self.name} (hewan ramah berbulu {self.fur_or_skin}, "
                f"mengenakan {self.clothing}, memiliki {self.unique_trait})"
            )
        elif self.character_type == "robot":
            return (
                f"{self.name} (robot kecil ramah warna {self.fur_or_skin}, "
                f"aksesori {self.accessories}, memiliki {self.unique_trait})"
            )
        else:
            gender_label = "anak" if self.gender == "netral" else f"anak {self.gender}"
            return (
                f"{self.name} ({gender_label} usia ~{self.age_approx} th, rambut {self.hair_style} {self.hair_color}, "
                f"pakaian {self.clothing}, {self.unique_trait})"
            )


class CharacterRegistry:
    """
    Registry container for all story characters.
    """

    def __init__(self):
        self.characters: List[Character] = []

    def set_main_character(self, character: Character):
        character.role = "tokoh_utama"
        # Remove any existing main character
        self.characters = [c for c in self.characters if c.role != "tokoh_utama"]
        self.characters.insert(0, character)

    def add_supporting_character(self, character: Character) -> bool:
        supporting_count = sum(1 for c in self.characters if c.role == "pendukung")
        if supporting_count >= 4:
            return False  # Max 4 supporting characters
        character.role = "pendukung"
        self.characters.append(character)
        return True

    def get_main_character(self) -> Optional[Character]:
        for c in self.characters:
            if c.role == "tokoh_utama":
                return c
        return self.characters[0] if self.characters else None

    def get_supporting_characters(self) -> List[Character]:
        return [c for c in self.characters if c.role == "pendukung"]

    def get_all_characters(self) -> List[Character]:
        return self.characters

    def get_character_by_name(self, name: str) -> Optional[Character]:
        for c in self.characters:
            if c.name.lower() == name.lower():
                return c
        return None

    def to_list(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.characters]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "CharacterRegistry":
        reg = cls()
        for item in data:
            char = Character(
                name=item.get("name", "Tokoh"),
                role=item.get("role", "tokoh_utama"),
                character_type=item.get("character_type", "anak"),
                gender=item.get("gender", "netral"),
                age_approx=item.get("age_approx", 7),
                fur_or_skin=item.get("fur_or_skin", "cerah"),
                hair_style=item.get("hair_style", "pendek"),
                hair_color=item.get("hair_color", "hitam"),
                clothing=item.get("clothing", "pakaian santai"),
                accessories=item.get("accessories", ""),
                default_expression=item.get("default_expression", "senang"),
                unique_trait=item.get("unique_trait", "")
            )
            if char.role == "tokoh_utama":
                reg.set_main_character(char)
            else:
                reg.add_supporting_character(char)
        return reg

