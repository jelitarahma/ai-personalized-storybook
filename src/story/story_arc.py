"""
Story Arc Planner for Multi-Page Children's Storybooks (3-6 Pages).
Defines page-by-page narrative progression ensuring continuity, age-appropriateness,
and a satisfying narrative arc (Beginning -> Rising Action -> Climax -> Resolution).
"""

from typing import Dict, Any, List


class StoryArcPlanner:
    """
    Plans the narrative trajectory for each page of the storybook.
    """

    @staticmethod
    def get_stage_definition(page_idx: int, total_pages: int) -> Dict[str, str]:
        """
        Returns the narrative stage and focus for a given page index (1-indexed).
        """
        if total_pages < 3:
            total_pages = 3
        elif total_pages > 6:
            total_pages = 6

        if page_idx == 1:
            return {
                "stage": "Pengenalan",
                "title": "Awal Mula Kisah",
                "focus": (
                    "Perkenalkan tokoh utama di tempat asalnya yang nyaman. "
                    "Gambarkan sifat ramah tokoh, suasana hari itu, dan keinginan atau rasa penasarannya."
                )
            }

        elif page_idx == total_pages:
            return {
                "stage": "Penyelesaian",
                "title": "Kebahagiaan dan Pelajaran Berharga",
                "focus": (
                    "Masalah telah terselesaikan dengan baik. Tokoh merasakan kedamaian, rasa syukur, "
                    "atau kehangatan persahabatan/keluarga. Tutup cerita dengan kalimat yang menenangkan dan ramah anak."
                )
            }

        elif total_pages == 3:
            # Page 2 of 3
            return {
                "stage": "Tantangan & Usaha",
                "title": "Petualangan Dimulai",
                "focus": (
                    "Tokoh menghadapi rintangan kecil atau memulai petualangan baru. "
                    "Tokoh berusaha dengan rasa ingin tahu dan keberanian untuk mencari jalan keluar."
                )
            }

        elif total_pages == 4:
            if page_idx == 2:
                return {
                    "stage": "Munculnya Petualangan",
                    "title": "Kejutan yang Menarik",
                    "focus": (
                        "Sebuah peristiwa tak terduga atau tanda petualangan muncul. "
                        "Tokoh memutuskan untuk menjelajah atau membantu sahabatnya."
                    )
                }
            else:  # page_idx == 3
                return {
                    "stage": "Momen Puncak",
                    "title": "Kerja Sama & Keberanian",
                    "focus": (
                        "Tokoh menemukan jalan keluar atau mengatasi rintangan penting "
                        "berkat kebaikan hati, kecerdikan, atau bantuan sahabatnya."
                    )
                }

        elif total_pages == 5:
            if page_idx == 2:
                return {
                    "stage": "Awal Perjalanan",
                    "title": "Langkah Pertama",
                    "focus": (
                        "Tokoh memulai perjalanannya menuju tempat baru atau mencari benda penting. "
                        "Latar suasana terasa hidup dan memikat."
                    )
                }
            elif page_idx == 3:
                return {
                    "stage": "Interaksi & Rintangan",
                    "title": "Pertemuan yang Berarti",
                    "focus": (
                        "Tokoh bertemu dengan sahabat atau menghadapi teka-teki kecil. "
                        "Mereka saling bertukar senyum dan saling menolong."
                    )
                }
            else:  # page_idx == 4
                return {
                    "stage": "Klimaks & Penemuan",
                    "title": "Momen Emas",
                    "focus": (
                        "Puncak petualangan: rahasia terungkap atau masalah utama terpecahkan dengan cara yang indah."
                    )
                }

        else:  # total_pages == 6
            stages = {
                2: {
                    "stage": "Gagasan Besar",
                    "title": "Rencana Indah",
                    "focus": "Tokoh mendapatkan ide istimewa dan bersiap mewujudkannya bersama sahabat."
                },
                3: {
                    "stage": "Langkah Awal",
                    "title": "Menembus Alam Baru",
                    "focus": "Tokoh mulai menjelajah, menikmati pemandangan alam dan menemukan hal-hal baru."
                },
                4: {
                    "stage": "Ujian Kerjasama",
                    "title": "Tantangan Bersama",
                    "focus": "Sebuah rintangan kecil muncul yang menguji kesabaran dan kebersamaan tokoh."
                },
                5: {
                    "stage": "Klimaks Cemerlang",
                    "title": "Cahaya Keberhasilan",
                    "focus": "Tokoh berhasil menuntaskan misinya dengan penuh sukacita dan kebanggaan."
                }
            }
            return stages.get(page_idx, {
                "stage": "Perkembangan",
                "title": "Perjalanan Cerita",
                "focus": "Alur terus berkembang menuju penyelesaian."
            })

    @classmethod
    def get_full_outline(cls, total_pages: int) -> List[Dict[str, str]]:
        outline = []
        for i in range(1, total_pages + 1):
            st = cls.get_stage_definition(i, total_pages)
            outline.append({
                "page": i,
                "stage": st["stage"],
                "title": st["title"],
                "focus": st["focus"]
            })
        return outline

