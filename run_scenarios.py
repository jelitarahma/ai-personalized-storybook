"""
Batch Runner for 5 Distinct Multi-Page Storybook Scenarios.
Covers: Fantasi, Alam/Hewan, Kehidupan Sehari-hari, Budaya Lokal, dan Ide Bebas.
Generates multi-page stories, Scene JSON, per-page watercolor illustrations,
ReportLab PDF export, and evaluation metrics.
"""

import os
import json
import time
from typing import List, Dict, Any

from src.story.character_registry import Character, CharacterRegistry
from src.story.story_generator import MultiPageStoryGenerator
from src.vision.scene_planner import ScenePlanner
from src.vision.latent_mapping import SemanticLatentMapper
from src.vision.illustration_synthesizer import StorybookIllustrationSynthesizer
from src.vision.pdf_exporter import StorybookPDFExporter
from src.ethics.child_safety import ChildDataProtector
from src.eval.evaluator import compute_distinct_n, compute_repetition_rate, compute_dataset_ngram_overlap


SCENARIOS_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "scenario_1",
        "category": "Fantasi & Imajinasi",
        "premise": "Kiko si naga kecil hijau yang ramah ingin membantu kawan-kawan menyalakan kembang api warna-warni di puncak bukit.",
        "target_age": 7,
        "page_count": 3,
        "moral_theme": "Kebaikan hati dan menolong sesama",
        "main_char": {
            "name": "Kiko",
            "type": "hewan",
            "gender": "netral",
            "appearance": "sisik hijau lembut, sayap mungil, senyuman hangat"
        },
        "supporting": [
            {"name": "Mimi", "type": "hewan", "appearance": "kelinci putih berbulu halus"}
        ],
        "world": {
            "location": "puncak bukit hijau berkabut lembut dengan bunga matahari",
            "weather": "pagi cerah berawan menyenangkan",
            "atmosphere": "damai dan penuh keajaiban",
            "visual_style": "Watercolor Storybook Ilustrasi Halus"
        },
        "seed": 42
    },
    {
        "id": "scenario_2",
        "category": "Alam & Hewan",
        "premise": "Katak kecil petualang mendayung perahu kayu menyusuri sungai jernih untuk mengantarkan buah beri kepada sahabatnya.",
        "target_age": 6,
        "page_count": 3,
        "moral_theme": "Kesabaran dan indahnya persahabatan sejati",
        "main_char": {
            "name": "Toadi",
            "type": "hewan",
            "gender": "netral",
            "appearance": "tubuh hijau bintik ramah, topi baret merah, rompi kotak-kotak"
        },
        "supporting": [
            {"name": "Domba Putih", "type": "hewan", "appearance": "bulu seputih awan, syal oranye"}
        ],
        "world": {
            "location": "aliran sungai jernih dengan pepohonan willow",
            "weather": "siang hari hangat dengan riak air tenang",
            "atmosphere": "riang dan damai",
            "visual_style": "Watercolor Storybook Ilustrasi Halus"
        },
        "seed": 101
    },
    {
        "id": "scenario_3",
        "category": "Kehidupan Sehari-hari",
        "premise": "Budi dan adiknya bermain perahu kertas di halaman rumah tepi parit desa saat cuaca cerah.",
        "target_age": 8,
        "page_count": 3,
        "moral_theme": "Kerjasama dan kehangatan persaudaraan",
        "main_char": {
            "name": "Budi",
            "type": "anak",
            "gender": "laki-laki",
            "appearance": "rambut pendek rapi hitam, celana berpenjepit, senyum ramah"
        },
        "supporting": [
            {"name": "Siti", "type": "anak", "appearance": "gaun merah muda, rambut kepang pirang"}
        ],
        "world": {
            "location": "halaman rumah klasik bercerobong hangat di pedesaan",
            "weather": "sore hari sejuk menenangkan",
            "atmosphere": "penuh canda tawa dan kehangatan keluarga",
            "visual_style": "Watercolor Storybook Ilustrasi Halus"
        },
        "seed": 202
    },
    {
        "id": "scenario_4",
        "category": "Budaya Lokal Nusantara",
        "premise": "Kirana gadis cilik yang gemar membaca menjelajahi kebun teh dan sungai di kaki gunung Sunda.",
        "target_age": 9,
        "page_count": 3,
        "moral_theme": "Mencintai kearifan alam dan budaya Nusantara",
        "main_char": {
            "name": "Kirana",
            "type": "anak",
            "gender": "perempuan",
            "appearance": "gaun sederhana bersahaja, membawa buku dongeng bersampul kuning"
        },
        "supporting": [],
        "world": {
            "location": "hamparan perbukitan hijau asri di tatar Sunda",
            "weather": "pagi berkabut sejuk dengan sinar mentari lembut",
            "atmosphere": "teduh dan memikat",
            "visual_style": "Watercolor Storybook Ilustrasi Halus"
        },
        "seed": 303
    },
    {
        "id": "scenario_5",
        "category": "Ide Bebas & Misteri Ringan",
        "premise": "Seekor kelinci bersahaja dengan arloji saku mencari jam dinding raksasa yang tersembunyi di balik semak bunga liar.",
        "target_age": 10,
        "page_count": 3,
        "moral_theme": "Ketelitian dan menghargai waktu",
        "main_char": {
            "name": "Kelinci Waktu",
            "type": "hewan",
            "gender": "netral",
            "appearance": "bulu putih lebat, jas rompi kuning, memegang arloji saku rantai"
        },
        "supporting": [
            {"name": "Domba Hitam", "type": "hewan", "appearance": "bulu wol hitam lebat yang hangat"}
        ],
        "world": {
            "location": "kebun bunga liar bersemak rimbun",
            "weather": "senja keemasan temaram",
            "atmosphere": "penasaran dan penuh petualangan",
            "visual_style": "Watercolor Storybook Ilustrasi Halus"
        },
        "seed": 404
    }
]


def run_all_scenarios(base_output_dir: str = "outputs", model_name_or_path: str = None) -> Dict[str, Any]:
    """
    Executes and generates full multi-page storybooks for all 5 scenarios.
    """
    print("=" * 70)
    print("      RUNNING 5 COMPREHENSIVE MULTI-PAGE STORYBOOK SCENARIOS")
    print("=" * 70)

    protector = ChildDataProtector()
    story_gen = MultiPageStoryGenerator(model_name_or_path=model_name_or_path)
    planner = ScenePlanner()
    latent_mapper = SemanticLatentMapper()
    synthesizer = StorybookIllustrationSynthesizer()

    scenarios_summary = []

    for sc in SCENARIOS_CATALOG:
        sc_id = sc["id"]
        sc_dir = os.path.join(base_output_dir, sc_id)
        os.makedirs(sc_dir, exist_ok=True)

        print(f"\n>>> Memproses [{sc_id.upper()}] - {sc['category']} ({sc['main_char']['name']}, {sc['target_age']} th)...")
        start_time = time.time()

        # 1. Build character registry with pseudonymization check
        clean_profile, pseudonym_applied = protector.pseudonymize_profile({
            "name": sc["main_char"]["name"],
            "age": sc["target_age"],
            "theme": sc["moral_theme"],
            "genre": sc["category"]
        }, use_cute_alias=False)

        reg = CharacterRegistry()
        main_c = Character(
            name=clean_profile["name"],
            role="tokoh_utama",
            character_type=sc["main_char"]["type"],
            gender=sc["main_char"]["gender"],
            age_approx=sc["target_age"],
            clothing=sc["main_char"]["appearance"]
        )
        reg.set_main_character(main_c)

        for sup in sc["supporting"]:
            s_c = Character(
                name=sup["name"],
                role="pendukung",
                character_type=sup["type"],
                clothing=sup["appearance"]
            )
            reg.add_supporting_character(s_c)

        # 2. Multi-page story generation
        pages = story_gen.generate_full_book(
            premise=sc["premise"],
            age=sc["target_age"],
            character_reg=reg,
            total_pages=sc["page_count"],
            world_settings=sc["world"],
            moral_theme=sc["moral_theme"],
            strategy="top_p",
            base_seed=sc["seed"]
        )

        # 3. Scene planning, latent mapping, and illustration synthesis per page
        illustration_paths = []
        for idx, page in enumerate(pages):
            p_num = page["page_number"]
            scene_json = planner.plan_scene_for_page(
                page_idx=p_num,
                total_pages=sc["page_count"],
                page_text=page["text"],
                character_reg=reg,
                world_settings=sc["world"]
            )

            latent_info = latent_mapper.map_scene_to_latent(
                scene_json=scene_json,
                identity_seed=sc["seed"],
                page_variation_seed=sc["seed"] + (p_num * 19)
            )

            img_file = os.path.join(sc_dir, f"page_{p_num}_illustration.png")
            synthesizer.synthesize_scene_illustration(
                scene_json=scene_json,
                character_reg=reg,
                output_path=img_file
            )
            illustration_paths.append(img_file)

            page["scene_json"] = scene_json
            page["latent_info"] = latent_info
            page["illustration_file"] = os.path.basename(img_file)

        # 4. Export storybook to PDF
        pdf_file = os.path.join(sc_dir, f"{sc_id}_buku_cerita.pdf")
        StorybookPDFExporter.export_storybook_pdf(
            title=f"Kisah {main_c.name}",
            child_name=main_c.name,
            age=sc["target_age"],
            pages_data=pages,
            illustration_paths=illustration_paths,
            output_path=pdf_file,
            moral_theme=sc["moral_theme"],
            world_settings=sc["world"]
        )

        elapsed = round(time.time() - start_time, 2)

        # 5. Save scenario artifacts
        full_text = "\n\n".join([f"[Halaman {p['page_number']}: {p['stage_title']}]\n{p['text']}" for p in pages])
        with open(os.path.join(sc_dir, "story.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        with open(os.path.join(sc_dir, "scenes.json"), "w", encoding="utf-8") as f:
            json.dump([p["scene_json"] for p in pages], f, indent=2, ensure_ascii=False)

        with open(os.path.join(sc_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "scenario_id": sc_id,
                "category": sc["category"],
                "premise": sc["premise"],
                "target_age": sc["target_age"],
                "moral_theme": sc["moral_theme"],
                "character_registry": reg.to_list(),
                "world_settings": sc["world"],
                "pseudonym_applied": pseudonym_applied,
                "execution_seconds": elapsed
            }, f, indent=2, ensure_ascii=False)

        # Quick evaluation metrics
        all_tokens = full_text.split()
        d1 = compute_distinct_n(all_tokens, 1)
        sc_summary = {
            "scenario_id": sc_id,
            "category": sc["category"],
            "main_character": main_c.name,
            "age": sc["target_age"],
            "pages": len(pages),
            "distinct_1": d1,
            "execution_seconds": elapsed,
            "pdf_generated": os.path.basename(pdf_file)
        }
        scenarios_summary.append(sc_summary)
        print(f"[OK] Selesai dalam {elapsed}s | PDF: {os.path.basename(pdf_file)} | Distinct-1: {d1:.4f}")

    # Save summary
    summary_path = os.path.join(base_output_dir, "scenarios_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scenarios_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] SEMUA 5 SKENARIO BERHASIL DILAKSANAKAN! Ringkasan: {summary_path}")
    print("=" * 70)
    return {"summary": scenarios_summary}


if __name__ == "__main__":
    run_all_scenarios()
