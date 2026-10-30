"""
Command-Line Interface (CLI) for AI Personalized Storybook.
Supports multi-mode operation: end-to-end generation, decoding experiments,
domain fine-tuning, ethical audit, and batch scenarios execution.
"""

import argparse
import json
import sys
import os
from src.pipeline import StorybookPipeline
from src.narrative.trainer import fine_tune_gpt2
from src.ethics.bias_checker import BiasAndFairnessChecker
from src.narrative.gpt2_generator import StoryGenerator
from run_experiments import run_decoding_experiments
from run_scenarios import run_all_scenarios


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Personalized Storybook: Personalized Narrative Generation, Visual Synthesis & Ethics"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "experiment", "train", "audit", "scenarios"],
        help="Execution mode: generate (single storybook), experiment (decoding strategies), train (fine-tune GPT-2), audit (fairness check), scenarios (batch 5 scenarios)"
    )

    # Child Profile Args
    parser.add_argument("--name", type=str, default="Gono", help="Nama anak")
    parser.add_argument("--age", type=int, default=7, help="Usia anak (tahun)")
    parser.add_argument("--genre", type=str, default="Fantasi", help="Genre cerita (Fantasi, Petualangan, Fabel, Sains, Persahabatan)")
    parser.add_argument("--theme", type=str, default="Keberanian dan Kebaikan Hati", help="Tema cerita anak")
    parser.add_argument("--culture", type=str, default="Nusantara", help="Latar budaya cerita (Jawa, Sunda, Minang, Papua, Bali, Dayak, Bugis, Nusantara)")
    parser.add_argument("--character", type=str, default="seekor naga kecil hijau yang ramah", help="Preferensi karakter")

    # Generation Hyperparameters
    parser.add_argument("--strategy", type=str, default="top_p", choices=["top_p", "top_k", "greedy", "hybrid"], help="Strategi decoding teks")
    parser.add_argument("--top_k", type=int, default=50, help="Nilai K untuk Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nilai p untuk Top-p (nucleus) sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature sampling")
    parser.add_argument("--max_tokens", type=int, default=140, help="Maksimal token baru yang digenerasikan")
    parser.add_argument("--seed", type=int, default=42, help="Seed angka acak untuk reproduktifitas")

    # Ethical flags
    parser.add_argument("--anonymous", action="store_true", help="Aktifkan mode pseudonimisasi nama anak untuk perlindungan privasi")

    # Paths
    parser.add_argument("--output_dir", type=str, default="outputs/cli_run", help="Direktori penyimpanan output")
    parser.add_argument("--model", type=str, default="cahya/gpt2-small-indonesian-522M", help="Hugging Face model checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Jumlah epoch untuk fine-tuning")

    return parser.parse_args()


def main():
    args = parse_arguments()
    print("=" * 70)
    print("      AI PERSONALIZED STORYBOOK - GENERATIVE AI SYSTEM")
    print("=" * 70)
    print(f"Mode Operasi : {args.mode.upper()}")

    if args.mode == "generate":
        print(f"Membangun cerita personal untuk {args.name} (Usia {args.age} tahun, Genre: {args.genre})...")
        pipeline = StorybookPipeline(model_name_or_path=args.model, seed=args.seed)
        profile = {
            "name": args.name,
            "age": args.age,
            "genre": args.genre,
            "theme": args.theme,
            "cultural_background": args.culture,
            "character_preference": args.character
        }
        res = pipeline.run(
            child_profile=profile,
            output_dir=args.output_dir,
            strategy=args.strategy,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            anonymous_mode=args.anonymous,
            seed=args.seed
        )
        print("\n--- HASIL CERITA ---")
        print(res["story"])
        print("\n--- ATRIBUT VISUAL TERDETEKSI ---")
        print(json.dumps(res["narrative_metrics"]["profile_alignment"], indent=2, ensure_ascii=False))
        print(f"\nIlustrasi dan Buku Cerita berhasil disimpan ke: {args.output_dir}/")

    elif args.mode == "experiment":
        print("Menjalankan eksperimen decoding Top-K vs Top-p...")
        run_decoding_experiments(
            output_csv=os.path.join("outputs", "decoding_experiments.csv"),
            model_name_or_path=args.model
        )

    elif args.mode == "train":
        print("Menjalankan fine-tuning GPT-2 pada dataset cerita anak...")
        fine_tune_gpt2(
            model_name=args.model,
            output_dir="models/checkpoints/gpt2_storybook",
            num_epochs=args.epochs,
            batch_size=2,
            seed=args.seed
        )

    elif args.mode == "audit":
        print("Menjalankan Audit Etika Kontrafaktual Gender & Budaya...")
        generator = StoryGenerator(model_name_or_path=args.model)
        checker = BiasAndFairnessChecker()
        base_scenario = {
            "name": "Budi",
            "age": 8,
            "genre": "Petualangan",
            "theme": "Keberanian dan Kebaikan Hati"
        }
        audit_res = checker.run_comparative_counterfactual_audit(generator, base_scenario)
        print("\n--- HASIL AUDIT KONTRAFAKTUAL ---")
        print(json.dumps(audit_res, indent=2, ensure_ascii=False))

    elif args.mode == "scenarios":
        print("Menjalankan 5 Skenario Profil Beragam...")
        run_all_scenarios(base_output_dir="outputs", model_name_or_path=args.model)


if __name__ == "__main__":
    main()

