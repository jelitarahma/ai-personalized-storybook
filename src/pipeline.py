"""
End-to-End Storybook Generation Pipeline.
Integrates Profile Validation -> Child Data Protection -> Prompt Building ->
GPT-2 Narrative Generation -> Attribute Extraction -> Semantic Latent Mapping ->
Dual Visual Generation (Character & Scene) -> Storybook Page Compilation ->
Ethical & Performance Benchmarking.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path for standalone execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.reproducibility import set_seed
from src.utils.system_monitor import measure_stage, get_memory_usage
from src.narrative.prompt_builder import build_personalized_prompt, build_story_seed_prefix
from src.narrative.gpt2_generator import StoryGenerator
from src.narrative.evaluator import evaluate_story_comprehensive
from src.visual.attribute_extractor import extract_visual_attributes
from src.visual.renderer import VisualRenderer
from src.ethics.child_safety import ChildDataProtector
from src.ethics.bias_checker import BiasAndFairnessChecker
from src.ethics.content_filter import ChildContentFilter


class StorybookPipeline:
    """
    Main orchestrator for the AI Personalized Storybook application.
    """

    def __init__(
        self,
        model_name_or_path: str = "cahya/gpt2-small-indonesian-522M",
        device: Optional[str] = None,
        seed: int = 42
    ):
        self.seed = seed
        set_seed(seed)
        self.protector = ChildDataProtector()
        self.content_filter = ChildContentFilter()
        self.bias_checker = BiasAndFairnessChecker()
        self.generator = StoryGenerator(model_name_or_path=model_name_or_path, device=device)
        self.renderer = VisualRenderer(nz=100, device=device, seed=seed)

        # Load reference training dataset for novelty/overlap evaluation
        self.dataset_stories = []
        ds_path = "data/raw/indonesian_children_stories.json"
        if os.path.exists(ds_path):
            with open(ds_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                self.dataset_stories = [item["story"] for item in raw_data]

    def run(
        self,
        child_profile: Dict[str, Any],
        output_dir: str = "outputs/generated_book",
        strategy: str = "top_p",
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        temperature: float = 0.8,
        max_new_tokens: int = 140,
        anonymous_mode: bool = False,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the end-to-end storybook generation pipeline.

        Returns:
            dict containing generated text, extracted attributes, image paths,
            and comprehensive ethical and performance metrics.
        """
        exec_seed = seed if seed is not None else self.seed
        set_seed(exec_seed)
        os.makedirs(output_dir, exist_ok=True)
        stage_benchmarks: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # Stage 1: Child Privacy & Safety Pre-checks
        # -------------------------------------------------------------
        with measure_stage("Stage_1_Child_Safety_Validation") as m:
            is_valid, msg, clean_profile = self.protector.validate_and_minimize(child_profile)
            if not is_valid:
                raise ValueError(f"Child profile validation failed: {msg}")

            if anonymous_mode:
                clean_profile, alias = self.protector.pseudonymize_profile(clean_profile)

            # Check prompt theme for unsafe terms
            is_safe_prompt, prompt_flags = self.content_filter.screen_text(clean_profile.get("theme", ""))
            if not is_safe_prompt:
                clean_profile["theme"], _, _ = self.content_filter.sanitize_text(clean_profile["theme"])
        stage_benchmarks["privacy_and_safety"] = m

        # -------------------------------------------------------------
        # Stage 2: Prompt Construction
        # -------------------------------------------------------------
        with measure_stage("Stage_2_Prompt_Building") as m:
            instruction_prompt = build_personalized_prompt(clean_profile, format_type="natural")
            starter_prefix = build_story_seed_prefix(clean_profile)
        stage_benchmarks["prompt_building"] = m

        # -------------------------------------------------------------
        # Stage 3: Narrative Generation via Indonesian GPT-2
        # -------------------------------------------------------------
        with measure_stage("Stage_3_GPT2_Narrative_Generation") as m:
            gen_result = self.generator.generate(
                prompt=starter_prefix,
                max_new_tokens=max_new_tokens,
                strategy=strategy,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=exec_seed
            )
            raw_story = gen_result["story"]

            # Content moderation & sanitization on generated text
            sanitized_story, was_sanitized, replaced_words = self.content_filter.sanitize_text(raw_story)
        stage_benchmarks["narrative_generation"] = m

        # -------------------------------------------------------------
        # Stage 4: Narrative Evaluation & Memorization Check
        # -------------------------------------------------------------
        with measure_stage("Stage_4_Narrative_Evaluation") as m:
            narrative_metrics = evaluate_story_comprehensive(
                story_text=sanitized_story,
                profile=clean_profile,
                dataset_stories=self.dataset_stories,
                model=self.generator.model,
                tokenizer=self.generator.tokenizer,
                device=self.generator.device
            )
        stage_benchmarks["narrative_evaluation"] = m

        # -------------------------------------------------------------
        # Stage 5: Text-to-Visual Attribute Extraction
        # -------------------------------------------------------------
        with measure_stage("Stage_5_Attribute_Extraction") as m:
            visual_attrs = extract_visual_attributes(sanitized_story, clean_profile)
        stage_benchmarks["attribute_extraction"] = m

        # -------------------------------------------------------------
        # Stage 6: Visual Synthesis (Character, Scene, & Book Page)
        # -------------------------------------------------------------
        with measure_stage("Stage_6_Visual_Synthesis") as m:
            char_path = os.path.join(output_dir, "character.png")
            scene_path = os.path.join(output_dir, "scene.png")
            page_path = os.path.join(output_dir, "storybook_page.png")

            char_img = self.renderer.render_character(
                visual_attrs["character"],
                output_path=char_path,
                seed=exec_seed
            )

            scene_img = self.renderer.render_scene(
                visual_attrs["scene"],
                output_path=scene_path,
                seed=exec_seed
            )

            book_page = self.renderer.compose_storybook_page(
                story_text=sanitized_story,
                profile=clean_profile,
                char_img=char_img,
                scene_img=scene_img,
                output_path=page_path
            )
        stage_benchmarks["visual_synthesis"] = m

        # -------------------------------------------------------------
        # Stage 7: Ethical & Bias Auditing
        # -------------------------------------------------------------
        with measure_stage("Stage_7_Ethical_Bias_Audit") as m:
            narrative_audit = self.bias_checker.audit_narrative_bias(sanitized_story, clean_profile)
            visual_audit = self.bias_checker.audit_visual_attribute_bias(visual_attrs["character"], clean_profile)
        stage_benchmarks["ethical_audit"] = m

        # -------------------------------------------------------------
        # Save Artifacts to Destination Directory
        # -------------------------------------------------------------
        profile_file = os.path.join(output_dir, "input_profile.json")
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(clean_profile, f, ensure_ascii=False, indent=2)

        prompt_file = os.path.join(output_dir, "final_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(instruction_prompt + "\n\nStarter Prefix:\n" + starter_prefix)

        params_file = os.path.join(output_dir, "generation_params.json")
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(gen_result["params"], f, indent=2)

        story_file = os.path.join(output_dir, "generated_story.txt")
        with open(story_file, "w", encoding="utf-8") as f:
            f.write(sanitized_story)

        attrs_file = os.path.join(output_dir, "visual_attributes.json")
        with open(attrs_file, "w", encoding="utf-8") as f:
            json.dump(visual_attrs, f, ensure_ascii=False, indent=2)

        total_duration = sum(s.get("duration_seconds", 0.0) for s in stage_benchmarks.values())

        metrics_file = os.path.join(output_dir, "pipeline_metrics.json")
        final_report = {
            "profile": clean_profile,
            "story": sanitized_story,
            "artifacts": {
                "character_image": char_path,
                "scene_image": scene_path,
                "storybook_page": page_path
            },
            "performance_and_latency": {
                "total_duration_seconds": round(total_duration, 4),
                "stages": stage_benchmarks
            },
            "narrative_metrics": narrative_metrics,
            "ethical_audit": {
                "narrative": narrative_audit,
                "visual": visual_audit,
                "sanitization_applied": was_sanitized,
                "sanitized_replacements": replaced_words
            }
        }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        return final_report

