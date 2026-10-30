"""
AI Personalized Storybook for Children.
Interactive Streamlit Application following the 4-Page User Flow:
PAGE 1: Welcome Page (Exact 1:1 reproduction of design, full canvas, no borders, interactive buttons)
PAGE 2: Child Profile (Interactive form with age guidance & character chips)
PAGE 3: Story Customization (Genre, themes, setting & length)
PAGE 4: Story Preview (Sequential 2-stage generation: Narrative -> Visual Synthesis -> PDF Reader)
"""

import os
import sys
import base64
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from PIL import Image

from src.story.character_registry import Character, CharacterRegistry
from src.story.story_generator import MultiPageStoryGenerator
from src.story.story_arc import StoryArcPlanner
from src.story.prompt_builder import get_age_linguistic_guidance
from src.vision.scene_planner import ScenePlanner
from src.vision.latent_mapping import SemanticLatentMapper
from src.vision.illustration_synthesizer import StorybookIllustrationSynthesizer
from src.vision.pdf_exporter import StorybookPDFExporter
from src.ethics.child_safety import ChildDataProtector
from src.eval.evaluator import compute_distinct_n, compute_repetition_rate

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Storybook — Cerita Anak Personal",
    page_icon="✨📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper to load and encode image
def get_image_base64(rel_path: str) -> str:
    abs_path = os.path.join(project_root, rel_path)
    if os.path.exists(abs_path):
        with open(abs_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

HERO_BG_B64 = get_image_base64("assets/hero_stage_bg.png")

# -----------------------------------------------------------------------------
# DYNAMIC, WHIMSICAL STORYBOOK CSS
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&family=Shantell+Sans:ital,wght@0,600;0,700;1,500&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Lora:ital,wght@0,600;0,700;1,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #2F241D;
}}

/* Base Canvas Warm Paper Texture - Full Screen */
.stApp {{
    background-color: #FAF6EE !important;
}}

/* Remove Streamlit default container padding so hero canvas fits perfectly */
header[data-testid="stHeader"] {{
    display: none !important;
}}

.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1040px !important;
    margin: 0 auto !important;
}}

/* Top Navbar (Seamless, transparent on canvas) */
.nav-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px 18px 10px;
    max-width: 1024px;
    margin: 0 auto;
}}

.nav-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Fredoka', 'Lora', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #2F241E;
    text-decoration: none;
}}

.nav-links {{
    display: flex;
    gap: 14px;
    align-items: center;
}}

.nav-link-active {{
    background-color: #EFE8FA;
    color: #6C5CE7 !important;
    font-weight: 700;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.95rem;
    text-decoration: none;
}}

.nav-link-item {{
    color: #6C594C !important;
    font-weight: 600;
    font-size: 0.95rem;
    text-decoration: none;
    padding: 6px 10px;
    transition: color 0.2s;
}}

.nav-link-item:hover {{
    color: #2F241E !important;
}}

.nav-cta {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #7C67EE 0%, #6350E2 100%);
    color: #FFFFFF !important;
    font-weight: 700;
    padding: 9px 24px;
    border-radius: 25px;
    font-size: 0.95rem;
    text-decoration: none !important;
    box-shadow: 0 4px 16px rgba(124, 103, 238, 0.32);
    transition: all 0.25s ease;
}}

.nav-cta:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(124, 103, 238, 0.45);
    color: #FFFFFF !important;
}}

/* Hero Canvas (Seamless full-artwork background, NO ugly border/box) */
.hero-canvas {{
    position: relative;
    width: 100%;
    max-width: 1024px;
    margin: 0 auto;
    aspect-ratio: 1024 / 682;
    background-image: url('data:image/png;base64,{HERO_BG_B64}');
    background-size: 100% 100%;
    background-repeat: no-repeat;
    background-position: center;
    border: none !important;
    box-shadow: none !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding-top: 24px;
    box-sizing: border-box;
    text-align: center;
}}

/* Center content box */
.hero-content {{
    max-width: 500px;
    margin: 0 auto;
    text-align: center;
}}

/* Crown doodle */
.crown-icon {{
    margin-bottom: 2px;
}}

/* Headline: "we take care about your boo-boo." */
.hero-title {{
    font-family: 'Fredoka', 'Shantell Sans', cursive, sans-serif;
    font-size: 2.85rem;
    font-weight: 700;
    color: #2F241E;
    line-height: 1.05;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}}

.boo-b1 {{ color: #72CBA0; }}
.boo-o1 {{ color: #8EC4E6; }}
.boo-o2 {{ color: #F5B095; }}
.boo-dash {{ color: #A49487; }}
.boo-b2 {{ color: #72CBA0; }}
.boo-o3 {{ color: #F293A6; }}
.boo-o4 {{ color: #B38AE6; }}

/* Subtitle & Description */
.hero-subtitle {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    color: #382C24;
    margin-bottom: 6px;
    line-height: 1.35;
}}

.hero-desc {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.86rem;
    color: #6C5A4E;
    line-height: 1.45;
    max-width: 440px;
    margin: 0 auto 16px auto;
}}

/* Hero CTA Button */
.hero-btn {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #7C67EE 0%, #6350E2 100%);
    color: #FFFFFF !important;
    text-decoration: none !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    padding: 11px 34px;
    border-radius: 30px;
    box-shadow: 0 6px 20px rgba(100, 80, 225, 0.38);
    transition: all 0.25s ease;
    cursor: pointer;
}}

.hero-btn:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(100, 80, 225, 0.5);
    color: #FFFFFF !important;
}}

/* Form Card Styling for Pages 2, 3, 4 */
.form-card {{
    background: #FFFFFF;
    border: 1.5px solid #EAE2D7;
    border-radius: 24px;
    padding: 32px;
    box-shadow: 0 10px 30px rgba(110,85,60,0.04);
    margin-bottom: 24px;
}}

.step-badge {{
    background: #EFE8FA;
    color: #6C5CE7;
    padding: 6px 18px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-block;
    margin-bottom: 12px;
}}

/* Native Streamlit button styling */
div[data-testid="stButton"] button {{
    background: linear-gradient(135deg, #7C67EE 0%, #6350E2 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
    border-radius: 30px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(100, 80, 225, 0.3) !important;
    transition: all 0.25s ease !important;
}}

div[data-testid="stButton"] button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 26px rgba(100, 80, 225, 0.42) !important;
}}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE & QUERY PARAMS INITIALIZATION
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Handle URL query param navigation
if "page" in st.query_params:
    try:
        req_page = int(st.query_params["page"])
        if req_page in [1, 2, 3, 4] and req_page != st.session_state.current_page:
            st.session_state.current_page = req_page
    except Exception:
        pass

# Page 2: Child Profile data
if "child_name" not in st.session_state:
    st.session_state.child_name = ""
if "child_age" not in st.session_state:
    st.session_state.child_age = 7
if "char_preference" not in st.session_state:
    st.session_state.char_preference = "Seekor naga kecil hijau yang ramah"
if "favorite_color" not in st.session_state:
    st.session_state.favorite_color = "Hijau Daun"

# Page 3: Story Customization data
if "story_genre" not in st.session_state:
    st.session_state.story_genre = "Fantasi & Imajinasi (Negeri Ajaib, Benda Berbintang)"
if "story_theme" not in st.session_state:
    st.session_state.story_theme = "Keberanian dan Kebaikan Hati"
if "story_setting" not in st.session_state:
    st.session_state.story_setting = "Puncak bukit hijau dengan istana dongeng dan sungai jernih"
if "story_length" not in st.session_state:
    st.session_state.story_length = 4
if "extra_ideas" not in st.session_state:
    st.session_state.extra_ideas = ""

# Page 4: Generated outputs
if "generated_pages" not in st.session_state:
    st.session_state.generated_pages = []
if "active_preview_page" not in st.session_state:
    st.session_state.active_preview_page = 1
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# -----------------------------------------------------------------------------
# TOP NAVBAR COMPONENT
# -----------------------------------------------------------------------------
def render_navbar():
    if st.session_state.current_page == 1:
        st.markdown(
            """
            <div class="nav-bar">
                <a href="?page=1" target="_self" class="nav-brand">
                    <span style="font-size:1.6rem;">✨📖</span>
                    <span>AI Storybook</span>
                </a>
                <div class="nav-links">
                    <a href="?page=1" target="_self" class="nav-link-active">Home</a>
                    <a href="#about" class="nav-link-item">About</a>
                    <a href="#how-it-works" class="nav-link-item">How It Works</a>
                    <a href="#contact" class="nav-link-item">Contact</a>
                </div>
                <div>
                    <a href="?page=2" target="_self" class="nav-cta">Start Creating &nbsp;→</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="nav-bar">
                <a href="?page=1" target="_self" class="nav-brand">
                    <span style="font-size:1.6rem;">✨📖</span>
                    <span>AI Storybook</span>
                </a>
                <div class="nav-links">
                    <span style="font-weight:700; color:#6C5CE7; background:#EFE8FA; padding:5px 16px; border-radius:20px;">
                        Tahap {st.session_state.current_page} dari 4
                    </span>
                </div>
                <div>
                    <a href="?page=1" target="_self" class="nav-cta" style="background:#55483E;">🏠 Kembali ke Beranda</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# PAGE 1: WELCOME PAGE (EXACT 1:1 REPRODUCTION OF DESIGN)
# =============================================================================
if st.session_state.current_page == 1:
    render_navbar()

    # The Hero Canvas with exact artwork and live centered typography
    st.markdown(
        """
        <div class="hero-canvas">
            <div class="hero-content">
                <div class="crown-icon">
                    <svg width="40" height="28" viewBox="0 0 42 30" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2.5 9.5L9.5 27.5H32.5L39.5 9.5L28.5 18.5L21 2.5L13.5 18.5L2.5 9.5Z" stroke="#7A6048" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
                        <circle cx="2.5" cy="7.5" r="2.5" fill="#E8B257"/>
                        <circle cx="21" cy="2.5" r="2.5" fill="#E8B257"/>
                        <circle cx="39.5" cy="7.5" r="2.5" fill="#E8B257"/>
                    </svg>
                </div>
                <h1 class="hero-title">
                    we take care<br>
                    about your<br>
                    <span class="boo-b1">b</span><span class="boo-o1">o</span><span class="boo-o2">o</span><span class="boo-dash">-</span><span class="boo-b2">b</span><span class="boo-o3">o</span><span class="boo-o4">o</span>.
                </h1>
                <div class="hero-subtitle">
                    Ayo, salurkan imajinasimu dan buat cerita ajaibmu sendiri!
                </div>
                <div class="hero-desc">
                    AI Personalized Storybook untuk menciptakan cerita anak yang unik, sesuai dengan profil dan minat si kecil, dilengkapi dengan ilustrasi yang penuh warna dan imajinasi.
                </div>
                <div>
                    <a href="?page=2" target="_self" class="hero-btn">
                        Mulai Sekarang &nbsp;→
                    </a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 2: CHILD PROFILE
# =============================================================================
elif st.session_state.current_page == 2:
    render_navbar()
    st.markdown("<div class='step-badge'>Tahap 1 dari 2: Profil Anak</div>", unsafe_allow_html=True)
    st.markdown("# 👤 Profil Pembaca Cilik (Child Profile)")
    st.markdown("Berikan sedikit informasi tentang si kecil agar buku cerita terasa begitu personal dan istimewa.")

    with st.container():
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)

        c_p1, c_p2 = st.columns(2, gap="large")
        with c_p1:
            st.session_state.child_name = st.text_input(
                "Nama Anak / Panggilan Sayang:",
                value=st.session_state.child_name,
                placeholder="Misal: Kirana, Nanda, Budi..."
            )

            st.session_state.child_age = st.slider(
                "Usia Anak (Tahun):",
                min_value=4,
                max_value=12,
                value=st.session_state.child_age
            )

            guidance = get_age_linguistic_guidance(st.session_state.child_age)
            st.info(f"💡 **Panduan Usia {st.session_state.child_age} Tahun:** {guidance['sentence_structure']} ({guidance['pacing']})")

        with c_p2:
            st.session_state.char_preference = st.text_input(
                "Preferensi Karakter / Tokoh (Opsional):",
                value=st.session_state.char_preference,
                placeholder="Misal: anak penjelajah ramah, naga kecil hijau..."
            )

            st.caption("Pilihan Cepat Karakter:")
            q_cols = st.columns(3)
            with q_cols[0]:
                if st.button("🐉 Naga Kecil", use_container_width=True):
                    st.session_state.char_preference = "Seekor naga kecil hijau yang ramah"
                    st.rerun()
            with q_cols[1]:
                if st.button("🐰 Kelinci Ceria", use_container_width=True):
                    st.session_state.char_preference = "Kelinci putih berbulu lembut yang suka bertualang"
                    st.rerun()
            with q_cols[2]:
                if st.button("🚀 Penjelajah", use_container_width=True):
                    st.session_state.char_preference = "Seorang anak penjelajah cilik yang pemberani"
                    st.rerun()

            st.session_state.favorite_color = st.selectbox(
                "Warna Favorit Si Kecil:",
                ["Hijau Daun", "Biru Langit", "Kuning Mentari", "Merah Muda Ceria", "Oranye Hangat", "Ungu Bintang"],
                index=0
            )

        st.markdown("</div>", unsafe_allow_html=True)

    c_b2, c_n2 = st.columns([1, 1])
    with c_b2:
        if st.button("← Kembali ke Beranda", use_container_width=True):
            st.session_state.current_page = 1
            st.query_params["page"] = "1"
            st.rerun()
    with c_n2:
        if st.button("Lanjutkan: Kustomisasi Cerita →", type="primary", use_container_width=True):
            if not st.session_state.child_name.strip():
                st.session_state.child_name = "Sahabat Cilik"
            st.session_state.current_page = 3
            st.query_params["page"] = "3"
            st.rerun()


# =============================================================================
# PAGE 3: STORY CUSTOMIZATION
# =============================================================================
elif st.session_state.current_page == 3:
    render_navbar()
    st.markdown("<div class='step-badge'>Tahap 2 dari 2: Kustomisasi Cerita</div>", unsafe_allow_html=True)
    st.markdown("# 🎨 Kustomisasi Cerita (Story Customization)")
    st.markdown("Pilih tema, latar, dan arah petualangan yang ingin dijalani si kecil.")

    with st.container():
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)

        c_s1, c_s2 = st.columns(2, gap="large")
        with c_s1:
            st.session_state.story_genre = st.selectbox(
                "Genre Cerita:",
                [
                    "Fantasi & Imajinasi (Negeri Ajaib, Benda Berbintang)",
                    "Alam & Hewan (Sungai Jernih, Hutan Rimba, Fabel)",
                    "Kehidupan Sehari-hari (Bermain Bersama Sahabat & Keluarga)",
                    "Budaya & Cerita Rakyat Nusantara",
                    "Petualangan Sains Sederhana"
                ],
                index=0
            )

            st.session_state.story_theme = st.text_input(
                "Tema / Nilai Kebaikan:",
                value=st.session_state.story_theme,
                placeholder="Misal: Keberanian, tolong-menolong, kejujuran..."
            )

            st.session_state.story_length = st.select_slider(
                "Panjang Cerita (Jumlah Halaman):",
                options=[3, 4, 5, 6],
                value=st.session_state.story_length
            )

        with c_s2:
            st.session_state.story_setting = st.selectbox(
                "Setting / Latar Cerita:",
                [
                    "Puncak bukit hijau dengan istana dongeng dan sungai jernih",
                    "Tepi sungai berumput dengan pepohonan willow",
                    "Halaman rumah desa bercerobong hangat",
                    "Kebun bunga liar yang asri",
                    "Negeri awan berpelangi di atas pegunungan"
                ],
                index=0
            )

            st.session_state.extra_ideas = st.text_area(
                "Ide Tambahan / Pesan Khusus (Opsional):",
                value=st.session_state.extra_ideas,
                placeholder="Misal: sertakan balon udara oranye dan sahabat kelinci putih..."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    c_b3, c_n3 = st.columns([1, 1])
    with c_b3:
        if st.button("← Kembali ke Profil Anak", use_container_width=True):
            st.session_state.current_page = 2
            st.query_params["page"] = "2"
            st.rerun()
    with c_n3:
        if st.button("✨ Ciptakan Ceritaku Sekarang (Generate My Story) →", type="primary", use_container_width=True):
            st.session_state.generated_pages = []  # Trigger generation
            st.session_state.current_page = 4
            st.query_params["page"] = "4"
            st.rerun()


# =============================================================================
# PAGE 4: STORY PREVIEW & AI GENERATION PROCESS
# =============================================================================
elif st.session_state.current_page == 4:
    render_navbar()

    # If stories haven't been generated yet for this session, execute the 2-stage AI process
    if not st.session_state.generated_pages:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1.5px solid #EFE4D6; border-radius:24px; padding:36px; text-align:center; box-shadow:0 12px 35px rgba(110,85,60,0.06);">
                <span style="font-size:3rem;">🪄✨</span>
                <h2 style="font-family:'Lora',serif; color:#35261C; margin:12px 0 6px 0;">Sedang Merajut Cerita Ajaib...</h2>
                <p style="color:#6C5A4E;">Sistem sedang menjalankan proses kecerdasan buatan berurutan: Generasi Narasi disusul Sintesis Ilustrasi.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        progress_bar = st.progress(0, text="Langkah 1/2: Menghasilkan narasi cerita dengan Indonesian GPT-2...")
        output_dir = "outputs/user_storybook"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Setup Character Registry
        reg = CharacterRegistry()
        main_c = Character(
            name=st.session_state.child_name,
            role="tokoh_utama",
            character_type="anak" if "naga" not in st.session_state.char_preference.lower() else "hewan",
            age_approx=st.session_state.child_age,
            clothing=f"pakaian warna {st.session_state.favorite_color.lower()}, {st.session_state.char_preference}"
        )
        reg.set_main_character(main_c)

        story_gen = MultiPageStoryGenerator()
        planner = ScenePlanner()
        latent_mapper = SemanticLatentMapper()
        synthesizer = StorybookIllustrationSynthesizer()

        tot_pages = st.session_state.story_length

        premise_text = f"{st.session_state.char_preference}. {st.session_state.extra_ideas}".strip()
        if not premise_text:
            premise_text = f"Petualangan {st.session_state.child_name} di {st.session_state.story_setting}."

        # Stage 1: Narrative Generation (Single cohesive book generation)
        progress_bar.progress(20, text="Langkah 1/2: Merajut teks alur cerita berkesinambungan...")
        raw_pages = story_gen.generate_full_book(
            premise=premise_text,
            age=st.session_state.child_age,
            character_reg=reg,
            total_pages=tot_pages,
            world_settings={"location": st.session_state.story_setting, "weather": "cerah"},
            moral_theme=st.session_state.story_theme,
            strategy="top_p",
            base_seed=42
        )

        # Stage 2: Visual Generation per page
        generated_pages = []
        illustration_paths = []
        for p_idx, page in enumerate(raw_pages):
            p_num = p_idx + 1
            pct = 40 + int((p_num / tot_pages) * 45)
            progress_bar.progress(pct, text=f"Langkah 2/2: Melukis ilustrasi adegan halaman {p_num} dari {tot_pages}...")

            scene_json = planner.plan_scene_for_page(
                page_idx=p_num,
                total_pages=tot_pages,
                page_text=page["text"],
                character_reg=reg,
                world_settings={"location": st.session_state.story_setting, "weather": "cerah"}
            )

            latent_info = latent_mapper.map_scene_to_latent(
                scene_json=scene_json,
                identity_seed=42,
                page_variation_seed=42 + (p_num * 17)
            )

            img_path = os.path.join(output_dir, f"page_{p_num}_illustration.png")
            synthesizer.synthesize_scene_illustration(
                scene_json=scene_json,
                character_reg=reg,
                output_path=img_path
            )
            illustration_paths.append(img_path)

            tokens = page["text"].split()
            page["scene_json"] = scene_json
            page["latent_info"] = latent_info
            page["illustration_path"] = img_path
            page["metrics"] = {
                "distinct_1": compute_distinct_n(tokens, 1),
                "repetition_rate": compute_repetition_rate(tokens, 3)
            }
            generated_pages.append(page)

        # Stage 3: PDF Compilation
        progress_bar.progress(95, text="Mengompilasi buku cerita ke format PDF...")
        pdf_path = os.path.join(output_dir, "buku_cerita.pdf")
        pdf_bytes = StorybookPDFExporter.export_storybook_pdf(
            title=f"Kisah {st.session_state.child_name}",
            child_name=st.session_state.child_name,
            age=st.session_state.child_age,
            pages_data=generated_pages,
            illustration_paths=illustration_paths,
            output_path=pdf_path,
            moral_theme=st.session_state.story_theme,
            world_settings={"location": st.session_state.story_setting}
        )

        st.session_state.generated_pages = generated_pages
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.active_preview_page = 1
        progress_bar.progress(100, text="Selesai!")
        st.rerun()

    # Reader Mode Display
    pages = st.session_state.generated_pages
    total_p = len(pages)
    curr_p = st.session_state.active_preview_page
    page_data = pages[curr_p - 1]

    # Reader Top Controls
    c_phead, c_pnav = st.columns([2, 1])
    with c_phead:
        st.markdown(f"## 📖 Kisah Petualangan {st.session_state.child_name}")
        st.caption(f"Genre: {st.session_state.story_genre} &bull; Usia: {st.session_state.child_age} Tahun &bull; Tema: {st.session_state.story_theme}")
    with c_pnav:
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("◀", disabled=(curr_p <= 1), use_container_width=True):
                st.session_state.active_preview_page -= 1
                st.rerun()
        with p2:
            st.markdown(f"<p style='text-align:center; font-weight:700; padding-top:8px; margin:0;'>Halaman {curr_p} / {total_p}</p>", unsafe_allow_html=True)
        with p3:
            if st.button("▶", disabled=(curr_p >= total_p), use_container_width=True):
                st.session_state.active_preview_page += 1
                st.rerun()

    # Two-Column Book Spread
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    c_img, c_txt = st.columns([1, 1], gap="large")

    with c_img:
        img_p = page_data.get("illustration_path")
        if img_p and os.path.exists(img_p):
            st.image(img_p, use_container_width=True)
        else:
            st.info("Memuat gambar...")

    with c_txt:
        st.markdown(f"<div style='font-family:Lora,serif; font-size:1.35rem; font-weight:700; color:#D97358; margin-bottom:12px;'>Babak: {page_data.get('stage_title', 'Petualangan')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Lora,serif; font-size:1.18rem; line-height:1.85; color:#2F251E;'>{page_data.get('text', '')}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Action Toolbar
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        if st.button("🔄 Regenerasi Gambar", use_container_width=True):
            with st.spinner("Melukis ulang halaman ini..."):
                reg = CharacterRegistry()
                reg.set_main_character(Character(name=st.session_state.child_name, age_approx=st.session_state.child_age))
                synth = StorybookIllustrationSynthesizer()
                synth.synthesize_scene_illustration(page_data["scene_json"], reg, output_path=page_data["illustration_path"])
                st.rerun()

    with b_col2:
        if st.button("✏️ Regenerasi Cerita", use_container_width=True):
            with st.spinner("Menulis ulang narasi halaman ini..."):
                reg = CharacterRegistry()
                reg.set_main_character(Character(name=st.session_state.child_name, age_approx=st.session_state.child_age))
                story_gen = MultiPageStoryGenerator()
                st.session_state.generated_pages = story_gen.regenerate_page(
                    pages=st.session_state.generated_pages,
                    page_idx=curr_p,
                    premise=st.session_state.story_theme,
                    age=st.session_state.child_age,
                    character_reg=reg
                )
                st.rerun()

    with b_col3:
        if st.session_state.pdf_bytes:
            st.download_button(
                label="📥 Unduh PDF Buku",
                data=st.session_state.pdf_bytes,
                file_name=f"Buku_Cerita_{st.session_state.child_name}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with b_col4:
        if st.button("✨ Buat Cerita Baru", use_container_width=True):
            st.session_state.generated_pages = []
            st.session_state.current_page = 1
            st.query_params["page"] = "1"
            st.rerun()

    # Academic Technical Details Accordion
    with st.expander("🔍 Detail Teknis & Evaluasi Model (Module A, B, C)"):
        st.write("##### Scene Planner JSON:")
        st.json(page_data.get("scene_json", {}))
        st.write("##### Semantic Latent Vector (NumPy):")
        st.json(page_data.get("latent_info", {}))
        st.write("##### Metrik Narasi:")
        st.json(page_data.get("metrics", {}))
