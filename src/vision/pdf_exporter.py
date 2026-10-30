"""
PDF Exporter module for AI Personalized Storybook.
Compiles generated multi-page storybooks (cover, illustrations, narrative text, moral reflections)
into publication-quality PDF documents using ReportLab.
"""

import os
import io
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm


class StorybookPDFExporter:
    """
    Exports a completed storybook project into an A4 landscape storybook PDF.
    """

    @classmethod
    def export_storybook_pdf(
        cls,
        title: str,
        child_name: str,
        age: int,
        pages_data: List[Dict[str, Any]],
        illustration_paths: List[str],
        output_path: Optional[str] = None,
        moral_theme: Optional[str] = None,
        world_settings: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Builds the PDF and returns raw bytes (and writes to output_path if specified).
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            output_path or buffer,
            pagesize=landscape(A4),
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.2 * cm,
            bottomMargin=1.2 * cm
        )

        styles = getSampleStyleSheet()

        # Custom Warm Storybook Styles
        title_style = ParagraphStyle(
            'CoverTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=26,
            leading=32,
            textColor=colors.HexColor('#3D312A'),
            alignment=1,  # Center
            spaceAfter=12
        )

        subtitle_style = ParagraphStyle(
            'CoverSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#7C9A82'),
            alignment=1,
            spaceAfter=20
        )

        meta_style = ParagraphStyle(
            'CoverMeta',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=11,
            leading=15,
            textColor=colors.HexColor('#8C7D70'),
            alignment=1
        )

        page_title_style = ParagraphStyle(
            'PageStageTitle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#5A4638'),
            spaceAfter=8
        )

        story_text_style = ParagraphStyle(
            'StoryBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            leading=18,
            textColor=colors.HexColor('#2C241E'),
            spaceAfter=10
        )

        story_elements = []

        # ==========================================
        # 1. COVER PAGE
        # ==========================================
        story_elements.append(Spacer(1, 2 * cm))
        story_elements.append(Paragraph(f"📖 {title}", title_style))
        story_elements.append(Paragraph(f"Kisah Petualangan Spesial untuk <b>{child_name}</b> ({age} Tahun)", subtitle_style))
        story_elements.append(Spacer(1, 0.5 * cm))

        if illustration_paths and os.path.exists(illustration_paths[0]):
            story_elements.append(RLImage(illustration_paths[0], width=11 * inch * 0.48, height=4.2 * inch * 0.48))
            story_elements.append(Spacer(1, 0.8 * cm))

        world_desc = world_settings.get("location", "Dunia Imajinasi") if world_settings else "Dunia Imajinasi"
        moral_desc = f"Pesan Kasih: {moral_theme}" if moral_theme else "Dibuat dengan cinta & kehangatan"
        story_elements.append(Paragraph(f"Latar: {world_desc} &bull; {moral_desc}", meta_style))
        story_elements.append(PageBreak())

        # ==========================================
        # 2. STORY PAGES
        # ==========================================
        for idx, page in enumerate(pages_data):
            p_num = page.get("page_number", idx + 1)
            stage_title = page.get("stage_title", f"Halaman {p_num}")
            text_content = page.get("text", "")

            story_elements.append(Paragraph(f"Halaman {p_num}: {stage_title}", page_title_style))

            img_path = illustration_paths[idx] if idx < len(illustration_paths) else None
            
            # Create a 2-column spread or top-image / bottom-text layout
            if img_path and os.path.exists(img_path):
                img_elem = RLImage(img_path, width=5.2 * inch, height=3.6 * inch)
                text_elem = Paragraph(text_content, story_text_style)

                table_data = [[img_elem, text_elem]]
                table = Table(table_data, colWidths=[5.4 * inch, 4.4 * inch])
                table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (1, 0), (1, 0), 18),
                    ('RIGHTPADDING', (0, 0), (0, 0), 10),
                ]))
                story_elements.append(table)
            else:
                story_elements.append(Paragraph(text_content, story_text_style))

            story_elements.append(Spacer(1, 1 * cm))
            if idx < len(pages_data) - 1:
                story_elements.append(PageBreak())

        # Build document
        doc.build(story_elements)

        if output_path:
            with open(output_path, "rb") as f:
                return f.read()

        buffer.seek(0)
        return buffer.getvalue()

