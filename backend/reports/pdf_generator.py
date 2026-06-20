"""Renders a ReportDraft as a Komatsu-branded PDF using reportlab."""
import logging
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import settings
from reports.assembler import ReportDraft
from reports.chart_renderer import render_chart, should_chart

logger = logging.getLogger(__name__)

# Komatsu brand palette
_KOMATSU_YELLOW = colors.HexColor("#FFD100")
_KOMATSU_DARK = colors.HexColor("#1C1C1C")
_KOMATSU_GREY = colors.HexColor("#F5F5F5")

_PAGE_W, _PAGE_H = A4
_MARGIN = 2 * cm


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontSize=22,
            textColor=_KOMATSU_DARK,
            spaceAfter=6,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=4,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base["Heading1"],
            fontSize=14,
            textColor=_KOMATSU_DARK,
            spaceBefore=12,
            spaceAfter=6,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            spaceAfter=8,
        ),
        "subsection_heading": ParagraphStyle(
            "subsection_heading",
            parent=base["Heading2"],
            fontSize=11,
            textColor=_KOMATSU_DARK,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "appendix_heading": ParagraphStyle(
            "appendix_heading",
            parent=base["Heading2"],
            fontSize=11,
            textColor=colors.grey,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "appendix_item": ParagraphStyle(
            "appendix_item",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.grey,
            leftIndent=12,
            spaceAfter=2,
        ),
        "table_caption": ParagraphStyle(
            "table_caption",
            parent=base["Normal"],
            fontSize=9,
            textColor=_KOMATSU_DARK,
            spaceBefore=8,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontSize=7,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
    }


def _header_footer(canvas, doc, run_date: str) -> None:
    canvas.saveState()
    # Top bar
    canvas.setFillColor(_KOMATSU_YELLOW)
    canvas.rect(0, _PAGE_H - 1.0 * cm, _PAGE_W, 1.0 * cm, fill=1, stroke=0)
    canvas.setFillColor(_KOMATSU_DARK)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(_MARGIN, _PAGE_H - 0.65 * cm, "KOMATSU MARKET INTELLIGENCE")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(_PAGE_W - _MARGIN, _PAGE_H - 0.65 * cm, run_date)
    # Bottom bar
    canvas.setFillColor(_KOMATSU_GREY)
    canvas.rect(0, 0, _PAGE_W, 0.8 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.grey)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(_PAGE_W / 2, 0.25 * cm, f"Page {doc.page} — Confidential")
    canvas.restoreState()


_CHART_WIDTH = 15 * cm
_CHART_ASPECT = 3.2 / 6.2  # matches chart_renderer's figsize=(6.2, 3.2)

# Ranks domains by which one most plausibly "owns" a dataset that more than one
# domain happened to collect (e.g. a commodity price series belongs to
# commodities even if mining_projects also pulled it for a site's output; a
# company's own financials belong to competition/customers over mining_projects).
# This is purely a cross-domain dedup tie-breaker — it does NOT affect chapter
# display order, which stays as core/graph.py::_DOMAINS produced it.
_DATASET_OWNERSHIP_PRIORITY = [
    "commodities",
    "competition",
    "customers",
    "distributors",
    "mining_projects",
    "macro_geopolitics",
    "general_search",
]


def _dataset_key(dataset: dict):
    """Stable identity for cross-domain/intra-domain dedup of a dataset."""
    return dataset.get("series_id") or (dataset.get("tool", ""), dataset.get("title", ""))


def _build_dataset_owners(chapters: list[dict]) -> dict:
    """Decide which single domain renders each dataset that appears in >1 chapter.

    Walks chapters in `_DATASET_OWNERSHIP_PRIORITY` order (not display order) so
    the most topically relevant domain claims a shared series; chapters not in
    that list fall to the end, in their original relative order.
    """
    rank = {d: i for i, d in enumerate(_DATASET_OWNERSHIP_PRIORITY)}
    ordered = sorted(
        chapters,
        key=lambda ch: rank.get(ch.get("domain", ""), len(_DATASET_OWNERSHIP_PRIORITY)),
    )
    owner: dict = {}
    for chapter in ordered:
        domain = chapter.get("domain", "unknown")
        all_datasets = list(chapter.get("datasets") or [])
        for sc in chapter.get("subchapters") or []:
            all_datasets.extend(sc.get("datasets") or [])
        for ds in all_datasets:
            key = _dataset_key(ds)
            if key not in owner:
                owner[key] = domain
    return owner

_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), _KOMATSU_DARK),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _KOMATSU_GREY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
])


def _dataset_flowables(dataset: dict, styles: dict) -> list:
    """Render one Gate-2 dataset as a chart or captioned Table flowable, if it's table-shaped."""
    if dataset.get("kind") != "table":
        return []
    columns = dataset.get("columns") or []
    rows = dataset.get("rows") or []
    if not columns or not rows:
        return []

    max_rows = settings.report.report_table_max_rows
    caption = Paragraph(dataset.get("title", ""), styles["table_caption"])

    if should_chart(dataset, max_rows):
        png_bytes = render_chart(dataset)
        if png_bytes is not None:
            img = Image(BytesIO(png_bytes), width=_CHART_WIDTH, height=_CHART_WIDTH * _CHART_ASPECT)
            return [caption, img]
        # Chart rendering failed — fall through to the (truncated) table below
        # so the data is shown rather than silently dropped.

    table_data = [columns] + [list(r) for r in rows[:max_rows]]
    table = Table(table_data, repeatRows=1, hAlign="LEFT")
    table.setStyle(_TABLE_STYLE)

    flowables = [caption, table]
    if len(rows) > max_rows:
        flowables.append(Paragraph(
            f"<i>Showing first {max_rows} of {len(rows)} rows.</i>",
            styles["appendix_item"],
        ))
    return flowables


class PdfGenerator:
    @staticmethod
    def generate(draft: ReportDraft) -> bytes:
        output_dir = Path(settings.report.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{draft.run_id}.pdf"

        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        styles = _build_styles()

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=_MARGIN,
            rightMargin=_MARGIN,
            topMargin=1.8 * cm,
            bottomMargin=1.2 * cm,
            title=f"Komatsu Market Intelligence — {draft.run_id[:8]}",
            author="Komatsu Market Intelligence Agent",
        )

        # Bind run_date into callbacks
        def on_page(canvas, doc):
            _header_footer(canvas, doc, run_date)

        story = []

        # --- Cover / Page 1 ---
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Market Intelligence Report", styles["cover_title"]))
        query_snippet = draft.query[:120] + ("…" if len(draft.query) > 120 else "")
        story.append(Paragraph(f"Query: {query_snippet}", styles["cover_sub"]))
        story.append(Paragraph(f"Run ID: {draft.run_id}", styles["cover_sub"]))
        story.append(Paragraph(f"Generated: {run_date}", styles["cover_sub"]))
        story.append(HRFlowable(width="100%", thickness=1, color=_KOMATSU_YELLOW, spaceAfter=10))

        if draft.exec_summary:
            story.append(Paragraph("Executive Summary", styles["section_heading"]))
            for para in draft.exec_summary.split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(para, styles["body"]))
        else:
            story.append(Paragraph(
                "<i>(Executive summary not available for this run.)</i>",
                styles["body"],
            ))

        # --- Domain chapters ---
        owners = _build_dataset_owners(draft.chapters)
        for chapter in draft.chapters:
            domain = chapter.get("domain", "unknown")
            text = chapter.get("text", "")
            subchapters = chapter.get("subchapters") or []
            story.append(PageBreak())
            story.append(Paragraph(domain.replace("_", " ").title(), styles["section_heading"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=_KOMATSU_YELLOW, spaceAfter=8))

            if subchapters:
                # Tier-1: one sub-section per entity, with its own tables.
                attributed: set = set()
                for sc in subchapters:
                    label = sc.get("subdomain_label") or sc.get("subdomain_key") or "Entity"
                    story.append(Paragraph(label, styles["subsection_heading"]))
                    for para in (sc.get("text") or "").split("\n\n"):
                        para = para.strip()
                        if para:
                            story.append(Paragraph(para, styles["body"]))
                    for dataset in sc.get("datasets") or []:
                        key = _dataset_key(dataset)
                        attributed.add(key)
                        if owners.get(key) == domain:
                            story.extend(_dataset_flowables(dataset, styles))
                # Tier-2: the domain-level landscape synthesis.
                story.append(Paragraph("Landscape Summary", styles["subsection_heading"]))
                for para in text.split("\n\n"):
                    para = para.strip()
                    if para:
                        story.append(Paragraph(para, styles["body"]))
                # Domain tables not already shown under an entity, and owned by this domain.
                for dataset in chapter.get("datasets") or []:
                    key = _dataset_key(dataset)
                    if key not in attributed and owners.get(key) == domain:
                        story.extend(_dataset_flowables(dataset, styles))
            else:
                for para in text.split("\n\n"):
                    para = para.strip()
                    if para:
                        story.append(Paragraph(para, styles["body"]))
                for dataset in chapter.get("datasets") or []:
                    if owners.get(_dataset_key(dataset)) == domain:
                        story.extend(_dataset_flowables(dataset, styles))

        # --- Appendix (merge log + warnings) ---
        appendix_items = (
            [f"Merge resolution: {entry}" for entry in draft.merge_log]
            + [f"Warning: {w}" for w in draft.warnings]
        )
        if appendix_items:
            story.append(PageBreak())
            story.append(Paragraph("Appendix — Merge Log & Warnings", styles["appendix_heading"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6))
            for item in appendix_items:
                story.append(Paragraph(f"• {item}", styles["appendix_item"]))

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

        pdf_bytes = buf.getvalue()
        pdf_path.write_bytes(pdf_bytes)
        logger.info("PdfGenerator: wrote %d bytes to %s", len(pdf_bytes), pdf_path)
        return pdf_bytes
