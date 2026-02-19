"""Combined bookmark extraction pipeline.

Orchestrates the full extraction flow:
1. Check for existing bookmarks
2. Locate and parse TOC pages
3. Detect headings via font heuristics
4. Reconcile and produce final bookmark tree
5. Optionally inject bookmarks back into the PDF
"""

from __future__ import annotations

import logging
from typing import Optional

import fitz

from .bookmark_extractor import extract_existing_bookmarks, has_bookmarks, inject_bookmarks
from .font_heuristics import FontProfile, build_font_profile, detect_headings
from .models import BookmarkEntry, ExtractionResult, PageNumberMapping
from .pdf_utils import build_page_number_mapping, open_pdf
from .reconciler import reconcile_bookmarks
from .toc_parser import find_toc_pages, parse_toc_entries

logger = logging.getLogger(__name__)


def extract_bookmarks(
    pdf_path: str,
    force_rebuild: bool = False,
    use_toc: bool = True,
    use_font_heuristics: bool = True,
    inject_into_pdf: bool = False,
    output_path: Optional[str] = None,
    fuzzy_threshold: int = 70,
) -> ExtractionResult:
    """Main entry point: extract chapter structure from a PDF.

    Args:
        pdf_path: Path to the input PDF file.
        force_rebuild: If True, ignore existing bookmarks and rebuild.
        use_toc: Whether to try TOC page parsing (Approach 1).
        use_font_heuristics: Whether to use font-based heading detection (Approach 3).
        inject_into_pdf: If True, write bookmarks into the PDF.
        output_path: Path for the output PDF (if injecting). Defaults to overwriting input.
        fuzzy_threshold: Fuzzy match threshold for reconciliation (0-100).

    Returns:
        ExtractionResult with the extracted bookmark tree and metadata.
    """
    doc = open_pdf(pdf_path)
    result = ExtractionResult(pdf_path=pdf_path, total_pages=len(doc))

    try:
        # ── Step 1: Check for existing bookmarks ────────────────────────
        if not force_rebuild and has_bookmarks(doc):
            logger.info("PDF has existing bookmarks, using them directly.")
            existing = extract_existing_bookmarks(doc)
            result.bookmarks = existing
            result.method_used = "existing_bookmarks"
            return result

        if not force_rebuild and has_bookmarks(doc):
            logger.info("PDF has existing bookmarks but force_rebuild=True, rebuilding.")

        methods_used: list[str] = []

        # ── Step 2: Build page number mapping ───────────────────────────
        logger.info("Building page number mapping...")
        page_mapping = build_page_number_mapping(doc)
        result.page_mapping = page_mapping
        if page_mapping.confidence > 0:
            logger.info(
                f"Page mapping: offset={page_mapping.offset}, "
                f"confidence={page_mapping.confidence:.2f}"
            )
        else:
            result.warnings.append("Could not determine page number offset — using default mapping.")

        # ── Step 3: TOC page parsing ────────────────────────────────────
        toc_entries: list[BookmarkEntry] = []
        toc_page_indices: set[int] = set()

        if use_toc:
            logger.info("Searching for Table of Contents...")
            toc_location = find_toc_pages(doc)

            if toc_location:
                logger.info(
                    f"TOC found: pages {toc_location.start_page_index + 1}-"
                    f"{toc_location.end_page_index + 1} "
                    f'("{toc_location.heading_text}")'
                )
                toc_entries = parse_toc_entries(doc, toc_location)
                logger.info(f"Parsed {len(toc_entries)} TOC entries.")
                toc_page_indices = set(range(
                    toc_location.start_page_index,
                    toc_location.end_page_index + 1,
                ))
                methods_used.append("toc_parse")
            else:
                logger.info("No TOC page found.")
                result.warnings.append("No Table of Contents page detected.")

        # ── Step 4: Font-based heading detection ────────────────────────
        font_entries: list[BookmarkEntry] = []

        if use_font_heuristics:
            logger.info("Analyzing document font profile...")
            font_profile = build_font_profile(doc, skip_pages=toc_page_indices)
            logger.info(
                f"Body font: {font_profile.body_font_name} @ "
                f"{font_profile.body_font_size}pt, "
                f"{len(font_profile.heading_font_sizes)} heading sizes detected"
            )

            logger.info("Detecting headings by font analysis...")
            font_entries = detect_headings(
                doc, font_profile, skip_pages=toc_page_indices
            )
            logger.info(f"Detected {len(font_entries)} heading candidates.")
            methods_used.append("font_heuristics")

        # ── Step 5: Reconcile ───────────────────────────────────────────
        if toc_entries or font_entries:
            logger.info("Reconciling TOC entries with detected headings...")
            reconciled = reconcile_bookmarks(
                toc_entries, font_entries, page_mapping,
                fuzzy_threshold=fuzzy_threshold,
            )
            result.bookmarks = reconciled
            result.method_used = " + ".join(methods_used)
            logger.info(f"Final bookmark count: {len(result.flat_bookmarks())}")
        else:
            result.warnings.append("No bookmarks could be extracted from this PDF.")
            result.method_used = "none"

        # ── Step 6: Optionally inject bookmarks into PDF ────────────────
        if inject_into_pdf and result.bookmarks:
            logger.info("Injecting bookmarks into PDF...")
            inject_bookmarks(doc, result.flat_bookmarks())
            save_path = output_path or pdf_path
            if save_path == pdf_path:
                # Incremental save can fail with encryption changes;
                # fall back to full save into a temp file then replace.
                try:
                    doc.save(save_path, incremental=True, encryption=0)
                except Exception:
                    doc.save(save_path, encryption=0)
            else:
                doc.save(save_path, encryption=0)
            logger.info(f"Saved PDF with bookmarks to: {save_path}")

    finally:
        doc.close()

    return result
