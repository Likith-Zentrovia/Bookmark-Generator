"""Font-based heuristic heading detection.

Scans the body of the PDF (outside TOC pages) to detect headings
based on font size, weight, and text characteristics.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import fitz

from .models import BookmarkEntry
from .pdf_utils import TextLine, extract_page_lines


@dataclass
class FontProfile:
    """Statistical profile of fonts used in the document."""
    body_font_size: float = 0.0       # Most common font size (body text)
    body_font_name: str = ""
    heading_font_sizes: list[float] = None  # Font sizes larger than body
    min_heading_size: float = 0.0      # Smallest heading font size

    def __post_init__(self):
        if self.heading_font_sizes is None:
            self.heading_font_sizes = []


def build_font_profile(doc: fitz.Document, skip_pages: Optional[set[int]] = None) -> FontProfile:
    """Analyze the document to determine body vs heading font sizes.

    Samples pages from the document, counts font size occurrences,
    and identifies the body text font size. Anything significantly
    larger is considered a heading candidate.
    """
    if skip_pages is None:
        skip_pages = set()

    size_char_count: Counter = Counter()
    font_name_count: Counter = Counter()
    total_pages = len(doc)

    # Sample up to 30 pages spread across the document
    sample_indices = _get_sample_indices(total_pages, 30, skip_pages)

    for page_idx in sample_indices:
        page = doc[page_idx]
        lines = extract_page_lines(page, page_idx)
        for line in lines:
            text = line.text.strip()
            # Skip very short lines (likely headers/footers/page numbers)
            if len(text) < 20:
                continue
            for span in line.spans:
                char_count = len(span.text.strip())
                if char_count > 0:
                    # Round to nearest 0.5pt for clustering
                    rounded_size = round(span.font_size * 2) / 2
                    size_char_count[rounded_size] += char_count
                    font_name_count[span.font_name] += char_count

    if not size_char_count:
        return FontProfile()

    # Body font size = most common font size by character count
    body_size = size_char_count.most_common(1)[0][0]
    body_font = font_name_count.most_common(1)[0][0] if font_name_count else ""

    # Heading sizes = sizes larger than body by at least 1pt
    heading_sizes = sorted(
        [s for s in size_char_count if s > body_size + 0.5],
        reverse=True,
    )

    min_heading_size = min(heading_sizes) if heading_sizes else body_size + 2

    return FontProfile(
        body_font_size=body_size,
        body_font_name=body_font,
        heading_font_sizes=heading_sizes,
        min_heading_size=min_heading_size,
    )


def detect_headings(
    doc: fitz.Document,
    font_profile: FontProfile,
    skip_pages: Optional[set[int]] = None,
    start_page: int = 0,
    end_page: Optional[int] = None,
) -> list[BookmarkEntry]:
    """Detect headings in the document body using font heuristics.

    A line is considered a heading if:
    1. Its dominant font size >= min_heading_size
    2. It is short (< 120 chars, headings aren't paragraphs)
    3. It doesn't look like a header/footer/page number
    4. It's bold OR uses a heading-sized font
    """
    if skip_pages is None:
        skip_pages = set()
    if end_page is None:
        end_page = len(doc)

    entries: list[BookmarkEntry] = []
    body_size = font_profile.body_font_size

    if body_size <= 0:
        return entries

    for page_idx in range(start_page, min(end_page, len(doc))):
        if page_idx in skip_pages:
            continue

        page = doc[page_idx]
        lines = extract_page_lines(page, page_idx)
        page_height = page.rect.height

        for line in lines:
            text = line.text.strip()

            # ── Basic filters ───────────────────────────────────────────
            # Too short or too long for a heading
            if len(text) < 2 or len(text) > 150:
                continue

            # Skip lines in header/footer area
            y_center = (line.bbox[1] + line.bbox[3]) / 2
            if y_center < page_height * 0.05 or y_center > page_height * 0.95:
                continue

            # Skip lines that are just numbers (page numbers, etc.)
            if re.match(r'^[\d\s\-.|]+$', text):
                continue

            # ── Heading detection ───────────────────────────────────────
            dominant_size = line.dominant_font_size
            is_large = dominant_size >= font_profile.min_heading_size
            is_bold = line.is_mostly_bold
            is_much_larger = dominant_size >= body_size + 3

            # Must be either notably larger or bold+larger
            if not (is_large or (is_bold and dominant_size > body_size + 0.5)):
                continue

            # Additional checks to reduce false positives
            if _is_likely_noise(text):
                continue

            # ── Determine heading level from font size ──────────────────
            level = _size_to_level(dominant_size, font_profile)

            # Boost confidence if multiple signals agree
            confidence = 0.5
            if is_large:
                confidence += 0.2
            if is_bold:
                confidence += 0.15
            if is_much_larger:
                confidence += 0.15
            if _looks_like_chapter_title(text):
                confidence += 0.1
            confidence = min(confidence, 1.0)

            entries.append(BookmarkEntry(
                title=_clean_heading_text(text),
                page_number=page_idx + 1,  # Will be remapped later
                pdf_page_index=page_idx,
                level=level,
                confidence=confidence,
                source="font_heuristic",
            ))

    # Post-process: remove duplicate headings on the same page
    entries = _deduplicate_headings(entries)

    return entries


def _size_to_level(font_size: float, profile: FontProfile) -> int:
    """Convert a font size to a heading level based on the font profile."""
    if not profile.heading_font_sizes:
        return 1

    # Map heading sizes to levels: largest = level 1, etc.
    for i, hs in enumerate(profile.heading_font_sizes):
        if font_size >= hs - 0.3:
            return min(i + 1, 4)

    return min(len(profile.heading_font_sizes) + 1, 4)


def _looks_like_chapter_title(text: str) -> bool:
    """Check if text looks like a chapter/section title."""
    patterns = [
        r'^(?:chapter|part|section|unit|module|lesson|appendix)\s+',
        r'^\d+\.\s+',         # "1. Title"
        r'^\d+\.\d+\s+',      # "1.1 Title"
        r'^[IVXLC]+\.\s+',    # "IV. Title"
        r'^[A-Z]\.\s+',       # "A. Title"
    ]
    for p in patterns:
        if re.match(p, text, re.IGNORECASE):
            return True
    return False


def _is_likely_noise(text: str) -> bool:
    """Check if a line is likely noise (headers, footers, captions, etc.)."""
    noise_patterns = [
        r'^(?:figure|table|fig\.?|tab\.?)\s+\d+',  # Figure/Table captions
        r'^\(?\d+\)?\s*$',                          # Bare numbers
        r'^page\s+\d+',                              # "Page N"
        r'^\s*©',                                    # Copyright
        r'^\s*all\s+rights\s+reserved',              # Copyright text
        r'^\s*isbn',                                 # ISBN
    ]
    for p in noise_patterns:
        if re.match(p, text.strip(), re.IGNORECASE):
            return True
    return False


def _clean_heading_text(text: str) -> str:
    """Clean up heading text."""
    # Remove trailing dots/numbers that might be page numbers
    text = re.sub(r'\s*[.·…]+\s*\d+\s*$', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _deduplicate_headings(entries: list[BookmarkEntry]) -> list[BookmarkEntry]:
    """Remove duplicate headings that appear on the same page."""
    seen: dict[tuple[int, str], BookmarkEntry] = {}
    for entry in entries:
        key = (entry.pdf_page_index, entry.title.lower())
        if key not in seen or entry.confidence > seen[key].confidence:
            seen[key] = entry
    return list(seen.values())


def _get_sample_indices(total: int, sample_size: int, skip: set[int]) -> list[int]:
    """Get evenly-spaced sample page indices."""
    if total <= sample_size:
        return [i for i in range(total) if i not in skip]

    step = total / sample_size
    indices = []
    for i in range(sample_size):
        idx = int(i * step)
        if idx not in skip and idx < total:
            indices.append(idx)
    return indices
