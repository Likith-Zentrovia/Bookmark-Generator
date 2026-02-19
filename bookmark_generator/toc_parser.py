"""TOC (Table of Contents) page detection and parsing.

This module handles:
1. Locating the TOC page(s) in a PDF
2. Parsing TOC entries (title + page number + hierarchy level)
3. Handling various TOC formats (dotted leaders, tab-separated, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import fitz

from .models import BookmarkEntry
from .pdf_utils import TextLine, extract_page_lines, extract_page_text


# ─── TOC heading detection keywords ────────────────────────────────────────────

TOC_HEADING_PATTERNS = [
    re.compile(r'^\s*table\s+of\s+contents?\s*$', re.IGNORECASE),
    re.compile(r'^\s*contents?\s*$', re.IGNORECASE),
    re.compile(r'^\s*t\.?\s*o\.?\s*c\.?\s*$', re.IGNORECASE),
    re.compile(r'^\s*index\s*$', re.IGNORECASE),
    re.compile(r'^\s*विषय\s*सूची\s*$', re.IGNORECASE),  # Hindi
    re.compile(r'^\s*विषय[-\s]*अनुक्रमणिका\s*$', re.IGNORECASE),  # Hindi
    re.compile(r'^\s*table\s+des\s+matières\s*$', re.IGNORECASE),  # French
    re.compile(r'^\s*índice\s*$', re.IGNORECASE),  # Spanish/Portuguese
    re.compile(r'^\s*inhaltsverzeichnis\s*$', re.IGNORECASE),  # German
    re.compile(r'^\s*目\s*次\s*$'),  # Japanese/Chinese
    re.compile(r'^\s*목\s*차\s*$'),  # Korean
]

# ─── TOC entry line patterns ───────────────────────────────────────────────────

# These patterns match lines like:
#   "Chapter 1: Introduction ......... 5"
#   "1.2.3  Some Heading    45"
#   "Introduction                     3"
#   "  A. Appendix Title         120"

TOC_ENTRY_PATTERNS = [
    # Pattern: Title followed by dots/dashes and page number
    # e.g., "Introduction ........... 3" or "Chapter 1 --- 12"
    re.compile(
        r'^(?P<indent>\s*)'
        r'(?P<title>.+?)'
        r'\s*[.\-·…_]{2,}\s*'
        r'(?P<page>\d+)\s*$'
    ),
    # Pattern: Title with large whitespace gap then page number
    # e.g., "Introduction                     3"
    re.compile(
        r'^(?P<indent>\s*)'
        r'(?P<title>.+?)'
        r'\s{3,}'
        r'(?P<page>\d+)\s*$'
    ),
    # Pattern: Numbered heading with page number at end
    # e.g., "1.2.3  Some Heading    45"
    re.compile(
        r'^(?P<indent>\s*)'
        r'(?P<title>(?:\d+\.)+\d*\s+.+?)'
        r'\s{2,}'
        r'(?P<page>\d+)\s*$'
    ),
    # Pattern: Title followed by tab and page number
    re.compile(
        r'^(?P<indent>\s*)'
        r'(?P<title>.+?)'
        r'\t+'
        r'(?P<page>\d+)\s*$'
    ),
]

# For detecting numbered headings and inferring hierarchy
NUMBERING_PATTERN = re.compile(
    r'^(?P<number>(?:\d+\.)*\d+\.?)\s+(?P<rest>.+)$'
)

# Chapter/Part prefixed entries
CHAPTER_PREFIX_PATTERN = re.compile(
    r'^(?:chapter|part|section|unit|module|lesson|appendix|annex)\s+',
    re.IGNORECASE,
)

# Letter-based entries (A., B., etc.)
LETTER_PREFIX_PATTERN = re.compile(
    r'^(?P<letter>[A-Z])\.\s+(?P<rest>.+)$'
)


@dataclass
class TOCLocation:
    """Location of the Table of Contents in the PDF."""
    start_page_index: int  # Physical page index where TOC starts
    end_page_index: int    # Physical page index where TOC ends (inclusive)
    heading_text: str      # The actual heading text found


def find_toc_pages(doc: fitz.Document, max_search_pages: int = 20) -> Optional[TOCLocation]:
    """Locate the TOC page(s) in the PDF.

    Searches the first `max_search_pages` pages for a TOC heading,
    then determines where the TOC ends.
    """
    total_pages = len(doc)
    search_limit = min(max_search_pages, total_pages)

    toc_start: Optional[int] = None
    toc_heading: str = ""

    for page_idx in range(search_limit):
        page = doc[page_idx]
        lines = extract_page_lines(page, page_idx)

        for line in lines:
            text = line.text.strip()
            if not text:
                continue

            for pattern in TOC_HEADING_PATTERNS:
                if pattern.match(text):
                    toc_start = page_idx
                    toc_heading = text
                    break

            if toc_start is not None:
                break

        if toc_start is not None:
            break

    if toc_start is None:
        return None

    # Determine where the TOC ends
    # The TOC typically spans consecutive pages with entries matching TOC patterns.
    toc_end = toc_start

    for page_idx in range(toc_start, min(toc_start + 15, total_pages)):
        page = doc[page_idx]
        text = extract_page_text(page)
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        if page_idx == toc_start:
            # Skip the heading itself on the first page
            continue

        # Count how many lines look like TOC entries
        toc_line_count = 0
        total_content_lines = 0
        for line_text in lines:
            if len(line_text) < 3:
                continue
            total_content_lines += 1
            if _is_toc_entry_line(line_text):
                toc_line_count += 1

        # If more than 30% of content lines are TOC entries, this is still a TOC page
        if total_content_lines > 0 and (toc_line_count / total_content_lines) > 0.3:
            toc_end = page_idx
        else:
            break

    return TOCLocation(
        start_page_index=toc_start,
        end_page_index=toc_end,
        heading_text=toc_heading,
    )


def _is_toc_entry_line(text: str) -> bool:
    """Quick check if a line looks like a TOC entry."""
    for pattern in TOC_ENTRY_PATTERNS:
        if pattern.match(text):
            return True
    # Also check for lines ending with a number (common in TOCs)
    if re.search(r'\d+\s*$', text) and len(text) > 5:
        return True
    return False


def parse_toc_entries(
    doc: fitz.Document,
    toc_location: TOCLocation,
) -> list[BookmarkEntry]:
    """Parse TOC entries from the identified TOC pages.

    Returns a list of BookmarkEntry objects with titles, page numbers,
    and inferred hierarchy levels.
    """
    entries: list[BookmarkEntry] = []
    raw_entries: list[dict] = []

    for page_idx in range(toc_location.start_page_index, toc_location.end_page_index + 1):
        page = doc[page_idx]

        # Try structured extraction first (using font metadata)
        structured_entries = _parse_toc_page_structured(page, page_idx)
        if structured_entries:
            raw_entries.extend(structured_entries)
        else:
            # Fall back to plain text parsing
            text_entries = _parse_toc_page_text(page, page_idx)
            raw_entries.extend(text_entries)

    # Filter out the TOC heading itself
    raw_entries = [
        e for e in raw_entries
        if not _is_toc_heading(e["title"])
    ]

    # Deduplicate (same title + same page)
    seen = set()
    unique_entries = []
    for entry in raw_entries:
        key = (entry["title"].strip().lower(), entry["page"])
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    raw_entries = unique_entries

    # Infer hierarchy levels
    _assign_hierarchy_levels(raw_entries)

    # Convert to BookmarkEntry objects
    for raw in raw_entries:
        entries.append(BookmarkEntry(
            title=raw["title"].strip(),
            page_number=raw["page"],
            level=raw.get("level", 1),
            confidence=raw.get("confidence", 0.8),
            source="toc_parse",
        ))

    return entries


def _parse_toc_page_structured(page: fitz.Page, page_idx: int) -> list[dict]:
    """Parse TOC entries using font metadata for better accuracy.

    Uses text line positions to detect entries where the page number
    is right-aligned and the title is left-aligned.
    """
    lines = extract_page_lines(page, page_idx)
    entries: list[dict] = []

    page_width = page.rect.width

    for line in lines:
        text = line.text.strip()
        if not text or len(text) < 3:
            continue

        # Skip the TOC heading
        if _is_toc_heading(text):
            continue

        # Try to parse with regex patterns first
        match = _match_toc_entry(text)
        if match:
            indent = len(match["indent"]) if match.get("indent") else 0
            entries.append({
                "title": match["title"].strip(),
                "page": int(match["page"]),
                "indent": indent,
                "font_size": line.dominant_font_size,
                "is_bold": line.is_mostly_bold,
                "x_offset": line.bbox[0],
                "confidence": 0.9,
            })
            continue

        # Try to detect right-aligned page numbers by looking at span positions
        # In many TOCs, the page number is a separate span far to the right
        if len(line.spans) >= 2:
            last_span = line.spans[-1]
            last_text = last_span.text.strip()

            # Check if the last span is a number aligned to the right
            if re.match(r'^\d+$', last_text) and last_span.bbox[0] > page_width * 0.7:
                title_text = "".join(s.text for s in line.spans[:-1]).strip()
                # Clean up dot leaders from the title
                title_text = re.sub(r'[.\-·…_]+\s*$', '', title_text).strip()
                if title_text and len(title_text) > 1:
                    entries.append({
                        "title": title_text,
                        "page": int(last_text),
                        "indent": 0,
                        "font_size": line.dominant_font_size,
                        "is_bold": line.is_mostly_bold,
                        "x_offset": line.bbox[0],
                        "confidence": 0.85,
                    })

    return entries


def _parse_toc_page_text(page: fitz.Page, page_idx: int) -> list[dict]:
    """Parse TOC entries from plain text as a fallback."""
    text = extract_page_text(page)
    entries: list[dict] = []

    for line_text in text.split('\n'):
        line_text_stripped = line_text.strip()
        if not line_text_stripped or len(line_text_stripped) < 3:
            continue

        if _is_toc_heading(line_text_stripped):
            continue

        match = _match_toc_entry(line_text)
        if match:
            indent_len = len(line_text) - len(line_text.lstrip())
            entries.append({
                "title": match["title"].strip(),
                "page": int(match["page"]),
                "indent": indent_len,
                "font_size": 0,
                "is_bold": False,
                "x_offset": indent_len,
                "confidence": 0.7,
            })

    return entries


def _match_toc_entry(text: str) -> Optional[dict]:
    """Try all TOC entry patterns against a line of text."""
    for pattern in TOC_ENTRY_PATTERNS:
        m = pattern.match(text)
        if m:
            title = m.group("title").strip()
            page_str = m.group("page")
            indent = m.group("indent") if "indent" in m.groupdict() else ""

            # Validate: title should not be just numbers/symbols
            if not re.search(r'[a-zA-Z\u0900-\u097F\u4e00-\u9fff\uac00-\ud7af]', title):
                continue

            # Validate: page number should be reasonable
            try:
                page_num = int(page_str)
                if page_num < 0 or page_num > 9999:
                    continue
            except ValueError:
                continue

            return {
                "title": title,
                "page": page_num,
                "indent": indent,
            }
    return None


def _is_toc_heading(text: str) -> bool:
    """Check if text is a TOC heading (not an entry)."""
    text = text.strip()
    for pattern in TOC_HEADING_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _assign_hierarchy_levels(entries: list[dict]) -> None:
    """Infer hierarchy levels for TOC entries.

    Uses multiple signals:
    1. Numbering scheme (1 vs 1.1 vs 1.1.1)
    2. Indentation
    3. Font size
    4. Bold/non-bold
    5. Chapter/Part prefixes
    """
    if not entries:
        return

    # ── Signal 1: Numbering scheme ──────────────────────────────────────
    for entry in entries:
        title = entry["title"]

        # Check for "Chapter X" or "Part X" prefix -> level 1
        if CHAPTER_PREFIX_PATTERN.match(title):
            entry["_numbering_level"] = 1
            continue

        # Check for numbered headings: "1", "1.1", "1.1.1", etc.
        num_match = NUMBERING_PATTERN.match(title)
        if num_match:
            number_str = num_match.group("number").rstrip('.')
            depth = number_str.count('.') + 1
            entry["_numbering_level"] = min(depth, 4)
            continue

        # Check for letter prefix: "A. Title"
        letter_match = LETTER_PREFIX_PATTERN.match(title)
        if letter_match:
            entry["_numbering_level"] = 1
            continue

        entry["_numbering_level"] = None

    # ── Signal 2: Indentation clustering ────────────────────────────────
    x_offsets = sorted(set(e.get("x_offset", 0) for e in entries))
    indent_levels = {x: i + 1 for i, x in enumerate(x_offsets[:5])}

    # ── Signal 3: Font size clustering ──────────────────────────────────
    font_sizes = sorted(set(e.get("font_size", 0) for e in entries if e.get("font_size", 0) > 0), reverse=True)
    size_levels = {s: i + 1 for i, s in enumerate(font_sizes[:5])}

    # ── Combine signals ─────────────────────────────────────────────────
    for entry in entries:
        signals: list[int] = []

        # Numbering level (strongest signal)
        if entry.get("_numbering_level") is not None:
            signals.append(entry["_numbering_level"])
            signals.append(entry["_numbering_level"])  # Double-weight

        # Indentation level
        x_off = entry.get("x_offset", 0)
        if x_off in indent_levels:
            signals.append(indent_levels[x_off])

        # Font size level
        fsize = entry.get("font_size", 0)
        if fsize > 0 and fsize in size_levels:
            signals.append(size_levels[fsize])

        # Bold -> likely higher level
        if entry.get("is_bold"):
            signals.append(1)

        if signals:
            # Use the median signal as the level
            signals.sort()
            entry["level"] = signals[len(signals) // 2]
        else:
            entry["level"] = 1

        # Clean up temporary keys
        entry.pop("_numbering_level", None)
