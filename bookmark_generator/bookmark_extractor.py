"""Extract existing bookmarks (outline/TOC) from PDFs using PyMuPDF."""

from __future__ import annotations

import fitz

from .models import BookmarkEntry


def extract_existing_bookmarks(doc: fitz.Document) -> list[BookmarkEntry]:
    """Extract existing bookmarks/outline from a PDF document.

    PyMuPDF's get_toc() returns a list of [level, title, page_number] entries.
    Returns an empty list if the PDF has no bookmarks.
    """
    toc = doc.get_toc(simple=True)

    if not toc:
        return []

    entries: list[BookmarkEntry] = []
    for level, title, page_num in toc:
        # PyMuPDF returns 1-based page numbers in get_toc()
        # The page_num here is already the physical page number (1-based)
        entries.append(BookmarkEntry(
            title=title.strip(),
            page_number=page_num,
            pdf_page_index=page_num - 1,  # Convert to 0-based
            level=level,
            confidence=1.0,
            source="existing_bookmark",
        ))

    return entries


def has_bookmarks(doc: fitz.Document) -> bool:
    """Check if a PDF document has existing bookmarks."""
    toc = doc.get_toc(simple=True)
    return bool(toc)


def inject_bookmarks(doc: fitz.Document, bookmarks: list[BookmarkEntry]) -> None:
    """Write bookmark entries into the PDF document's outline.

    This replaces any existing bookmarks with the provided entries.
    Normalizes levels to satisfy PyMuPDF constraints:
    - First entry must be level 1.
    - No entry may jump more than 1 level deeper than its predecessor.
    """
    toc_entries: list[list] = []
    _flatten_to_toc(bookmarks, toc_entries)
    toc_entries = _normalize_toc_levels(toc_entries)
    if toc_entries:
        doc.set_toc(toc_entries)


def _flatten_to_toc(entries: list[BookmarkEntry], result: list) -> None:
    """Convert BookmarkEntry tree to PyMuPDF TOC format [level, title, page]."""
    for entry in entries:
        page = entry.pdf_page_index + 1 if entry.pdf_page_index >= 0 else entry.page_number
        page = max(1, page)  # PyMuPDF requires page >= 1
        result.append([entry.level, entry.title, page])
        _flatten_to_toc(entry.children, result)


def _normalize_toc_levels(toc: list[list]) -> list[list]:
    """Normalize TOC levels so PyMuPDF's set_toc() accepts them.

    Rules enforced:
    1. First entry is always level 1.
    2. No entry's level exceeds previous_level + 1.
    3. Minimum level across all entries is shifted to 1.
    """
    if not toc:
        return toc

    # Shift so the minimum level becomes 1
    min_level = min(entry[0] for entry in toc)
    if min_level != 1:
        shift = min_level - 1
        for entry in toc:
            entry[0] -= shift

    # Ensure first entry is level 1
    toc[0][0] = 1

    # Clamp so no entry jumps more than +1 from its predecessor
    for i in range(1, len(toc)):
        max_allowed = toc[i - 1][0] + 1
        if toc[i][0] > max_allowed:
            toc[i][0] = max_allowed

    return toc
