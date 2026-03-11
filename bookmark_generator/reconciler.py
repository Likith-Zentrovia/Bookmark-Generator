"""Reconciliation engine: merges TOC-parsed entries with font-detected headings.

The reconciler fuzzy-matches TOC entries to headings found in the document body,
validates page numbers, and produces a final unified bookmark tree.
"""

from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz, process

from .models import BookmarkEntry, PageNumberMapping


def reconcile_bookmarks(
    toc_entries: list[BookmarkEntry],
    font_entries: list[BookmarkEntry],
    page_mapping: Optional[PageNumberMapping] = None,
    fuzzy_threshold: int = 70,
) -> list[BookmarkEntry]:
    """Reconcile TOC-parsed entries with font-heuristic-detected headings.

    Strategy:
    1. For each TOC entry, try to find a matching font-detected heading.
    2. If matched, use the font entry's pdf_page_index (more reliable) and
       boost confidence. Keep the TOC entry's title (usually cleaner).
    3. Unmatched TOC entries are kept with lower confidence.
    4. Unmatched font entries that look like major headings are also kept.
    5. Apply page number mapping to convert logical -> physical page indices.

    Args:
        toc_entries: Entries parsed from the TOC page.
        font_entries: Entries detected by font heuristics in the body.
        page_mapping: Optional mapping between logical and physical pages.
        fuzzy_threshold: Minimum fuzzy match score (0-100) to consider a match.

    Returns:
        Reconciled list of BookmarkEntry objects.
    """
    if not toc_entries and not font_entries:
        return []

    # If we only have one source, use it directly
    if not toc_entries:
        result = _resolve_page_indices(font_entries, page_mapping)
        return _build_hierarchy(result)

    if not font_entries:
        result = _resolve_page_indices(toc_entries, page_mapping)
        return _build_hierarchy(result)

    # ── Build the reconciled list ───────────────────────────────────────────
    reconciled: list[BookmarkEntry] = []
    used_font_indices: set[int] = set()

    # Prepare font entry titles for fuzzy matching
    font_titles = [_normalize_title(e.title) for e in font_entries]

    for toc_entry in toc_entries:
        toc_title_norm = _normalize_title(toc_entry.title)

        # Find the best matching font entry
        match_result = process.extractOne(
            toc_title_norm,
            font_titles,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=fuzzy_threshold,
        )

        if match_result is not None:
            matched_text, score, match_idx = match_result

            if match_idx not in used_font_indices:
                used_font_indices.add(match_idx)
                font_entry = font_entries[match_idx]

                # Merge: use TOC title (cleaner), font entry's page index (more precise)
                merged = BookmarkEntry(
                    title=toc_entry.title,
                    page_number=toc_entry.page_number,
                    pdf_page_index=font_entry.pdf_page_index,
                    level=toc_entry.level,  # TOC numbering gives better hierarchy
                    confidence=min(1.0, (toc_entry.confidence + font_entry.confidence) / 2 + 0.2),
                    source="reconciled",
                )
                reconciled.append(merged)
                continue

        # No match found — keep the TOC entry as-is
        toc_entry.confidence = max(0.3, toc_entry.confidence - 0.1)
        reconciled.append(toc_entry)

    # Add unmatched font entries that look significant
    for idx, font_entry in enumerate(font_entries):
        if idx not in used_font_indices and font_entry.confidence >= 0.7:
            font_entry.confidence *= 0.8  # Slight penalty for being unmatched
            reconciled.append(font_entry)

    # Sort by page position
    reconciled.sort(key=lambda e: (e.pdf_page_index if e.pdf_page_index >= 0 else e.page_number, -e.level))

    # Resolve page indices
    reconciled = _resolve_page_indices(reconciled, page_mapping)

    # Build hierarchy
    return _build_hierarchy(reconciled)


def _resolve_page_indices(
    entries: list[BookmarkEntry],
    page_mapping: Optional[PageNumberMapping],
) -> list[BookmarkEntry]:
    """Resolve logical page numbers to physical PDF page indices."""
    if page_mapping is None:
        # No mapping available — assume 1-based page numbers = physical + 1
        for entry in entries:
            if entry.pdf_page_index < 0:
                entry.pdf_page_index = entry.page_number - 1
        return entries

    for entry in entries:
        if entry.pdf_page_index < 0:
            entry.pdf_page_index = page_mapping.logical_to_physical(entry.page_number)

    return entries


def _build_hierarchy(
    flat_entries: list[BookmarkEntry],
    presorted: bool = False,
) -> list[BookmarkEntry]:
    """Convert a flat list of entries into a nested tree based on levels.

    Entries with level 1 are top-level. Level 2 entries become children
    of the preceding level 1, etc.

    Args:
        flat_entries: Flat list of bookmark entries.
        presorted: If True, entries are already in correct reading order
                   (e.g., from vision TOC extraction) and should NOT be
                   re-sorted. If False (default), entries are sorted by
                   page index before building hierarchy.
    """
    if not flat_entries:
        return []

    if not presorted:
        # Sort by page then by appearance order
        flat_entries.sort(key=lambda e: (
            e.pdf_page_index if e.pdf_page_index >= 0 else e.page_number * 1000
        ))

    root: list[BookmarkEntry] = []
    stack: list[BookmarkEntry] = []  # Stack of (entry) for nesting

    for entry in flat_entries:
        # Clear children (in case of re-processing)
        entry.children = []

        if not stack or entry.level <= 1:
            root.append(entry)
            stack = [entry]
        else:
            # Find the appropriate parent
            while len(stack) > 1 and stack[-1].level >= entry.level:
                stack.pop()

            stack[-1].children.append(entry)
            stack.append(entry)

    return root


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy comparison."""
    # Remove numbering prefixes
    title = re.sub(r'^(?:\d+\.)+\d*\.?\s*', '', title)
    # Remove chapter/section prefixes
    title = re.sub(
        r'^(?:chapter|part|section|unit|appendix)\s+[\dIVXLCivxlc]+[.:]\s*',
        '', title, flags=re.IGNORECASE,
    )
    # Normalize whitespace and case
    title = re.sub(r'\s+', ' ', title).strip().lower()
    return title
