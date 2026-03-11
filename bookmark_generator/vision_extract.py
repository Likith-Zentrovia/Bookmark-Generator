"""Vision-based TOC extraction and cross-verification.

Uses Claude Vision API to:
1. Discover TOC pages (hybrid: regex first, vision fallback)
2. Extract all TOC entries from rendered page images
3. Cross-verify each entry by checking if the heading exists on its claimed page

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY in .env file or environment variable
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

# Type alias for progress callbacks
# callback(phase, step, total_steps, message)
ProgressCallback = Callable[[str, int, int, str], None]

# Load .env file from project root
try:
    from dotenv import load_dotenv
    # Search upward from this file to find .env
    _project_root = Path(__file__).resolve().parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

from .models import BookmarkEntry, PageNumberMapping
from .pdf_utils import render_page_to_png

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Page image cache
# ═══════════════════════════════════════════════════════════════════════════════

class _PageImageCache:
    """Cache rendered page images to avoid re-rendering the same page."""

    def __init__(self, doc, render_width: int = 1400):
        self._doc = doc
        self._width = render_width
        self._cache: dict[int, bytes] = {}

    def get(self, page_index: int) -> bytes:
        if page_index not in self._cache:
            page = self._doc[page_index]
            self._cache[page_index] = render_page_to_png(page, self._width)
        return self._cache[page_index]


# ═══════════════════════════════════════════════════════════════════════════════
#  Vision API caller
# ═══════════════════════════════════════════════════════════════════════════════

def call_vision_api(
    prompt: str,
    images: list[bytes],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16384,
) -> Optional[str]:
    """Call Claude Vision API with one or more page images.

    Args:
        prompt: Text prompt describing what to extract.
        images: List of PNG image bytes.
        model: Anthropic model ID (must support vision).
        max_tokens: Maximum response tokens.

    Returns:
        Response text, or None on failure.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "'anthropic' package not installed. Install with: pip install anthropic  "
            "Vision extraction requires the anthropic package."
        )
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Vision extraction requires an API key. "
            "Set it with: export ANTHROPIC_API_KEY=sk-..."
        )
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build multi-modal content: images first, then text prompt
        content = []
        for img_bytes in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img_bytes).decode("utf-8"),
                },
            })
        content.append({"type": "text", "text": prompt})

        logger.info(f"Calling Vision API ({model}) with {len(images)} image(s)...")
        start = time.time()

        # Log total image payload size for debugging
        total_size = sum(len(img) for img in images)
        logger.info(f"Total image payload: {total_size / 1024:.0f} KB across {len(images)} image(s)")

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            timeout=120.0,
            messages=[{"role": "user", "content": content}],
        )

        elapsed = time.time() - start
        logger.info(f"Vision API responded in {elapsed:.1f}s")

        result = ""
        for block in response.content:
            if hasattr(block, "text"):
                result += block.text
        return result

    except Exception as e:
        logger.warning(f"Vision API call failed: {e}")
        return None


def _parse_json_response(response_text: str) -> Optional[list[dict]]:
    """Parse a JSON array response, handling common LLM quirks."""
    if not response_text:
        return None

    text = response_text.strip()

    # Extract JSON from markdown code fences (handles ```json ... ``` and ``` ... ```)
    # This pattern handles text before/after the code fence
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    elif text.startswith("```"):
        # Fallback: strip leading/trailing fences line by line
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # If text doesn't start with '[', try to find the JSON array in it
    if not text.startswith("["):
        bracket_idx = text.find("[")
        if bracket_idx >= 0:
            text = text[bracket_idx:]

    # Fix trailing commas
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        logger.warning(f"Vision API returned {type(data).__name__} instead of list")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Vision JSON: {e}. First 300 chars: {text[:300]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Roman numeral helper
# ═══════════════════════════════════════════════════════════════════════════════

def _roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to an integer. Returns 0 on failure."""
    roman_values = {
        "i": 1, "v": 5, "x": 10, "l": 50,
        "c": 100, "d": 500, "m": 1000,
    }
    s = s.lower().strip()
    if not s or not all(c in roman_values for c in s):
        return 0
    result = 0
    for i, char in enumerate(s):
        val = roman_values[char]
        if i + 1 < len(s) and val < roman_values.get(s[i + 1], 0):
            result -= val
        else:
            result += val
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1: TOC page discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_toc_pages(
    doc,
    model: str = "claude-sonnet-4-20250514",
    max_search_pages: int = 20,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[int]:
    """Find TOC page indices using hybrid regex + vision approach.

    Strategy:
    1. Try existing regex-based detection.
    2. If that fails, render pages and ask Vision API to identify TOC pages.

    Returns:
        Sorted list of 0-based page indices containing TOC content.
    """
    # Step 1: Try regex detection
    from .toc_parser import find_toc_pages

    if progress_callback:
        progress_callback("discovery", 0, 1, "Searching for TOC pages (regex)...")

    toc_location = find_toc_pages(doc, max_search_pages=max_search_pages)
    if toc_location:
        pages = list(range(toc_location.start_page_index, toc_location.end_page_index + 1))
        logger.info(
            f"Regex found TOC on pages {[p + 1 for p in pages]} "
            f'("{toc_location.heading_text}")'
        )
        if progress_callback:
            progress_callback("discovery", 1, 1, f"Found TOC on pages {[p + 1 for p in pages]}")
        return pages

    # Step 2: Vision fallback — scan pages in batches
    logger.info("Regex TOC detection failed. Using Vision API to find TOC pages...")
    total_pages = len(doc)
    search_limit = min(max_search_pages, total_pages)
    cache = _PageImageCache(doc, render_width=800)

    toc_pages: list[int] = []
    batch_size = 5

    total_batches = (search_limit + batch_size - 1) // batch_size
    for batch_num, batch_start in enumerate(range(0, search_limit, batch_size)):
        batch_end = min(batch_start + batch_size, search_limit)
        page_indices = list(range(batch_start, batch_end))
        images = [cache.get(idx) for idx in page_indices]

        if progress_callback:
            progress_callback(
                "discovery", batch_num + 1, total_batches,
                f"Scanning pages {batch_start + 1}-{batch_end} for TOC..."
            )

        page_labels = ", ".join(str(i + 1) for i in page_indices)
        prompt = (
            f"Look at these PDF pages (pages {page_labels}).\n\n"
            "Which of these pages contain a Table of Contents, Index, or a listing "
            "of chapters/sections with page numbers? This could be in any language "
            "and might be titled differently (e.g., 'Contents', 'Índice', 'Inhaltsverzeichnis', "
            "'目次', 'विषय सूची', or simply have no title but show a list of chapters with page numbers).\n\n"
            "Respond with ONLY a JSON array of 1-indexed page numbers that contain "
            "TOC/Index content. If none of these pages contain a TOC, respond with [].\n\n"
            "Example: [3, 4] means pages 3 and 4 contain TOC content."
        )

        response = call_vision_api(prompt, images, model=model, max_tokens=256)
        if response:
            parsed = _parse_json_response(response)
            if parsed and isinstance(parsed, list):
                for page_num in parsed:
                    if isinstance(page_num, (int, float)):
                        idx = int(page_num) - 1  # Convert 1-indexed to 0-indexed
                        if 0 <= idx < total_pages:
                            toc_pages.append(idx)

        # Stop searching if we found TOC pages and moved past them
        if toc_pages and batch_start > max(toc_pages) + batch_size:
            break

    toc_pages = sorted(set(toc_pages))
    if toc_pages:
        logger.info(f"Vision found TOC on pages {[p + 1 for p in toc_pages]}")
    else:
        logger.info("Vision did not find any TOC pages.")

    return toc_pages


# ═══════════════════════════════════════════════════════════════════════════════
#  TOC page header filter
# ═══════════════════════════════════════════════════════════════════════════════

# Titles that are headings OF the TOC page itself, not actual entries
_TOC_PAGE_HEADERS = {
    "contents", "table of contents", "index", "toc",
    "índice", "inhaltsverzeichnis", "目次", "sommaire",
    "inhoud", "contenido", "indice",
}


def _is_toc_page_header(title: str) -> bool:
    """Check if a title is a TOC page heading rather than an actual entry."""
    normalized = title.strip().lower()
    # Exact match
    if normalized in _TOC_PAGE_HEADERS:
        return True
    # "Contents (continued)" or "Contents (cont'd)"
    if any(normalized.startswith(h) and "cont" in normalized for h in _TOC_PAGE_HEADERS):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: TOC entry extraction via Vision
# ═══════════════════════════════════════════════════════════════════════════════

def extract_toc_entries_vision(
    doc,
    toc_page_indices: list[int],
    model: str = "claude-sonnet-4-20250514",
    render_width: int = 1000,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[BookmarkEntry]:
    """Extract TOC entries from rendered TOC page images using Vision API.

    Sends TOC page images to Claude Vision and gets structured entries.
    Uses 1000px width (sufficient for TOC text readability) to keep payload small.

    Returns:
        List of BookmarkEntry objects extracted from the TOC.
    """
    if not toc_page_indices:
        return []

    cache = _PageImageCache(doc, render_width=render_width)
    entries: list[BookmarkEntry] = []

    # Batch TOC pages (up to 3 per API call to keep payload manageable)
    batch_size = 3
    total_batches = (len(toc_page_indices) + batch_size - 1) // batch_size
    for batch_num, batch_start in enumerate(range(0, len(toc_page_indices), batch_size)):
        batch_indices = toc_page_indices[batch_start:batch_start + batch_size]
        images = [cache.get(idx) for idx in batch_indices]
        page_labels = ", ".join(str(i + 1) for i in batch_indices)
        logger.info(f"TOC extraction batch {batch_num + 1}/{total_batches}: pages {page_labels}")
        if progress_callback:
            progress_callback(
                "extraction", batch_num + 1, total_batches,
                f"Reading TOC pages {page_labels}..."
            )

        prompt = (
            f"You are examining Table of Contents / Index pages from a PDF document. "
            f"These are pages {page_labels} (1-indexed).\n\n"
            "Extract EVERY entry from these TOC pages. For each entry, provide:\n"
            '- "title": The exact title text as shown (combine multi-line titles into one)\n'
            '- "page_number": The page number listed next to this entry (as an integer; '
            "convert Roman numerals like 'iv' to 4, 'xii' to 12, etc.)\n"
            '- "level": Hierarchy level (1=chapter/part, 2=section, 3=subsection, 4=sub-subsection)\n\n'
            "Use indentation, numbering patterns (1, 1.1, 1.1.1), font size differences, "
            "bold vs regular text, and structural cues to determine the hierarchy level.\n\n"
            "IMPORTANT RULES:\n"
            "- Include ALL entries, even appendices, bibliography, index, glossary, etc.\n"
            "- If an entry spans multiple lines, combine them into a single title\n"
            "- Preserve the exact spelling, capitalization, and numbering of titles\n"
            "- Page numbers must be integers\n"
            "- Levels must be 1-4, no jumps greater than 1 from previous entry\n"
            "- Do NOT include page headings like 'Contents', 'Table of Contents', 'Index', etc. "
            "that are titles OF the TOC page itself (not entries that point to other pages)\n\n"
            "Respond ONLY with a JSON array, no other text:\n"
            '[{"title": "Chapter 1: Introduction", "page_number": 1, "level": 1}, ...]'
        )

        response = call_vision_api(prompt, images, model=model)
        if not response:
            logger.warning(f"Vision API returned no response for TOC pages {page_labels}")
            continue

        parsed = _parse_json_response(response)
        if not parsed:
            logger.warning(f"Failed to parse Vision response for TOC pages {page_labels}")
            continue

        for item in parsed:
            title = item.get("title", "").strip()
            if not title:
                continue

            # Skip TOC page headers (these are headings OF the TOC page, not entries)
            if _is_toc_page_header(title):
                logger.debug(f"Skipping TOC page header: {title}")
                continue

            page_num_raw = item.get("page_number", 0)
            try:
                page_num = int(page_num_raw)
            except (ValueError, TypeError):
                # Try Roman numeral conversion
                if isinstance(page_num_raw, str):
                    page_num = _roman_to_int(page_num_raw)
                else:
                    page_num = 0

            if page_num <= 0:
                logger.debug(f"Skipping entry with invalid page number: {title}")
                continue

            level = item.get("level", 1)
            level = max(1, min(int(level), 4))

            entries.append(BookmarkEntry(
                title=title,
                page_number=page_num,
                pdf_page_index=-1,  # Will be resolved via page mapping
                level=level,
                confidence=0.9,
                source="vision_toc",
            ))

    logger.info(f"Vision extracted {len(entries)} TOC entries.")
    if progress_callback:
        progress_callback("extraction", total_batches, total_batches, f"Extracted {len(entries)} entries from TOC")
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Cross-verification
# ═══════════════════════════════════════════════════════════════════════════════

def cross_verify_entries(
    doc,
    entries: list[BookmarkEntry],
    page_mapping: PageNumberMapping,
    model: str = "claude-sonnet-4-20250514",
    render_width: int = 1000,
    batch_size: int = 5,
    nearby_range: int = 2,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[BookmarkEntry]:
    """Verify each TOC entry by checking if the heading exists on its claimed page.

    For each entry, renders the target page (and nearby pages) and asks
    the Vision API to confirm the heading is present.

    Returns:
        Verified list of BookmarkEntry objects with corrected pages/levels.
    """
    if not entries:
        return []

    total_pages = len(doc)
    cache = _PageImageCache(doc, render_width=render_width)

    # Resolve page indices first
    for entry in entries:
        if entry.pdf_page_index < 0:
            entry.pdf_page_index = page_mapping.logical_to_physical(entry.page_number)
        # Clamp to valid range
        entry.pdf_page_index = max(0, min(entry.pdf_page_index, total_pages - 1))

    # Create index-ordered list for batching by page proximity,
    # but track original TOC order so we can restore it later.
    indexed_entries = list(enumerate(entries))  # (original_index, entry)
    indexed_entries.sort(key=lambda x: x[1].pdf_page_index)
    verified_entries: list[tuple[int, BookmarkEntry]] = []

    total_verify_batches = (len(indexed_entries) + batch_size - 1) // batch_size
    for batch_num, batch_start in enumerate(range(0, len(indexed_entries), batch_size)):
        batch = indexed_entries[batch_start:batch_start + batch_size]

        if progress_callback:
            entries_done = min(batch_start + batch_size, len(indexed_entries))
            progress_callback(
                "verification", batch_num + 1, total_verify_batches,
                f"Verifying entries {batch_start + 1}-{entries_done} of {len(indexed_entries)}..."
            )

        # Collect all pages needed for this batch (target + nearby)
        needed_pages: set[int] = set()
        for _orig_idx, entry in batch:
            target = entry.pdf_page_index
            for offset in range(-nearby_range, nearby_range + 1):
                p = target + offset
                if 0 <= p < total_pages:
                    needed_pages.add(p)

        needed_pages_sorted = sorted(needed_pages)
        images = [cache.get(idx) for idx in needed_pages_sorted]
        page_labels = ", ".join(str(p + 1) for p in needed_pages_sorted)

        # Build entry list for the prompt
        entry_descriptions = []
        for i, (_orig_idx, entry) in enumerate(batch):
            entry_descriptions.append(
                f'{i + 1}. "{entry.title}" — claimed to be on page {entry.page_number} '
                f"(PDF page {entry.pdf_page_index + 1})"
            )
        entries_text = "\n".join(entry_descriptions)

        prompt = (
            "I extracted these entries from a Table of Contents. I need you to verify "
            "each one actually appears as a heading on its claimed page.\n\n"
            f"Entries to verify:\n{entries_text}\n\n"
            f"The attached images are PDF pages {page_labels} (1-indexed).\n\n"
            "For EACH entry, check:\n"
            "1. Does this heading/title appear on the claimed page (or within "
            f"±{nearby_range} pages) as a heading or section title?\n"
            "2. If found on a different page than claimed, what is the correct page number?\n\n"
            "Respond ONLY with a JSON array, one object per entry:\n"
            "[\n"
            "  {\n"
            '    "title": "the title from the TOC (unchanged)",\n'
            '    "verified": true/false,\n'
            '    "actual_page": <1-indexed page where heading was found, or null if not found>\n'
            "  }\n"
            "]"
        )

        response = call_vision_api(prompt, images, model=model)
        if not response:
            # API failed — keep entries as-is with reduced confidence
            logger.warning(f"Verification API call failed for batch starting at index {batch_start}")
            for orig_idx, entry in batch:
                entry.confidence = max(entry.confidence - 0.1, 0.5)
            verified_entries.extend(batch)
            continue

        parsed = _parse_json_response(response)
        if not parsed:
            logger.warning(f"Failed to parse verification response for batch starting at index {batch_start}")
            for orig_idx, entry in batch:
                entry.confidence = max(entry.confidence - 0.1, 0.5)
            verified_entries.extend(batch)
            continue

        # Process verification results
        for i, (orig_idx, entry) in enumerate(batch):
            if i >= len(parsed):
                # LLM returned fewer results than entries — keep as-is
                entry.confidence = max(entry.confidence - 0.1, 0.5)
                verified_entries.append((orig_idx, entry))
                continue

            result = parsed[i]
            is_verified = result.get("verified", False)
            actual_page = result.get("actual_page")

            if is_verified:
                if actual_page is not None:
                    actual_page_idx = int(actual_page) - 1
                    if actual_page_idx == entry.pdf_page_index:
                        # Verified on correct page
                        entry.confidence = 0.95
                    else:
                        # Verified but on a different page — correct the page
                        entry.pdf_page_index = max(0, min(actual_page_idx, total_pages - 1))
                        entry.page_number = int(actual_page)
                        entry.confidence = 0.85
                        logger.debug(
                            f'Entry "{entry.title}" corrected to page {actual_page}'
                        )
                else:
                    entry.confidence = 0.9

                # NOTE: We do NOT update level or title from verification.
                # The TOC's hierarchy and exact title text are the authoritative source.
                # Verification only confirms page correctness.
                entry.source = "vision_verified"
            else:
                # Not verified — keep but with low confidence
                entry.confidence = 0.4
                entry.source = "vision_unverified"
                logger.debug(f'Entry "{entry.title}" could not be verified on any nearby page')

            verified_entries.append((orig_idx, entry))

    # Restore original TOC order (critical for correct hierarchy building)
    verified_entries.sort(key=lambda x: x[0])
    result_entries = [entry for _idx, entry in verified_entries]

    verified_count = sum(1 for e in result_entries if e.source == "vision_verified")
    unverified_count = sum(1 for e in result_entries if e.source == "vision_unverified")
    logger.info(
        f"Cross-verification complete: {verified_count} verified, "
        f"{unverified_count} unverified out of {len(result_entries)} total"
    )

    return result_entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def vision_extract_bookmarks(
    doc,
    page_mapping: PageNumberMapping,
    model: str = "claude-sonnet-4-20250514",
    render_width: int = 1000,
    verify: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[list[BookmarkEntry]]:
    """Vision-based TOC extraction with optional cross-verification.

    Orchestrates:
    1. Discover TOC pages (hybrid regex + vision)
    2. Extract all entries from TOC page images
    3. Optionally cross-verify entries against target pages

    Args:
        doc: Open PyMuPDF document.
        page_mapping: Logical-to-physical page mapping.
        model: Vision-capable Anthropic model to use.
        render_width: Width to render pages at (px).
        verify: Whether to run cross-verification.

    Returns:
        List of BookmarkEntry objects, or None on failure.
    """
    # Phase 1: Discover TOC pages
    logger.info("Phase 1: Discovering TOC pages...")
    if progress_callback:
        progress_callback("discovery", 0, 1, "Discovering TOC pages...")
    toc_pages = discover_toc_pages(doc, model=model, progress_callback=progress_callback)

    if not toc_pages:
        logger.warning("No TOC pages found by either regex or vision.")
        return None

    # Phase 2: Extract entries from TOC page images
    logger.info("Phase 2: Extracting TOC entries via Vision...")
    if progress_callback:
        progress_callback("extraction", 0, 1, "Extracting entries from TOC images...")
    entries = extract_toc_entries_vision(
        doc, toc_pages, model=model, render_width=render_width,
        progress_callback=progress_callback,
    )

    if not entries:
        logger.warning("Vision extracted 0 entries from TOC pages.")
        return None

    # Phase 3: Cross-verify entries (only if the PDF has enough content pages)
    if verify:
        total_pages = len(doc)
        # Check how many entries reference pages beyond the PDF's actual page count
        out_of_range = sum(1 for e in entries if e.page_number > total_pages)
        out_of_range_pct = out_of_range / len(entries) * 100 if entries else 0

        if out_of_range_pct > 50:
            logger.info(
                f"Skipping Phase 3 (cross-verification): {out_of_range}/{len(entries)} entries "
                f"({out_of_range_pct:.0f}%) reference pages beyond the PDF's {total_pages} pages. "
                f"This PDF appears to be a TOC-only extract without full content."
            )
            # Resolve page indices anyway (clamped to valid range)
            for entry in entries:
                if entry.pdf_page_index < 0:
                    entry.pdf_page_index = page_mapping.logical_to_physical(entry.page_number)
                entry.pdf_page_index = max(0, min(entry.pdf_page_index, total_pages - 1))
        else:
            logger.info("Phase 3: Cross-verifying entries against target pages...")
            if progress_callback:
                progress_callback("verification", 0, 1, "Starting cross-verification...")
            entries = cross_verify_entries(
                doc, entries, page_mapping, model=model, render_width=render_width,
                progress_callback=progress_callback,
            )

    # Fix chapter/section levels (Vision API loses context across batches)
    if progress_callback:
        progress_callback("finalizing", 1, 2, "Building bookmark hierarchy...")
    entries = _fix_chapter_section_levels(entries)

    # Normalize hierarchy levels — ensure no jumps > 1
    entries = _normalize_levels(entries)

    if progress_callback:
        progress_callback("finalizing", 2, 2, f"Done! {len(entries)} bookmarks ready.")

    return entries


def _fix_chapter_section_levels(entries: list[BookmarkEntry]) -> list[BookmarkEntry]:
    """Fix chapter vs section level assignments from Vision API.

    The Vision API processes TOC pages in batches and sometimes loses context
    about the hierarchy across batch boundaries, causing numbered chapters
    to be assigned level 1 instead of level 2.

    This function detects numbered chapters and ensures they're at level 2
    (under their section), then cascades the fix to child entries.
    """
    if not entries:
        return entries

    # Detect if document uses section/chapter structure
    # by checking for entries matching "Section [I/V/X...]" pattern
    section_pattern = re.compile(
        r"^Section\s+[IVXLC]+\b", re.IGNORECASE
    )
    # Chapter pattern: starts with a number (e.g., "1 TITLE", "19 LIVER")
    chapter_pattern = re.compile(r"^\d+\s+[A-Z]")

    has_sections = any(section_pattern.match(e.title) for e in entries)
    if not has_sections:
        return entries  # No sections detected, skip fix

    # Collect all chapter numbers that appear in the entries
    chapter_entries = [(i, e) for i, e in enumerate(entries) if chapter_pattern.match(e.title)]
    if not chapter_entries:
        return entries

    # Check if any numbered chapters are misassigned to L1
    misassigned = [
        (i, e) for i, e in chapter_entries
        if e.level == 1 and not section_pattern.match(e.title)
    ]
    if not misassigned:
        return entries  # All chapters correctly assigned

    logger.info(
        f"Fixing {len(misassigned)} chapter entries misassigned to level 1 "
        f"(should be level 2)"
    )

    # Fix: bump misassigned chapters from L1 to L2, and cascade to children
    for idx, entry in misassigned:
        level_shift = entry.level - 2  # How much to shift (negative = bump down)
        entry.level = 2

        # Cascade: shift all following entries until the next L1 or L2 chapter/section
        for j in range(idx + 1, len(entries)):
            next_entry = entries[j]
            # Stop cascading at the next section heading or at any entry
            # that was originally L1 or L2 (these need their own evaluation)
            if next_entry.level <= 2 and (
                section_pattern.match(next_entry.title)
                or chapter_pattern.match(next_entry.title)
                or next_entry.level == 1
            ):
                break
            # Only shift entries that were under this chapter
            if level_shift != 0:
                next_entry.level = max(1, next_entry.level - level_shift)

    # Second pass: fix non-chapter entries at L2 that appear between two chapters.
    # These are subtopics (like "Case Situation", "Review Terms") that the Vision API
    # assigned to L2 instead of L3 due to batch boundary context loss.
    for i in range(len(entries)):
        entry = entries[i]
        if (
            entry.level == 2
            and not chapter_pattern.match(entry.title)
            and not section_pattern.match(entry.title)
        ):
            # Check if there's a preceding chapter (L2 numbered entry) that should be the parent
            prev_chapter_idx = None
            for k in range(i - 1, -1, -1):
                if entries[k].level <= 1:
                    break  # Reached a section, stop
                if entries[k].level == 2 and chapter_pattern.match(entries[k].title):
                    prev_chapter_idx = k
                    break

            if prev_chapter_idx is not None:
                # This entry should be L3 (subtopic of the preceding chapter)
                entry.level = 3

    return entries


def _normalize_levels(entries: list[BookmarkEntry]) -> list[BookmarkEntry]:
    """Ensure hierarchy levels have no gaps and no jumps > 1."""
    if not entries:
        return entries

    # Ensure first entry is level 1
    if entries[0].level != 1:
        entries[0].level = 1

    # Fix jumps > 1
    for i in range(1, len(entries)):
        prev_level = entries[i - 1].level
        if entries[i].level > prev_level + 1:
            entries[i].level = prev_level + 1

    return entries
