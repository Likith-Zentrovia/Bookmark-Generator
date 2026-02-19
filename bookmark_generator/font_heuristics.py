"""Font-based heuristic heading detection.

Scans the body of the PDF (outside TOC pages) to detect headings
based on font size, weight, text characteristics, and structural patterns.

Incorporates multi-signal scoring: font size ratio, bold, uppercase, chapter
patterns, numbered heading patterns, Roman numerals, text length, font
difference, and period/colon penalties.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import fitz

from .models import BookmarkEntry
from .pdf_utils import TextLine, extract_page_lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Pattern matchers (ported from the hybrid auto-bookmarker)
# ═══════════════════════════════════════════════════════════════════════════════

CHAPTER_PATTERNS = [
    re.compile(r"^(chapter|chapitre|kapitel|capitulo|capitolo)\s+\d+", re.IGNORECASE),
    re.compile(r"^(part|partie|teil|parte)\s+\w+", re.IGNORECASE),
    re.compile(r"^(section|appendix|annex|annexe)\s+\w+", re.IGNORECASE),
    re.compile(r"^(unit|module|lesson)\s+\d+", re.IGNORECASE),
    re.compile(
        r"^(prologue|epilogue|preface|foreword|introduction|conclusion|"
        r"bibliography|glossary|index|references|acknowledgements?|abstract|summary)\b",
        re.IGNORECASE,
    ),
]

NUMBERED_HEADING_PATTERN = re.compile(
    r"^(\d{1,3}(?:\.\d{1,3})*\.?)\s+[A-Z\u00C0-\u024F]"
)

ROMAN_NUMERAL_PATTERN = re.compile(
    r"^(I{1,3}|IV|V|VI{0,3}|IX|X{0,3}|XI{0,3}|XII{0,3}|XIII|XIV|XV)\.\s+\S",
    re.IGNORECASE,
)

EXCLUDE_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^(page|pg\.?)\s*\d+", re.IGNORECASE),
    re.compile(r"^(figure|fig\.?|table|tab\.?)\s+\d+", re.IGNORECASE),
    re.compile(r"^\(?\d+\)"),
    re.compile(r"^https?://"),
    re.compile(r"^[a-zA-Z0-9._%+-]+@"),
    re.compile(r"^\d+\.\s+\S.*\d+\.\s+\S"),  # Inline numbered lists
    re.compile(r"^\s*©"),
    re.compile(r"^\s*all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"^\s*isbn", re.IGNORECASE),
    re.compile(r"^[\d\s\-.|]+$"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FontProfile:
    """Statistical profile of fonts used in the document."""
    body_font_size: float = 0.0       # Most common font size (body text)
    body_font_name: str = ""
    heading_font_sizes: list[float] = None  # Font sizes larger than body
    min_heading_size: float = 0.0      # Smallest heading font size
    page_height: float = 792.0
    page_width: float = 612.0

    def __post_init__(self):
        if self.heading_font_sizes is None:
            self.heading_font_sizes = []


# ═══════════════════════════════════════════════════════════════════════════════
#  Font profiling
# ═══════════════════════════════════════════════════════════════════════════════

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

    # Heading sizes = sizes larger than body by at least 0.5pt
    heading_sizes = sorted(
        [s for s in size_char_count if s > body_size + 0.5],
        reverse=True,
    )

    min_heading_size = min(heading_sizes) if heading_sizes else body_size + 2

    # Get page dimensions
    page_height = doc[0].rect.height if total_pages > 0 else 792.0
    page_width = doc[0].rect.width if total_pages > 0 else 612.0

    return FontProfile(
        body_font_size=body_size,
        body_font_name=body_font,
        heading_font_sizes=heading_sizes,
        min_heading_size=min_heading_size,
        page_height=page_height,
        page_width=page_width,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Header/footer detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_headers_footers(
    doc: fitz.Document,
    font_profile: FontProfile,
    skip_pages: Optional[set[int]] = None,
) -> set[tuple[int, str]]:
    """Detect running headers/footers by finding repeated text at the same Y position.

    Text that appears at the same Y position on many pages with LOW text variety
    (the same text repeating) is a running header/footer.
    High variety at the same position = chapter titles, not headers.

    Returns a set of (page_index, text) tuples to exclude.
    """
    if skip_pages is None:
        skip_pages = set()

    page_height = font_profile.page_height
    total_pages = len(doc)
    top_zone = page_height * 0.12
    bottom_zone = page_height * 0.88

    # Collect lines in header/footer zones grouped by (rounded_y, rounded_size)
    position_groups: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)

    sample_indices = _get_sample_indices(total_pages, 50, skip_pages)

    for page_idx in sample_indices:
        page = doc[page_idx]
        lines = extract_page_lines(page, page_idx)
        for line in lines:
            y_center = (line.bbox[1] + line.bbox[3]) / 2
            if y_center < top_zone or y_center > bottom_zone:
                text = line.text.strip()
                if not text:
                    continue
                key = (round(y_center / 5) * 5, round(line.dominant_font_size))
                position_groups[key].append((page_idx, text))

    exclude: set[tuple[int, str]] = set()
    min_repeat = max(3, total_pages * 0.15)

    for key, items in position_groups.items():
        if len(items) < min_repeat:
            continue

        pages_seen = set(page_idx for page_idx, _ in items)
        if len(pages_seen) < min_repeat:
            continue

        texts = [text for _, text in items]
        unique_texts = set(texts)
        uniqueness_ratio = len(unique_texts) / len(texts) if texts else 1.0

        # Low variety = running header/footer
        if uniqueness_ratio > 0.7:
            continue

        for page_idx, text in items:
            exclude.add((page_idx, text))

    return exclude


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-signal heading scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_heading_candidate(
    line: TextLine,
    font_profile: FontProfile,
) -> tuple[float, int]:
    """Score a line from 0.0 (not a heading) to 1.0 (definitely a heading).

    Uses multiple signals: font size ratio, bold, uppercase, chapter patterns,
    numbered headings, Roman numerals, text length, font difference, and
    period/colon penalties.

    Returns (score, suggested_level).
    """
    text = line.text.strip()
    body_size = font_profile.body_font_size

    if len(text) < 2 or len(text) > 200:
        return 0.0, 0

    for pattern in EXCLUDE_PATTERNS:
        if pattern.match(text):
            return 0.0, 0

    score = 0.0
    level = 3
    dominant_size = line.dominant_font_size
    size_ratio = dominant_size / body_size if body_size > 0 else 1.0
    is_bold = line.is_mostly_bold
    is_uppercase = text.upper() == text and any(c.isalpha() for c in text)

    # ── Font size signal ──────────────────────────────────────────────────
    if size_ratio >= 1.8:
        score += 0.45
        level = 1
    elif size_ratio >= 1.4:
        score += 0.40
        level = 1
    elif size_ratio >= 1.2:
        score += 0.30
        level = 2
    elif size_ratio >= 1.05:
        score += 0.15
        level = 3

    # ── Bold signal ───────────────────────────────────────────────────────
    if is_bold:
        score += 0.20
        if size_ratio >= 1.05:
            score += 0.10

    # ── UPPERCASE signal ──────────────────────────────────────────────────
    if is_uppercase and len(text) > 3:
        score += 0.15
        if level > 2:
            level = 2

    # ── Chapter pattern signal ────────────────────────────────────────────
    for pattern in CHAPTER_PATTERNS:
        if pattern.match(text):
            score += 0.35
            level = 1
            break

    # ── Numbered heading signal ───────────────────────────────────────────
    numbered_match = NUMBERED_HEADING_PATTERN.match(text)
    if numbered_match:
        number_part = numbered_match.group(1).rstrip(".")
        dot_count = number_part.count(".")
        # Avoid matching list items (small font, not bold)
        if dot_count == 0 and size_ratio < 1.05 and not is_bold:
            pass
        else:
            score += 0.30
            level = min(dot_count + 1, 4)

    # ── Roman numeral signal ──────────────────────────────────────────────
    if ROMAN_NUMERAL_PATTERN.match(text):
        score += 0.25
        level = 1

    # ── Text length signal ────────────────────────────────────────────────
    word_count = len(text.split())
    if word_count <= 8:
        score += 0.10
    elif word_count <= 15:
        score += 0.05
    elif word_count > 25:
        score -= 0.15

    # ── Different font signal ─────────────────────────────────────────────
    dominant_font = _get_dominant_font_name(line)
    if dominant_font and dominant_font != font_profile.body_font_name and size_ratio >= 1.0:
        score += 0.10

    # ── Penalties ─────────────────────────────────────────────────────────
    if text.endswith(".") and not any(p.match(text) for p in CHAPTER_PATTERNS):
        score -= 0.15

    if text.endswith(":"):
        score -= 0.05

    return min(max(score, 0.0), 1.0), level


def _get_dominant_font_name(line: TextLine) -> str:
    """Get the font name covering the most characters in a line."""
    font_chars: dict[str, int] = {}
    for span in line.spans:
        chars = len(span.text.strip())
        if chars > 0:
            font_chars[span.font_name] = font_chars.get(span.font_name, 0) + chars
    if not font_chars:
        return ""
    return max(font_chars, key=font_chars.get)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main heading detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_headings(
    doc: fitz.Document,
    font_profile: FontProfile,
    skip_pages: Optional[set[int]] = None,
    start_page: int = 0,
    end_page: Optional[int] = None,
    min_score: float = 0.35,
) -> list[BookmarkEntry]:
    """Detect headings in the document body using multi-signal font heuristics.

    Each line is scored 0.0-1.0 based on font size ratio, bold, uppercase,
    structural patterns, text length, and font identity. Lines scoring above
    min_score are kept as heading candidates.
    """
    if skip_pages is None:
        skip_pages = set()
    if end_page is None:
        end_page = len(doc)

    if font_profile.body_font_size <= 0:
        return []

    # Detect running headers/footers to exclude
    header_footer_set = detect_headers_footers(doc, font_profile, skip_pages)

    entries: list[BookmarkEntry] = []

    for page_idx in range(start_page, min(end_page, len(doc))):
        if page_idx in skip_pages:
            continue

        page = doc[page_idx]
        lines = extract_page_lines(page, page_idx)
        page_height = page.rect.height

        for line in lines:
            text = line.text.strip()

            # ── Basic filters ─────────────────────────────────────────────
            if len(text) < 2 or len(text) > 200:
                continue

            # Skip lines in extreme header/footer area (top/bottom 5%)
            y_center = (line.bbox[1] + line.bbox[3]) / 2
            if y_center < page_height * 0.05 or y_center > page_height * 0.95:
                continue

            # Skip known running headers/footers
            if (page_idx, text) in header_footer_set:
                continue

            # ── Score the candidate ───────────────────────────────────────
            score, level = score_heading_candidate(line, font_profile)

            if score >= min_score:
                entries.append(BookmarkEntry(
                    title=_clean_heading_text(text),
                    page_number=page_idx + 1,
                    pdf_page_index=page_idx,
                    level=level,
                    confidence=score,
                    source="font_heuristic",
                ))

    # Post-process
    entries = _deduplicate_headings(entries)
    entries = _refine_heading_hierarchy(entries, font_profile)

    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Hierarchy refinement
# ═══════════════════════════════════════════════════════════════════════════════

def _refine_heading_hierarchy(
    headings: list[BookmarkEntry],
    font_profile: FontProfile,
) -> list[BookmarkEntry]:
    """Assign clean hierarchy levels based on font size ranking + numbered patterns.

    Separates headings into numbered (level from dot-depth) and non-numbered
    (level from font-size ranking), then normalizes so there are no gaps.
    """
    if not headings:
        return headings

    non_numbered = [h for h in headings if not NUMBERED_HEADING_PATTERN.match(h.title)]
    numbered = [h for h in headings if NUMBERED_HEADING_PATTERN.match(h.title)]

    # For non-numbered headings, ensure chapter-pattern entries are level 1
    for h in non_numbered:
        if any(p.match(h.title) for p in CHAPTER_PATTERNS):
            h.level = 1

    # For numbered headings, level from dot-depth
    for h in numbered:
        match = NUMBERED_HEADING_PATTERN.match(h.title)
        if match:
            number_part = match.group(1).rstrip(".")
            h.level = min(number_part.count(".") + 1, 4)

    return _normalize_hierarchy_levels(headings)


def _normalize_hierarchy_levels(headings: list[BookmarkEntry]) -> list[BookmarkEntry]:
    """Ensure no level jumps > 1 (required by PDF bookmark spec).

    Maps the used levels to a contiguous 1-based sequence, then clamps
    sequential jumps.
    """
    if not headings:
        return headings

    # Map used levels to contiguous sequence starting at 1
    unique_levels = sorted(set(h.level for h in headings))
    level_map = {old: new + 1 for new, old in enumerate(unique_levels)}

    for h in headings:
        h.level = level_map[h.level]

    # Clamp sequential jumps
    prev_level = 0
    for h in headings:
        if h.level > prev_level + 1:
            h.level = prev_level + 1
        prev_level = h.level

    return headings


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

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
