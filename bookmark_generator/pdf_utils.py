"""Core PDF utility functions using PyMuPDF."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from .models import PageNumberMapping


@dataclass
class TextSpan:
    """A span of text with font metadata."""
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    color: int  # RGB as integer
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_index: int


@dataclass
class TextLine:
    """A line of text composed of spans."""
    spans: list[TextSpan] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    page_index: int = 0

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans).strip()

    @property
    def dominant_font_size(self) -> float:
        if not self.spans:
            return 0.0
        size_counts: dict[float, int] = {}
        for s in self.spans:
            text_len = len(s.text.strip())
            if text_len > 0:
                size_counts[s.font_size] = size_counts.get(s.font_size, 0) + text_len
        if not size_counts:
            return self.spans[0].font_size
        return max(size_counts, key=size_counts.get)

    @property
    def is_bold(self) -> bool:
        return any(s.is_bold for s in self.spans if s.text.strip())

    @property
    def is_mostly_bold(self) -> bool:
        bold_chars = sum(len(s.text) for s in self.spans if s.is_bold)
        total_chars = sum(len(s.text) for s in self.spans)
        return total_chars > 0 and (bold_chars / total_chars) > 0.5


def open_pdf(path: str) -> fitz.Document:
    """Open a PDF file and return the document object."""
    return fitz.open(path)


def extract_page_text(page: fitz.Page) -> str:
    """Extract plain text from a single page."""
    return page.get_text("text")


def extract_page_lines(page: fitz.Page, page_index: int) -> list[TextLine]:
    """Extract text lines with full font metadata from a page."""
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    lines: list[TextLine] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Only text blocks
            continue
        for line_data in block.get("lines", []):
            spans: list[TextSpan] = []
            for span_data in line_data.get("spans", []):
                font_name = span_data.get("font", "")
                flags = span_data.get("flags", 0)
                # PyMuPDF font flags: bit 0 = superscript, bit 1 = italic,
                # bit 2 = serif, bit 3 = monospace, bit 4 = bold
                is_bold = bool(flags & (1 << 4)) or _font_name_indicates_bold(font_name)
                is_italic = bool(flags & (1 << 1)) or "italic" in font_name.lower()

                span = TextSpan(
                    text=span_data.get("text", ""),
                    font_name=font_name,
                    font_size=round(span_data.get("size", 0), 2),
                    is_bold=is_bold,
                    is_italic=is_italic,
                    color=span_data.get("color", 0),
                    bbox=tuple(span_data.get("bbox", (0, 0, 0, 0))),
                    page_index=page_index,
                )
                spans.append(span)

            if spans:
                line_bbox = (
                    min(s.bbox[0] for s in spans),
                    min(s.bbox[1] for s in spans),
                    max(s.bbox[2] for s in spans),
                    max(s.bbox[3] for s in spans),
                )
                text_line = TextLine(
                    spans=spans,
                    bbox=line_bbox,
                    page_index=page_index,
                )
                if text_line.text:  # Skip empty lines
                    lines.append(text_line)

    return lines


def _font_name_indicates_bold(font_name: str) -> bool:
    """Check if the font name suggests bold weight."""
    name_lower = font_name.lower()
    bold_indicators = ["bold", "black", "heavy", "demi", "semibold"]
    return any(ind in name_lower for ind in bold_indicators)


def extract_page_number_from_text(page: fitz.Page) -> Optional[str]:
    """Try to extract the printed page number from a page's header/footer.

    Looks at the bottom and top of the page for standalone numbers.
    """
    lines = extract_page_lines(page, 0)
    if not lines:
        return None

    page_height = page.rect.height

    # Check bottom 10% of the page for page numbers
    bottom_threshold = page_height * 0.90
    # Check top 10% for page numbers (some books put them at top)
    top_threshold = page_height * 0.10

    candidates: list[tuple[str, float]] = []

    for line in lines:
        y_center = (line.bbox[1] + line.bbox[3]) / 2
        text = line.text.strip()

        # Skip empty or very long lines (not page numbers)
        if not text or len(text) > 20:
            continue

        # Check if this line is in header/footer area
        if y_center > bottom_threshold or y_center < top_threshold:
            # Try to extract a number
            # Handle formats like: "12", "- 12 -", "Page 12", "| 12 |", "xii"
            number_match = re.match(
                r'^[\s\-|]*(?:page\s*)?(\d+)[\s\-|]*$',
                text, re.IGNORECASE
            )
            if number_match:
                candidates.append((number_match.group(1), y_center))
                continue

            # Check for Roman numerals
            roman_match = re.match(
                r'^[\s\-|]*((?:x{0,3})(?:ix|iv|v?i{0,3}))[\s\-|]*$',
                text, re.IGNORECASE
            )
            if roman_match and roman_match.group(1):
                candidates.append((roman_match.group(1).lower(), y_center))

    if candidates:
        # Prefer bottom of page
        bottom_candidates = [(num, y) for num, y in candidates if y > bottom_threshold]
        if bottom_candidates:
            return bottom_candidates[0][0]
        return candidates[0][0]

    return None


def build_page_number_mapping(doc: fitz.Document, sample_pages: Optional[list[int]] = None) -> PageNumberMapping:
    """Build a mapping between physical PDF page indices and logical page numbers.

    Samples pages throughout the document to detect the offset between
    printed page numbers and PDF page indices.
    """
    mapping = PageNumberMapping()

    total_pages = len(doc)
    if sample_pages is None:
        # Sample strategically: first few pages, some middle, and end
        sample_pages = []
        # First 20 pages (likely to find where numbering starts)
        sample_pages.extend(range(min(20, total_pages)))
        # Some middle pages
        if total_pages > 40:
            mid = total_pages // 2
            sample_pages.extend(range(mid - 2, min(mid + 3, total_pages)))
        # Last few pages
        if total_pages > 10:
            sample_pages.extend(range(max(0, total_pages - 3), total_pages))
        sample_pages = sorted(set(sample_pages))

    offsets: list[int] = []

    for page_idx in sample_pages:
        if page_idx >= total_pages:
            continue
        page = doc[page_idx]
        page_num_str = extract_page_number_from_text(page)
        if page_num_str is not None:
            mapping.physical_to_logical[page_idx] = page_num_str
            try:
                logical_num = int(page_num_str)
                offset = page_idx - logical_num + 1
                offsets.append(offset)
            except ValueError:
                pass  # Roman numeral or other format

    if offsets:
        # Use the most common offset
        from collections import Counter
        offset_counts = Counter(offsets)
        most_common_offset = offset_counts.most_common(1)[0][0]
        mapping.offset = most_common_offset
        mapping.confidence = offset_counts[most_common_offset] / len(offsets)

    return mapping
