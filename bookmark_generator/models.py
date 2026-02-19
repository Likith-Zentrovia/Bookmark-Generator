"""Data models for bookmark extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class HeadingLevel(Enum):
    """Heading hierarchy level."""
    CHAPTER = auto()      # e.g., "Chapter 1", "Part I"
    SECTION = auto()      # e.g., "1.1 Introduction"
    SUBSECTION = auto()   # e.g., "1.1.1 Background"
    SUB_SUBSECTION = auto()  # e.g., "1.1.1.1 Details"


@dataclass
class BookmarkEntry:
    """A single bookmark/chapter entry."""
    title: str
    page_number: int          # Logical page number (as printed in the PDF)
    pdf_page_index: int = -1  # Physical 0-based PDF page index
    level: int = 1            # Nesting depth: 1=chapter, 2=section, 3=subsection, etc.
    children: list[BookmarkEntry] = field(default_factory=list)
    confidence: float = 1.0   # 0.0-1.0, how confident we are in this entry
    source: str = ""          # "existing_bookmark", "toc_parse", "font_heuristic", "reconciled"

    def __repr__(self) -> str:
        indent = "  " * (self.level - 1)
        return f"{indent}[L{self.level}] {self.title} -> p.{self.page_number} (pdf:{self.pdf_page_index}) [{self.source}]"

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return {
            "title": self.title,
            "page_number": self.page_number,
            "pdf_page_index": self.pdf_page_index,
            "level": self.level,
            "confidence": self.confidence,
            "source": self.source,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class PageNumberMapping:
    """Maps logical (printed) page numbers to physical PDF page indices."""
    # Maps physical PDF page index -> logical page number string
    physical_to_logical: dict[int, str] = field(default_factory=dict)
    # The offset: physical_index = logical_number + offset
    offset: int = 0
    confidence: float = 0.0

    def logical_to_physical(self, logical_page: int) -> int:
        """Convert a logical page number to a physical PDF page index."""
        # First try the explicit mapping
        for phys, log_str in self.physical_to_logical.items():
            try:
                if int(log_str) == logical_page:
                    return phys
            except ValueError:
                continue
        # Fall back to offset
        return logical_page + self.offset - 1

    def physical_to_logical_num(self, physical_index: int) -> Optional[int]:
        """Convert a physical PDF page index to a logical page number."""
        log_str = self.physical_to_logical.get(physical_index)
        if log_str is not None:
            try:
                return int(log_str)
            except ValueError:
                return None
        return physical_index - self.offset + 1


@dataclass
class ExtractionResult:
    """Complete result of bookmark extraction from a PDF."""
    pdf_path: str
    bookmarks: list[BookmarkEntry] = field(default_factory=list)
    page_mapping: Optional[PageNumberMapping] = None
    total_pages: int = 0
    method_used: str = ""       # Which method(s) produced the result
    warnings: list[str] = field(default_factory=list)

    def flat_bookmarks(self) -> list[BookmarkEntry]:
        """Return a flat list of all bookmarks including children."""
        result = []
        def _flatten(entries: list[BookmarkEntry]) -> None:
            for entry in entries:
                result.append(entry)
                _flatten(entry.children)
        _flatten(self.bookmarks)
        return result

    def print_tree(self) -> str:
        """Pretty-print the bookmark tree."""
        lines = []
        def _print(entries: list[BookmarkEntry], depth: int = 0) -> None:
            for entry in entries:
                indent = "  " * depth
                page_info = f"p.{entry.page_number}"
                if entry.pdf_page_index >= 0:
                    page_info += f" (pdf:{entry.pdf_page_index})"
                lines.append(f"{indent}├── {entry.title} [{page_info}] ({entry.source}, conf:{entry.confidence:.2f})")
                _print(entry.children, depth + 1)
        _print(self.bookmarks)
        return "\n".join(lines)
