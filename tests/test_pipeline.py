"""Tests for the bookmark extraction pipeline.

Creates synthetic PDFs with known structure to validate each component
and the full end-to-end pipeline.
"""

from __future__ import annotations

import os
import tempfile

import fitz
import pytest

from bookmark_generator.bookmark_extractor import (
    extract_existing_bookmarks,
    has_bookmarks,
    inject_bookmarks,
)
from bookmark_generator.font_heuristics import (
    build_font_profile,
    detect_headings,
    detect_headers_footers,
    score_heading_candidate,
    FontProfile,
    CHAPTER_PATTERNS,
    NUMBERED_HEADING_PATTERN,
)
from bookmark_generator.models import BookmarkEntry, PageNumberMapping
from bookmark_generator.pdf_utils import (
    build_page_number_mapping,
    extract_page_lines,
    extract_page_number_from_text,
    extract_page_text,
)
from bookmark_generator.pipeline import extract_bookmarks
from bookmark_generator.reconciler import reconcile_bookmarks
from bookmark_generator.toc_parser import find_toc_pages, parse_toc_entries


# ─── Helpers: Synthetic PDF generation ──────────────────────────────────────────

def _create_pdf_with_toc_and_chapters(path: str) -> None:
    """Create a synthetic PDF with a TOC page and chapter headings.

    Structure:
    - Page 0: Title page
    - Page 1: Table of Contents
    - Page 2: Chapter 1: Introduction (p.1)
    - Page 3: 1.1 Background (p.2)
    - Page 4: Chapter 2: Methods (p.3)
    - Page 5: 2.1 Data Collection (p.4)
    - Page 6: 2.2 Analysis (p.5)
    - Page 7: Chapter 3: Results (p.6)
    - Page 8: Appendix A (p.7)
    """
    doc = fitz.open()

    # ── Page 0: Title page ──────────────────────────────────────────────
    page = doc.new_page(width=595, height=842)
    page.insert_text((200, 300), "My Research Paper", fontsize=24, fontname="helv")
    page.insert_text((220, 360), "By Author Name", fontsize=14, fontname="helv")

    # ── Page 1: Table of Contents ───────────────────────────────────────
    page = doc.new_page(width=595, height=842)
    page.insert_text((220, 60), "Table of Contents", fontsize=16, fontname="helv")
    y = 120
    toc_items = [
        ("Chapter 1: Introduction", "1"),
        ("  1.1 Background", "2"),
        ("Chapter 2: Methods", "3"),
        ("  2.1 Data Collection", "4"),
        ("  2.2 Analysis", "5"),
        ("Chapter 3: Results", "6"),
        ("Appendix A", "7"),
    ]
    for title, page_num in toc_items:
        dots = "." * (50 - len(title) - len(page_num))
        page.insert_text((72, y), f"{title} {dots} {page_num}", fontsize=11, fontname="helv")
        y += 20

    # ── Pages 2-8: Content pages ────────────────────────────────────────
    chapters = [
        ("Chapter 1: Introduction", "This is the introduction chapter with some body text. " * 10),
        ("1.1 Background", "Background information goes here. " * 15),
        ("Chapter 2: Methods", "This chapter describes the methodology. " * 10),
        ("2.1 Data Collection", "Data was collected from various sources. " * 12),
        ("2.2 Analysis", "Statistical analysis was performed. " * 12),
        ("Chapter 3: Results", "The results of the study are presented here. " * 10),
        ("Appendix A", "Supplementary material is included in this appendix. " * 10),
    ]

    for i, (heading, body) in enumerate(chapters):
        page = doc.new_page(width=595, height=842)
        # Heading in larger, bold font
        heading_fontsize = 18 if heading.startswith("Chapter") or heading.startswith("Appendix") else 14
        page.insert_text((72, 60), heading, fontsize=heading_fontsize, fontname="helv")
        # Body text
        _insert_wrapped_text(page, body, x=72, y=100, fontsize=11, max_width=450)
        # Page number at bottom
        page.insert_text((280, 810), str(i + 1), fontsize=10, fontname="helv")

    doc.save(path)
    doc.close()


def _create_pdf_with_existing_bookmarks(path: str) -> None:
    """Create a PDF that already has bookmarks in its outline."""
    doc = fitz.open()

    for i in range(5):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), f"Chapter {i + 1}", fontsize=18, fontname="helv")
        page.insert_text((72, 100), f"Content for chapter {i + 1}. " * 20, fontsize=11, fontname="helv")

    # Set bookmarks
    toc = [
        [1, "Chapter 1", 1],
        [1, "Chapter 2", 2],
        [1, "Chapter 3", 3],
        [2, "Section 3.1", 3],
        [1, "Chapter 4", 4],
        [1, "Chapter 5", 5],
    ]
    doc.set_toc(toc)
    doc.save(path)
    doc.close()


def _create_pdf_no_toc(path: str) -> None:
    """Create a PDF with chapter headings but no TOC page."""
    doc = fitz.open()

    chapters = [
        ("Introduction", 20),
        ("Literature Review", 18),
        ("Methodology", 18),
        ("Results and Discussion", 18),
        ("Conclusion", 18),
    ]

    for heading, size in chapters:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), heading, fontsize=size, fontname="helv")
        body = f"This section covers {heading.lower()}. " * 20
        _insert_wrapped_text(page, body, x=72, y=100, fontsize=11, max_width=450)
        # Page number at bottom
        page.insert_text((280, 810), str(len(doc)), fontsize=10, fontname="helv")

    doc.save(path)
    doc.close()


def _create_pdf_with_headers_footers(path: str) -> None:
    """Create a PDF with running headers/footers and numbered sections.

    Every page has:
    - Running header "SGNA Core Curriculum" at top
    - Running footer "Copyright 2024" at bottom
    - Page number at bottom-right
    Chapters use larger fonts, sections use numbered headings like "1.1 Title".
    """
    doc = fitz.open()

    sections = [
        ("Chapter 1 Introduction", 20, [
            ("1.1 Overview of the Field", 14),
            ("1.2 Historical Background", 14),
        ]),
        ("Chapter 2 Patient Assessment", 20, [
            ("2.1 Initial Evaluation", 14),
            ("2.2 Risk Stratification", 14),
            ("2.2.1 Cardiovascular Risk", 12),
        ]),
        ("Chapter 3 Procedures", 20, [
            ("3.1 Endoscopy", 14),
        ]),
    ]

    page_num = 1
    for chapter_title, chapter_size, subs in sections:
        # Chapter page
        page = doc.new_page(width=595, height=842)
        # Running header (same text every page)
        page.insert_text((72, 30), "SGNA Core Curriculum", fontsize=9, fontname="helv")
        # Chapter heading
        page.insert_text((72, 80), chapter_title, fontsize=chapter_size, fontname="helv")
        body = f"Content about {chapter_title.lower()}. " * 15
        _insert_wrapped_text(page, body, x=72, y=120, fontsize=11, max_width=450)
        # Running footer
        page.insert_text((72, 820), "Copyright 2024", fontsize=8, fontname="helv")
        page.insert_text((550, 820), str(page_num), fontsize=9, fontname="helv")
        page_num += 1

        # Section pages
        for sub_title, sub_size in subs:
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 30), "SGNA Core Curriculum", fontsize=9, fontname="helv")
            page.insert_text((72, 80), sub_title, fontsize=sub_size, fontname="helv")
            body = f"Details about {sub_title.lower()}. " * 20
            _insert_wrapped_text(page, body, x=72, y=120, fontsize=11, max_width=450)
            page.insert_text((72, 820), "Copyright 2024", fontsize=8, fontname="helv")
            page.insert_text((550, 820), str(page_num), fontsize=9, fontname="helv")
            page_num += 1

    doc.save(path)
    doc.close()


def _create_pdf_uppercase_headings(path: str) -> None:
    """Create a PDF where headings are uppercase bold, same font size as body."""
    doc = fitz.open()

    headings = ["INTRODUCTION", "METHODOLOGY", "RESULTS", "DISCUSSION", "CONCLUSION"]
    for heading in headings:
        page = doc.new_page(width=595, height=842)
        # Heading in bold uppercase — note: we use "hebo" for bold font
        page.insert_text((72, 80), heading, fontsize=12, fontname="hebo")
        body = f"This section discusses {heading.lower()} in detail. " * 20
        _insert_wrapped_text(page, body, x=72, y=110, fontsize=11, max_width=450)

    doc.save(path)
    doc.close()


def _insert_wrapped_text(page, text, x, y, fontsize, max_width):
    """Insert text with basic word wrapping."""
    words = text.split()
    line = ""
    current_y = y
    for word in words:
        test_line = f"{line} {word}".strip()
        # Rough character width estimate
        if len(test_line) * fontsize * 0.5 > max_width:
            page.insert_text((x, current_y), line, fontsize=fontsize, fontname="helv")
            current_y += fontsize + 4
            line = word
            if current_y > 780:
                break
        else:
            line = test_line
    if line and current_y <= 780:
        page.insert_text((x, current_y), line, fontsize=fontsize, fontname="helv")


# ─── Test fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def pdf_with_toc(tmp_path):
    path = str(tmp_path / "with_toc.pdf")
    _create_pdf_with_toc_and_chapters(path)
    return path


@pytest.fixture
def pdf_with_bookmarks(tmp_path):
    path = str(tmp_path / "with_bookmarks.pdf")
    _create_pdf_with_existing_bookmarks(path)
    return path


@pytest.fixture
def pdf_no_toc(tmp_path):
    path = str(tmp_path / "no_toc.pdf")
    _create_pdf_no_toc(path)
    return path


@pytest.fixture
def pdf_with_headers_footers(tmp_path):
    path = str(tmp_path / "headers_footers.pdf")
    _create_pdf_with_headers_footers(path)
    return path


@pytest.fixture
def pdf_uppercase_headings(tmp_path):
    path = str(tmp_path / "uppercase.pdf")
    _create_pdf_uppercase_headings(path)
    return path


# ─── Tests: PDF utilities ──────────────────────────────────────────────────────

class TestPDFUtils:
    def test_extract_page_text(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        text = extract_page_text(doc[0])
        assert "My Research Paper" in text
        doc.close()

    def test_extract_page_lines(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        lines = extract_page_lines(doc[0], 0)
        assert len(lines) > 0
        assert any("Research Paper" in l.text for l in lines)
        doc.close()

    def test_extract_page_number_from_text(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        # Content pages (index 2+) should have page numbers at the bottom
        page_num = extract_page_number_from_text(doc[2])
        # Page 2 (0-indexed) should show "1"
        assert page_num is not None
        assert page_num == "1"
        doc.close()

    def test_build_page_number_mapping(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        mapping = build_page_number_mapping(doc)
        assert isinstance(mapping, PageNumberMapping)
        # We should detect some page numbers
        assert len(mapping.physical_to_logical) > 0
        doc.close()


# ─── Tests: Existing bookmark extraction ───────────────────────────────────────

class TestBookmarkExtractor:
    def test_has_bookmarks_true(self, pdf_with_bookmarks):
        doc = fitz.open(pdf_with_bookmarks)
        assert has_bookmarks(doc) is True
        doc.close()

    def test_has_bookmarks_false(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        assert has_bookmarks(doc) is False
        doc.close()

    def test_extract_existing_bookmarks(self, pdf_with_bookmarks):
        doc = fitz.open(pdf_with_bookmarks)
        entries = extract_existing_bookmarks(doc)
        assert len(entries) == 6
        assert entries[0].title == "Chapter 1"
        assert entries[0].level == 1
        assert entries[3].title == "Section 3.1"
        assert entries[3].level == 2
        doc.close()

    def test_inject_bookmarks(self, tmp_path):
        path = str(tmp_path / "inject_test.pdf")
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1}", fontsize=12)
        doc.save(path)
        doc.close()

        doc = fitz.open(path)
        bookmarks = [
            BookmarkEntry(title="Chapter 1", page_number=1, pdf_page_index=0, level=1),
            BookmarkEntry(title="Chapter 2", page_number=2, pdf_page_index=1, level=1),
        ]
        inject_bookmarks(doc, bookmarks)
        out_path = str(tmp_path / "inject_test_out.pdf")
        doc.save(out_path)
        doc.close()
        path = out_path

        # Verify
        doc = fitz.open(path)
        assert has_bookmarks(doc)
        extracted = extract_existing_bookmarks(doc)
        assert len(extracted) == 2
        assert extracted[0].title == "Chapter 1"
        doc.close()

    def test_inject_bookmarks_normalizes_levels(self, tmp_path):
        """Ensure inject handles entries that don't start at level 1."""
        path = str(tmp_path / "level_test.pdf")
        doc = fitz.open()
        for i in range(5):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1}", fontsize=12)
        doc.save(path)
        doc.close()

        doc = fitz.open(path)
        # Simulate entries where levels start at 2 and have gaps
        bookmarks = [
            BookmarkEntry(title="Section A", page_number=1, pdf_page_index=0, level=2),
            BookmarkEntry(title="Sub A.1", page_number=2, pdf_page_index=1, level=4),
            BookmarkEntry(title="Section B", page_number=3, pdf_page_index=2, level=2),
        ]
        inject_bookmarks(doc, bookmarks)
        out_path = str(tmp_path / "level_test_out.pdf")
        doc.save(out_path)
        doc.close()

        # Verify it saved without error and bookmarks are readable
        doc = fitz.open(out_path)
        assert has_bookmarks(doc)
        extracted = extract_existing_bookmarks(doc)
        assert len(extracted) == 3
        assert extracted[0].level == 1  # Normalized to start at 1
        doc.close()


# ─── Tests: TOC parser ─────────────────────────────────────────────────────────

class TestTOCParser:
    def test_find_toc_pages(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        toc_loc = find_toc_pages(doc)
        assert toc_loc is not None
        assert toc_loc.start_page_index == 1  # Second page
        assert "Table of Contents" in toc_loc.heading_text
        doc.close()

    def test_find_toc_pages_none(self, pdf_no_toc):
        doc = fitz.open(pdf_no_toc)
        toc_loc = find_toc_pages(doc)
        assert toc_loc is None
        doc.close()

    def test_parse_toc_entries(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        toc_loc = find_toc_pages(doc)
        assert toc_loc is not None

        entries = parse_toc_entries(doc, toc_loc)
        assert len(entries) >= 5  # At least 5 of the 7 TOC items

        # Check that titles are present
        titles = [e.title for e in entries]
        assert any("Introduction" in t for t in titles)
        assert any("Methods" in t for t in titles)
        assert any("Results" in t for t in titles)

        # Check page numbers
        intro_entry = next(e for e in entries if "Introduction" in e.title)
        assert intro_entry.page_number == 1

        doc.close()


# ─── Tests: Font heuristics ────────────────────────────────────────────────────

class TestFontHeuristics:
    def test_build_font_profile(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        profile = build_font_profile(doc, skip_pages={0, 1})
        assert profile.body_font_size > 0
        doc.close()

    def test_detect_headings(self, pdf_no_toc):
        doc = fitz.open(pdf_no_toc)
        profile = build_font_profile(doc)
        headings = detect_headings(doc, profile)
        assert len(headings) > 0
        titles = [h.title for h in headings]
        assert any("Introduction" in t for t in titles)
        doc.close()

    def test_detect_headings_with_skip(self, pdf_with_toc):
        doc = fitz.open(pdf_with_toc)
        skip = {0, 1}  # Skip title and TOC pages
        profile = build_font_profile(doc, skip_pages=skip)
        headings = detect_headings(doc, profile, skip_pages=skip)
        # Should detect chapter headings
        assert len(headings) >= 3
        doc.close()


# ─── Tests: Heading scoring ───────────────────────────────────────────────────

class TestHeadingScoring:
    """Tests for the multi-signal scoring engine."""

    def test_chapter_pattern_scores_high(self):
        """Lines matching 'Chapter N' should score high and get level 1."""
        from bookmark_generator.pdf_utils import TextSpan, TextLine

        span = TextSpan(
            text="Chapter 3 Patient Assessment",
            font_name="Helvetica-Bold", font_size=18.0,
            is_bold=True, is_italic=False, color=0,
            bbox=(72, 60, 400, 78), page_index=0,
        )
        line = TextLine(spans=[span], bbox=(72, 60, 400, 78), page_index=0)
        profile = FontProfile(body_font_size=11.0, body_font_name="Helvetica")

        score, level = score_heading_candidate(line, profile)
        assert score >= 0.7
        assert level == 1

    def test_numbered_heading_level_from_depth(self):
        """'1.2.3 Title' should get level 3 from dot-depth."""
        from bookmark_generator.pdf_utils import TextSpan, TextLine

        span = TextSpan(
            text="1.2.3 Subsection Title",
            font_name="Helvetica-Bold", font_size=13.0,
            is_bold=True, is_italic=False, color=0,
            bbox=(72, 60, 300, 73), page_index=0,
        )
        line = TextLine(spans=[span], bbox=(72, 60, 300, 73), page_index=0)
        profile = FontProfile(body_font_size=11.0, body_font_name="Helvetica")

        score, level = score_heading_candidate(line, profile)
        assert score >= 0.35
        assert level == 3

    def test_uppercase_bold_scores(self):
        """ALL CAPS bold text should score as a heading."""
        from bookmark_generator.pdf_utils import TextSpan, TextLine

        span = TextSpan(
            text="INTRODUCTION",
            font_name="Helvetica-Bold", font_size=12.0,
            is_bold=True, is_italic=False, color=0,
            bbox=(72, 60, 200, 72), page_index=0,
        )
        line = TextLine(spans=[span], bbox=(72, 60, 200, 72), page_index=0)
        profile = FontProfile(body_font_size=11.0, body_font_name="Helvetica")

        score, level = score_heading_candidate(line, profile)
        # Bold + uppercase + chapter pattern ("Introduction") + short text
        assert score >= 0.35

    def test_body_text_scores_low(self):
        """Normal body text should score below threshold."""
        from bookmark_generator.pdf_utils import TextSpan, TextLine

        span = TextSpan(
            text="This is a normal sentence with regular formatting and no special patterns.",
            font_name="Helvetica", font_size=11.0,
            is_bold=False, is_italic=False, color=0,
            bbox=(72, 200, 500, 211), page_index=0,
        )
        line = TextLine(spans=[span], bbox=(72, 200, 500, 211), page_index=0)
        profile = FontProfile(body_font_size=11.0, body_font_name="Helvetica")

        score, level = score_heading_candidate(line, profile)
        assert score < 0.35

    def test_period_ending_penalty(self):
        """Text ending with a period gets penalized (likely a sentence, not a heading)."""
        from bookmark_generator.pdf_utils import TextSpan, TextLine

        span = TextSpan(
            text="This looks like a heading but ends with a period.",
            font_name="Helvetica-Bold", font_size=14.0,
            is_bold=True, is_italic=False, color=0,
            bbox=(72, 60, 400, 74), page_index=0,
        )
        line = TextLine(spans=[span], bbox=(72, 60, 400, 74), page_index=0)
        profile = FontProfile(body_font_size=11.0, body_font_name="Helvetica")

        score_period, _ = score_heading_candidate(line, profile)

        # Same text without period should score higher
        span2 = TextSpan(
            text="This looks like a heading but ends with a period",
            font_name="Helvetica-Bold", font_size=14.0,
            is_bold=True, is_italic=False, color=0,
            bbox=(72, 60, 400, 74), page_index=0,
        )
        line2 = TextLine(spans=[span2], bbox=(72, 60, 400, 74), page_index=0)
        score_no_period, _ = score_heading_candidate(line2, profile)

        assert score_no_period > score_period

    def test_exclude_patterns_reject_noise(self):
        """Figure captions, URLs, emails should score 0."""
        from bookmark_generator.pdf_utils import TextSpan, TextLine

        noise_texts = [
            "Figure 3: Distribution of Results",
            "Table 2 Summary Statistics",
            "https://example.com/paper",
            "42",
        ]
        profile = FontProfile(body_font_size=11.0, body_font_name="Helvetica")

        for text in noise_texts:
            span = TextSpan(
                text=text, font_name="Helvetica-Bold", font_size=14.0,
                is_bold=True, is_italic=False, color=0,
                bbox=(72, 60, 400, 74), page_index=0,
            )
            line = TextLine(spans=[span], bbox=(72, 60, 400, 74), page_index=0)
            score, _ = score_heading_candidate(line, profile)
            assert score == 0.0, f"'{text}' should be excluded but scored {score}"


# ─── Tests: Header/footer detection ──────────────────────────────────────────

class TestHeaderFooterDetection:
    def test_detect_running_headers(self, pdf_with_headers_footers):
        """Running headers/footers should be detected and excluded."""
        doc = fitz.open(pdf_with_headers_footers)
        profile = build_font_profile(doc)
        excluded = detect_headers_footers(doc, profile)

        # Should detect repeated text
        excluded_texts = set(text for _, text in excluded)
        # "SGNA Core Curriculum" appears on every page at the same Y
        assert any("SGNA" in t or "Core" in t or "Copyright" in t for t in excluded_texts)
        doc.close()

    def test_headings_not_excluded_as_headers(self, pdf_with_headers_footers):
        """Chapter headings should NOT be excluded as headers (they vary)."""
        doc = fitz.open(pdf_with_headers_footers)
        profile = build_font_profile(doc)
        headings = detect_headings(doc, profile)

        titles = [h.title for h in headings]
        # Running headers should not appear as headings
        assert not any("SGNA Core Curriculum" in t for t in titles)
        assert not any("Copyright" in t for t in titles)
        doc.close()


# ─── Tests: Numbered heading hierarchy ────────────────────────────────────────

class TestNumberedHeadings:
    def test_numbered_headings_detected(self, pdf_with_headers_footers):
        """Numbered headings like '1.1 Title' should be detected with correct levels."""
        doc = fitz.open(pdf_with_headers_footers)
        profile = build_font_profile(doc)
        headings = detect_headings(doc, profile)

        titles = [h.title for h in headings]
        # Should find numbered sections
        assert any("1.1" in t for t in titles) or any("Overview" in t for t in titles)
        doc.close()

    def test_hierarchy_levels_from_numbering(self, pdf_with_headers_footers):
        """Deeper numbered headings should get higher level numbers."""
        doc = fitz.open(pdf_with_headers_footers)
        profile = build_font_profile(doc)
        headings = detect_headings(doc, profile)

        # Find a '2.2.1' style heading if detected
        for h in headings:
            if "2.2.1" in h.title or "Cardiovascular" in h.title:
                # Should be level 3 (or deeper than level 1)
                assert h.level >= 2
                break
        doc.close()


# ─── Tests: Reconciler ─────────────────────────────────────────────────────────

class TestReconciler:
    def test_reconcile_toc_only(self):
        toc = [
            BookmarkEntry(title="Chapter 1", page_number=1, level=1, source="toc_parse"),
            BookmarkEntry(title="Chapter 2", page_number=5, level=1, source="toc_parse"),
        ]
        result = reconcile_bookmarks(toc, [], page_mapping=None)
        assert len(result) == 2

    def test_reconcile_font_only(self):
        font = [
            BookmarkEntry(title="Introduction", page_number=1, pdf_page_index=0, level=1, confidence=0.8, source="font_heuristic"),
            BookmarkEntry(title="Methods", page_number=3, pdf_page_index=2, level=1, confidence=0.8, source="font_heuristic"),
        ]
        result = reconcile_bookmarks([], font, page_mapping=None)
        assert len(result) == 2

    def test_reconcile_matching_entries(self):
        toc = [
            BookmarkEntry(title="Chapter 1: Introduction", page_number=1, level=1, confidence=0.8, source="toc_parse"),
            BookmarkEntry(title="Chapter 2: Methods", page_number=5, level=1, confidence=0.8, source="toc_parse"),
        ]
        font = [
            BookmarkEntry(title="Chapter 1: Introduction", page_number=1, pdf_page_index=2, level=1, confidence=0.7, source="font_heuristic"),
            BookmarkEntry(title="Chapter 2: Methods", page_number=5, pdf_page_index=6, level=1, confidence=0.7, source="font_heuristic"),
        ]
        result = reconcile_bookmarks(toc, font)
        assert len(result) == 2
        # Should be reconciled entries with boosted confidence
        assert all(e.source == "reconciled" for e in result)

    def test_reconcile_with_page_mapping(self):
        mapping = PageNumberMapping(offset=2)
        toc = [
            BookmarkEntry(title="Chapter 1", page_number=1, level=1, source="toc_parse"),
        ]
        result = reconcile_bookmarks(toc, [], page_mapping=mapping)
        assert len(result) == 1
        assert result[0].pdf_page_index == 2  # offset applied

    def test_hierarchy_building(self):
        toc = [
            BookmarkEntry(title="Chapter 1", page_number=1, level=1, source="toc_parse"),
            BookmarkEntry(title="1.1 Section", page_number=2, level=2, source="toc_parse"),
            BookmarkEntry(title="1.2 Section", page_number=3, level=2, source="toc_parse"),
            BookmarkEntry(title="Chapter 2", page_number=4, level=1, source="toc_parse"),
        ]
        result = reconcile_bookmarks(toc, [])
        assert len(result) == 2  # 2 top-level chapters
        assert len(result[0].children) == 2  # Chapter 1 has 2 sections
        assert len(result[1].children) == 0


# ─── Tests: Full pipeline ──────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_with_existing_bookmarks(self, pdf_with_bookmarks):
        result = extract_bookmarks(pdf_with_bookmarks)
        assert result.method_used == "existing_bookmarks"
        assert len(result.flat_bookmarks()) == 6

    def test_pipeline_with_toc(self, pdf_with_toc):
        result = extract_bookmarks(pdf_with_toc)
        assert "toc_parse" in result.method_used
        assert len(result.flat_bookmarks()) >= 5

    def test_pipeline_font_only(self, pdf_no_toc):
        result = extract_bookmarks(pdf_no_toc)
        assert "font_heuristics" in result.method_used
        assert len(result.flat_bookmarks()) >= 3

    def test_pipeline_force_rebuild(self, pdf_with_bookmarks):
        result = extract_bookmarks(pdf_with_bookmarks, force_rebuild=True)
        assert result.method_used != "existing_bookmarks"

    def test_pipeline_inject_bookmarks(self, pdf_with_toc, tmp_path):
        output_path = str(tmp_path / "output.pdf")
        result = extract_bookmarks(
            pdf_with_toc,
            inject_into_pdf=True,
            output_path=output_path,
        )
        assert os.path.exists(output_path)

        # Verify the output PDF has bookmarks
        doc = fitz.open(output_path)
        assert has_bookmarks(doc)
        doc.close()

    def test_pipeline_json_output(self, pdf_with_toc):
        result = extract_bookmarks(pdf_with_toc)
        for entry in result.bookmarks:
            d = entry.to_dict()
            assert "title" in d
            assert "page_number" in d
            assert "children" in d

    def test_pipeline_disable_toc(self, pdf_with_toc):
        result = extract_bookmarks(pdf_with_toc, use_toc=False)
        assert "toc_parse" not in result.method_used

    def test_pipeline_disable_font(self, pdf_with_toc):
        result = extract_bookmarks(pdf_with_toc, use_font_heuristics=False)
        assert "font_heuristics" not in result.method_used


# ─── Tests: Models ──────────────────────────────────────────────────────────────

class TestModels:
    def test_bookmark_entry_repr(self):
        entry = BookmarkEntry(
            title="Chapter 1", page_number=5, pdf_page_index=6,
            level=1, source="toc_parse",
        )
        r = repr(entry)
        assert "Chapter 1" in r
        assert "toc_parse" in r

    def test_extraction_result_flat(self):
        from bookmark_generator.models import ExtractionResult
        result = ExtractionResult(pdf_path="test.pdf")
        parent = BookmarkEntry(title="Ch1", page_number=1, level=1)
        child = BookmarkEntry(title="S1.1", page_number=2, level=2)
        parent.children.append(child)
        result.bookmarks = [parent]
        flat = result.flat_bookmarks()
        assert len(flat) == 2

    def test_page_number_mapping(self):
        mapping = PageNumberMapping(offset=2)
        # logical page 1 -> physical index 2 (1 + 2 - 1 = 2)
        assert mapping.logical_to_physical(1) == 2
        assert mapping.logical_to_physical(5) == 6

    def test_page_number_mapping_explicit(self):
        mapping = PageNumberMapping()
        mapping.physical_to_logical = {0: "i", 1: "ii", 2: "1", 3: "2"}
        assert mapping.logical_to_physical(1) == 2
        assert mapping.logical_to_physical(2) == 3
        assert mapping.physical_to_logical_num(2) == 1
