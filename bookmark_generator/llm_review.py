"""LLM-powered heading validation and enhancement (Stage 2).

Optional module that uses the Anthropic API to review and improve
heuristic heading detections. Validates headings, finds missed ones,
removes false positives, and corrects hierarchy levels.

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# Load .env file from project root
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass

from .models import BookmarkEntry
from .font_heuristics import FontProfile, CHAPTER_PATTERNS, NUMBERED_HEADING_PATTERN

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Document map builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_document_map(
    doc,
    font_profile: FontProfile,
    heuristic_headings: list[BookmarkEntry],
) -> str:
    """Build a compact document map to send to the LLM.

    For each page, shows lines with their font properties.  Heuristic-detected
    headings are marked with a star.  Body text lines are truncated and limited
    to save tokens.

    A 500-page PDF typically compresses to ~5,000-15,000 tokens this way.
    """
    from .pdf_utils import extract_page_lines

    body_size = font_profile.body_font_size
    body_font = font_profile.body_font_name

    # Build lookup of heuristic headings by (text, page)
    heuristic_set: dict[tuple[str, int], BookmarkEntry] = {}
    for h in heuristic_headings:
        heuristic_set[(h.title.strip(), h.pdf_page_index)] = h

    lines_out: list[str] = []
    lines_out.append("DOCUMENT MAP")
    lines_out.append(f"Total pages: {len(doc)}")
    lines_out.append(f"Body text: {body_size}pt, font: {body_font}")
    lines_out.append("=" * 60)

    max_body_lines_per_page = 5
    max_body_text_len = 80

    current_page = -1
    body_lines_shown = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_lines = extract_page_lines(page, page_idx)

        for line in page_lines:
            text = line.text.strip()
            if not text:
                continue

            if page_idx != current_page:
                current_page = page_idx
                body_lines_shown = 0
                lines_out.append(f"\n--- PAGE {page_idx + 1} ---")

            dominant_size = line.dominant_font_size
            is_bold = line.is_mostly_bold
            is_uppercase = text.upper() == text and any(c.isalpha() for c in text)

            dominant_font = ""
            max_chars = 0
            for span in line.spans:
                sc = len(span.text.strip())
                if sc > max_chars:
                    max_chars = sc
                    dominant_font = span.font_name

            is_notable = (
                dominant_size > body_size * 1.03
                or is_bold
                or is_uppercase
                or dominant_font != body_font
                or (text, page_idx) in heuristic_set
            )

            # Build property string
            props = [f"{dominant_size}pt"]
            if is_bold:
                props.append("BOLD")
            if line.spans and any(s.is_italic for s in line.spans if s.text.strip()):
                props.append("italic")
            if is_uppercase:
                props.append("ALLCAPS")
            if dominant_font != body_font:
                props.append(f"font:{dominant_font}")
            prop_str = ", ".join(props)

            # Mark heuristic detections
            marker = ""
            h = heuristic_set.get((text, page_idx))
            if h is not None:
                marker = f" ★ HEURISTIC_HEADING L{h.level}"

            if is_notable:
                display = text[:117] + "..." if len(text) > 120 else text
                lines_out.append(f"  [{prop_str}] {display}{marker}")
            else:
                if body_lines_shown < max_body_lines_per_page:
                    display = text[:max_body_text_len - 3] + "..." if len(text) > max_body_text_len else text
                    lines_out.append(f"  [{prop_str}] {display}")
                    body_lines_shown += 1
                elif body_lines_shown == max_body_lines_per_page:
                    lines_out.append("  [...more body text on this page...]")
                    body_lines_shown += 1

    return "\n".join(lines_out)


def build_llm_prompt(
    document_map: str,
    heuristic_headings: list[BookmarkEntry],
    font_profile: FontProfile,
) -> str:
    """Build the prompt that asks the LLM to validate and improve heading detection."""
    if heuristic_headings:
        summary = "HEURISTIC STAGE DETECTED THESE HEADINGS:\n"
        for h in heuristic_headings:
            summary += f'  Level {h.level} | Page {h.pdf_page_index + 1} | Confidence {h.confidence:.2f} | "{h.title}"\n'
    else:
        summary = "HEURISTIC STAGE FOUND NO HEADINGS (this may be a flat-formatted document).\n"

    return f"""You are an expert document structure analyst. Your job is to identify the complete heading hierarchy (Table of Contents) of a PDF document.

I have already run a heuristic analysis that detects headings based on font size, boldness, and patterns. Now I need you to REVIEW and IMPROVE those results.

DOCUMENT INFORMATION:
- Body text is {font_profile.body_font_size}pt, font: {font_profile.body_font_name}
- Total pages: {len(document_map.splitlines())}

{summary}

Below is a condensed map of the document. Each line shows its font properties in brackets.
Lines marked with ★ HEURISTIC_HEADING were detected by the heuristic stage.
Body text lines are truncated for brevity.

{document_map}

YOUR TASK:
1. VALIDATE each heuristic heading — confirm it's really a heading (not a subtitle, caption, or list item)
2. FIND MISSED HEADINGS — look for headings the heuristics missed, especially:
   - Headings with the same font size as body text but different meaning
   - Section titles that only differ by being bold or italic
   - Headings in any language
   - "Introduction", "Conclusion", "References", etc. that may look like body text
3. REMOVE FALSE POSITIVES — reject lines that aren't really headings (e.g., subtitles on a title page, figure captions)
4. ASSIGN CORRECT LEVELS — Level 1 = major chapter/part, Level 2 = section, Level 3 = subsection, Level 4 = sub-subsection

IMPORTANT RULES:
- Only include actual structural headings — NOT figure captions, table titles, page numbers, or running headers
- Levels must be 1-4 only
- The first heading should be level 1
- No level can jump by more than 1 from the previous heading (e.g., L1 → L3 is NOT allowed; L1 → L2 → L3 is fine)
- Keep heading text EXACTLY as it appears in the document — do not paraphrase or modify it

Respond ONLY with a JSON array. No other text, no explanation, no markdown backticks. Each element should have:
- "text": the exact heading text
- "level": integer 1-4
- "page": 1-indexed page number
- "action": "keep" (confirmed heuristic heading), "add" (new heading you found), or "remove" (false positive to exclude)

Example format:
[
  {{"text": "Chapter 1: Introduction", "level": 1, "page": 2, "action": "keep"}},
  {{"text": "1.1 Background", "level": 2, "page": 2, "action": "keep"}},
  {{"text": "Summary", "level": 2, "page": 15, "action": "add"}},
  {{"text": "A Comprehensive Guide", "level": 2, "page": 1, "action": "remove"}}
]"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM API call
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, model: str = "claude-sonnet-4-20250514") -> Optional[str]:
    """Call the Anthropic API. Returns the response text or None on failure."""
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "'anthropic' package not installed. Install with: pip install anthropic  "
            "Falling back to heuristic-only mode."
        )
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Falling back to heuristic-only mode. "
            "Set it with: export ANTHROPIC_API_KEY=sk-..."
        )
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Calling LLM ({model})...")
        start = time.time()

        response = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        elapsed = time.time() - start
        logger.info(f"LLM responded in {elapsed:.1f}s")

        result = ""
        for block in response.content:
            if hasattr(block, "text"):
                result += block.text
        return result

    except Exception as e:
        logger.warning(f"LLM call failed: {e}. Falling back to heuristic-only mode.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Response parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_llm_response(response_text: str) -> Optional[list[dict]]:
    """Parse the LLM's JSON response, handling common quirks."""
    if not response_text:
        return None

    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Fix trailing commas (common LLM mistake)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        logger.warning(f"LLM returned {type(data).__name__} instead of list")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}. First 200 chars: {text[:200]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge heuristic + LLM results
# ═══════════════════════════════════════════════════════════════════════════════

def merge_heuristic_and_llm(
    heuristic_headings: list[BookmarkEntry],
    llm_data: list[dict],
    doc,
) -> list[BookmarkEntry]:
    """Merge heuristic and LLM results into a final heading list.

    - "keep"   -> keep heuristic heading, boost confidence, use LLM's level
    - "add"    -> add new heading found by LLM
    - "remove" -> drop the heuristic heading (false positive)
    - Heuristic headings not mentioned by LLM -> keep with lower confidence
    """
    from .pdf_utils import extract_page_lines
    from .font_heuristics import _normalize_hierarchy_levels

    final: list[BookmarkEntry] = []

    # Index heuristic headings by (text_lower, page_index)
    heuristic_map: dict[tuple[str, int], BookmarkEntry] = {}
    for h in heuristic_headings:
        heuristic_map[(h.title.strip().lower(), h.pdf_page_index)] = h

    # Build a (text_lower, page_index) -> y_position lookup from the PDF
    line_lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for line in extract_page_lines(page, page_idx):
            text = line.text.strip()
            if text:
                key = (text.lower(), page_idx)
                if key not in line_lookup:
                    line_lookup[key] = (
                        (line.bbox[1] + line.bbox[3]) / 2,
                        line.dominant_font_size,
                    )

    mentioned_keys: set[tuple[str, int]] = set()

    for item in llm_data:
        text = item.get("text", "").strip()
        level = item.get("level", 2)
        page = item.get("page", 1) - 1  # Convert to 0-indexed
        action = item.get("action", "keep").lower()

        level = max(1, min(level, 4))

        key = (text.lower(), page)
        mentioned_keys.add(key)

        if action == "remove":
            continue

        if action == "keep" and key in heuristic_map:
            h = heuristic_map[key]
            h.level = level
            h.confidence = min(h.confidence + 0.2, 1.0)
            h.source = "both"
            final.append(h)
        elif action in ("add", "keep"):
            lookup_key = (text.lower(), page)
            info = line_lookup.get(lookup_key)
            final.append(BookmarkEntry(
                title=text,
                page_number=page + 1,
                pdf_page_index=page,
                level=level,
                confidence=0.75,
                source="llm",
            ))

    # Keep unmentioned heuristic headings with lower confidence
    for key, h in heuristic_map.items():
        if key not in mentioned_keys:
            h.confidence = max(h.confidence - 0.1, 0.3)
            final.append(h)

    # Sort by position
    final.sort(key=lambda h: (h.pdf_page_index, h.page_number))

    # Normalize hierarchy
    final = _normalize_hierarchy_levels(final)

    return final


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def llm_review_headings(
    doc,
    font_profile: FontProfile,
    heuristic_headings: list[BookmarkEntry],
    model: str = "claude-sonnet-4-20250514",
) -> Optional[list[BookmarkEntry]]:
    """Run LLM review on heuristic headings. Returns improved list or None on failure.

    This is the single entry point the pipeline calls. It:
    1. Builds a compressed document map
    2. Constructs the LLM prompt
    3. Calls the Anthropic API
    4. Parses the JSON response
    5. Merges heuristic + LLM results
    """
    logger.info("Building document map for LLM...")
    doc_map = build_document_map(doc, font_profile, heuristic_headings)
    token_estimate = len(doc_map.split())
    logger.info(f"Document map: ~{token_estimate} words (~{int(token_estimate * 1.3)} tokens)")

    prompt = build_llm_prompt(doc_map, heuristic_headings, font_profile)

    response_text = call_llm(prompt, model=model)
    if not response_text:
        return None

    logger.info("Parsing LLM response...")
    llm_data = parse_llm_response(response_text)
    if not llm_data:
        return None

    actions = Counter(item.get("action", "keep") for item in llm_data)
    logger.info(
        f"LLM returned {len(llm_data)} entries: "
        + ", ".join(f"{action}={count}" for action, count in actions.most_common())
    )

    logger.info("Merging heuristic + LLM results...")
    merged = merge_heuristic_and_llm(heuristic_headings, llm_data, doc)
    logger.info(f"Final heading count after LLM review: {len(merged)}")

    return merged
