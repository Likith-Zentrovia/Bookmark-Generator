"""Command-line interface for the bookmark generator."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import extract_bookmarks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bookmark-generator",
        description="Extract and reconstruct chapter structure from PDFs.",
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output PDF path (writes bookmarks into the PDF). "
             "If not specified, prints the bookmark tree to stdout.",
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Inject extracted bookmarks into the PDF. "
             "Uses --output path or overwrites the input file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of bookmarks even if the PDF already has them.",
    )
    parser.add_argument(
        "--no-toc",
        action="store_true",
        help="Disable TOC page parsing.",
    )
    parser.add_argument(
        "--no-font",
        action="store_true",
        help="Disable font-based heading detection.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision-based TOC extraction (requires ANTHROPIC_API_KEY). "
             "Renders PDF pages as images and uses Claude Vision to extract "
             "TOC entries with cross-verification against actual pages.",
    )
    parser.add_argument(
        "--vision-model",
        default="claude-sonnet-4-20250514",
        metavar="MODEL",
        help="Anthropic model to use for vision extraction. Default: claude-sonnet-4-20250514.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM review stage (requires ANTHROPIC_API_KEY). "
             "The LLM validates heuristic headings, finds missed ones, "
             "and removes false positives.",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-sonnet-4-20250514",
        metavar="MODEL",
        help="Anthropic model to use for LLM review. Default: claude-sonnet-4-20250514.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=70,
        metavar="N",
        help="Fuzzy match threshold (0-100) for TOC-to-heading reconciliation. Default: 70.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of a tree.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    # Fix Windows terminal encoding for Unicode tree characters
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    # Configure logging — only our loggers get DEBUG, suppress noisy third-party libs
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=logging.WARNING,  # Default to WARNING for all loggers
        format="%(levelname)s: %(message)s",
    )
    # Set our package loggers to the requested level
    logging.getLogger("bookmark_generator").setLevel(log_level)
    # Keep third-party loggers (anthropic, httpx, httpcore) quiet
    for noisy_logger in ("anthropic", "httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    try:
        result = extract_bookmarks(
            pdf_path=args.pdf_path,
            force_rebuild=args.force,
            use_toc=not args.no_toc,
            use_font_heuristics=not args.no_font,
            use_llm=args.llm,
            use_vision=args.vision,
            vision_model=args.vision_model,
            llm_model=args.llm_model,
            inject_into_pdf=args.inject,
            output_path=args.output,
            fuzzy_threshold=args.fuzzy_threshold,
        )
    except FileNotFoundError:
        print(f"Error: File not found: {args.pdf_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output results
    if args.output_json:
        output = {
            "pdf_path": result.pdf_path,
            "total_pages": result.total_pages,
            "method_used": result.method_used,
            "warnings": result.warnings,
            "bookmark_count": len(result.flat_bookmarks()),
            "bookmarks": [b.to_dict() for b in result.bookmarks],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"PDF: {result.pdf_path}")
        print(f"Pages: {result.total_pages}")
        print(f"Method: {result.method_used}")
        if result.warnings:
            for w in result.warnings:
                print(f"  Warning: {w}")
        print(f"Bookmarks ({len(result.flat_bookmarks())} entries):")
        print()
        tree = result.print_tree()
        if tree:
            print(tree)
        else:
            print("  (no bookmarks found)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
