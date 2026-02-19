# Bookmark Generator

Extract and reconstruct chapter structure from PDFs — even when bookmarks are missing.

## How It Works

The pipeline uses a layered approach:

1. **Existing bookmarks** — If the PDF already has an outline, use it directly.
2. **TOC page parsing** — Locate the Table of Contents page, extract entries using regex pattern matching and font metadata analysis.
3. **Font-based heading detection** — Scan the document body for headings based on font size, weight, and text characteristics.
4. **Reconciliation** — Fuzzy-match TOC entries against body headings, merge signals, resolve page numbers, and build a hierarchical bookmark tree.

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Print the bookmark tree
bookmark-generator document.pdf

# Verbose output
bookmark-generator document.pdf -v

# Output as JSON
bookmark-generator document.pdf --json

# Inject bookmarks into a new PDF
bookmark-generator document.pdf --inject -o output.pdf

# Force rebuild (ignore existing bookmarks)
bookmark-generator document.pdf --force

# Disable specific methods
bookmark-generator document.pdf --no-toc       # skip TOC parsing
bookmark-generator document.pdf --no-font      # skip font heuristics
```

### Python API

```python
from bookmark_generator.pipeline import extract_bookmarks

result = extract_bookmarks("document.pdf")

print(f"Method: {result.method_used}")
print(f"Bookmarks found: {len(result.flat_bookmarks())}")
print(result.print_tree())

# Inject bookmarks into the PDF
result = extract_bookmarks(
    "document.pdf",
    inject_into_pdf=True,
    output_path="output_with_bookmarks.pdf",
)
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Dependencies

- **PyMuPDF** — PDF text extraction, font metadata, bookmark I/O
- **rapidfuzz** — Fuzzy string matching for TOC-to-heading reconciliation
