# Bookmark Generator

Extract and reconstruct chapter structure from PDFs — even when bookmarks are missing.

Automatically detects chapter headings from Table of Contents pages, font analysis, and optionally uses **Claude Vision AI** for high-accuracy extraction with cross-verification.

## Features

- **Multi-method extraction** — TOC parsing, font heuristics, and Vision AI working together
- **Claude Vision AI** — Renders PDF pages as images, reads TOC entries with 95%+ accuracy, and cross-verifies every entry against the actual page
- **Web-based editor** — Interactive UI to view, edit, reorder, and export bookmarks
- **CLI tool** — Quick command-line extraction with JSON or tree output
- **Smart reconciliation** — Fuzzy-matches TOC entries against body headings for accurate page mapping
- **Hierarchy detection** — Automatically builds nested bookmark trees (chapters, sections, subsections)

## Prerequisites

- **Python 3.9+**
- **Anthropic API key** (optional, required only for `--vision` and `--llm` features)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Bookmark-Generator.git
cd Bookmark-Generator
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install as an editable package (registers the `bookmark-generator` CLI command):

```bash
pip install -e .
```

### 4. Set up the API key (optional — needed for Vision/LLM features)

Copy the example environment file and add your Anthropic API key:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder with your actual key from [console.anthropic.com](https://console.anthropic.com/):

```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

> Without an API key, the tool still works using regex-based TOC parsing and font heuristics. The API key is only needed for the `--vision` and `--llm` flags (CLI) or the "Vision Extract" button (Web UI).

## Quick Start

### Option A: Web UI (recommended for interactive use)

```bash
python -m bookmark_generator.web
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

1. **Upload a PDF** — drag & drop or enter a local file path
2. **Auto-extraction** — bookmarks are extracted automatically using TOC + font analysis
3. **Vision Extract** — click the "Vision Extract" button for AI-powered extraction (requires API key)
4. **Edit** — click titles to rename, drag to reorder, use Tab/Shift+Tab to indent/outdent
5. **Export** — click "Export PDF" to download the PDF with bookmarks injected

### Option B: CLI

```bash
# Print the bookmark tree
bookmark-generator document.pdf

# Verbose output (shows extraction details)
bookmark-generator document.pdf -v

# Output as JSON
bookmark-generator document.pdf --json

# Inject bookmarks into a new PDF
bookmark-generator document.pdf --inject -o output.pdf

# Use Vision AI for high-accuracy extraction
bookmark-generator document.pdf --vision --inject -o output.pdf

# Force rebuild (ignore existing bookmarks in the PDF)
bookmark-generator document.pdf --force
```

### Option C: Python API

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

# Use Vision AI extraction
result = extract_bookmarks(
    "document.pdf",
    use_vision=True,
    inject_into_pdf=True,
    output_path="output_with_bookmarks.pdf",
)
```

## How It Works

The pipeline uses a layered approach to extract bookmarks:

```
PDF Input
    |
    v
1. Existing Bookmarks  ──>  If PDF has bookmarks, use them directly
    |
    v
2. Page Number Mapping  ──>  Detect logical-to-physical page offset
    |
    v
3. Vision AI (optional) ──>  Render TOC pages as images, extract entries
    |                        with Claude Vision, cross-verify against
    |                        actual pages (95%+ accuracy)
    |
    v
4. TOC Page Parsing     ──>  Regex + font analysis on TOC pages
    |
    v
5. Font Heuristics      ──>  Scan body text for heading-like fonts
    |
    v
6. Reconciliation       ──>  Fuzzy-match TOC entries to body headings,
    |                        build hierarchical tree
    v
7. LLM Review (optional) ──> Validate headings with Claude AI
    |
    v
Bookmark Tree Output
```

### Vision AI Extraction (3-Phase Process)

When `--vision` is enabled, the tool uses Claude's vision capabilities:

1. **Discovery** — Finds TOC pages (regex first, vision fallback)
2. **Extraction** — Reads every entry from TOC page images (title, page number, hierarchy level)
3. **Cross-verification** — Renders each target page and confirms the heading exists there, correcting page numbers when needed

This achieves 95%+ verification accuracy on real-world textbooks.

## CLI Reference

```
bookmark-generator [OPTIONS] PDF_PATH
```

| Flag | Description |
|---|---|
| `PDF_PATH` | Path to the input PDF file (required) |
| `-o, --output PATH` | Output PDF path for injected bookmarks |
| `--inject` | Inject bookmarks into the PDF |
| `--force` | Force rebuild even if PDF has existing bookmarks |
| `--no-toc` | Disable TOC page parsing |
| `--no-font` | Disable font-based heading detection |
| `--vision` | Enable Vision AI extraction (requires `ANTHROPIC_API_KEY`) |
| `--vision-model MODEL` | Anthropic model for vision (default: `claude-sonnet-4-20250514`) |
| `--llm` | Enable LLM review stage (requires `ANTHROPIC_API_KEY`) |
| `--llm-model MODEL` | Anthropic model for LLM review (default: `claude-sonnet-4-20250514`) |
| `--fuzzy-threshold N` | Fuzzy match threshold 0-100 (default: 70) |
| `--json` | Output results as JSON |
| `-v, --verbose` | Enable debug logging |

## Web UI Features

The web editor at `http://127.0.0.1:5000` provides:

- **Drag & drop** PDF upload
- **Real-time progress** bar during Vision AI extraction
- **Interactive bookmark tree** with expand/collapse
- **Inline editing** — click to rename titles, edit page numbers
- **Drag & drop reorder** — rearrange bookmarks freely
- **Keyboard shortcuts**:
  - `Ctrl+Z` / `Ctrl+Shift+Z` — Undo / Redo
  - `Tab` / `Shift+Tab` — Indent / Outdent selected bookmark
  - `F2` — Edit selected bookmark title
  - `Del` — Delete selected bookmark
  - `Arrow keys` — Navigate tree and pages
- **Confidence badges** — color-coded verification confidence per entry
- **Source labels** — shows how each bookmark was detected (Vision, TOC, Font, etc.)
- **JSON save/load** — export and import bookmark sets
- **PDF export** — download the final PDF with bookmarks injected

## Project Structure

```
Bookmark-Generator/
├── bookmark_generator/
│   ├── __init__.py            # Package init
│   ├── cli.py                 # CLI entry point
│   ├── pipeline.py            # Main extraction orchestrator
│   ├── vision_extract.py      # Claude Vision AI extraction
│   ├── toc_parser.py          # Regex-based TOC parsing
│   ├── font_heuristics.py     # Font-based heading detection
│   ├── reconciler.py          # TOC + heading reconciliation
│   ├── llm_review.py          # LLM validation stage
│   ├── bookmark_extractor.py  # Extract/inject PDF bookmarks
│   ├── pdf_utils.py           # PDF utility functions
│   ├── models.py              # Data models
│   └── web/
│       ├── __init__.py
│       ├── __main__.py        # Web app entry point
│       ├── app.py             # Flask application
│       └── templates/
│           └── index.html     # Single-page frontend
├── tests/
│   └── test_pipeline.py       # Test suite (52 tests)
├── .env.example               # API key template
├── .gitignore
├── pyproject.toml              # Project config & dependencies
├── requirements.txt            # Pip requirements
└── README.md
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v
```

## Dependencies

| Package | Purpose |
|---|---|
| **PyMuPDF** | PDF text extraction, font metadata, page rendering, bookmark I/O |
| **rapidfuzz** | Fuzzy string matching for TOC-to-heading reconciliation |
| **python-dotenv** | Load API keys from `.env` file |
| **Flask** | Web UI backend server |
| **anthropic** | *(optional)* Claude API client for Vision and LLM features |

## Troubleshooting

### "ANTHROPIC_API_KEY not found" in the Web UI
Make sure you have a `.env` file in the project root with your API key. Restart the web server after creating it.

### Vision extraction takes a long time
This is normal for large PDFs. The cross-verification phase checks each entry against the actual PDF page, requiring ~125 API calls for a 600-entry TOC. The web UI shows a real-time progress bar with phase indicators and elapsed time. Typical times:
- Small PDF (< 50 pages): 1-2 minutes
- Medium PDF (100-200 pages): 5-10 minutes
- Large PDF (500+ pages): 15-20 minutes

### "No module named bookmark_generator"
Make sure you installed the package. Run `pip install -e .` from the project root.

### Windows encoding issues
The CLI automatically handles Windows terminal encoding. If you see garbled characters, try running in Windows Terminal or PowerShell 7+.

## License

MIT
