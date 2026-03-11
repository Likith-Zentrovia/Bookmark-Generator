#!/usr/bin/env python3
"""Launch the PDF Bookmark Editor web UI.

Usage:
    python run_editor.py                    # Open empty editor
    python run_editor.py my_document.pdf    # Pre-load a PDF
    python run_editor.py --port 8080        # Custom port
    python run_editor.py --no-browser       # Don't auto-open browser
"""

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

# Load .env file from project root
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="PDF Bookmark Editor - Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to a PDF file to pre-load on startup",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser",
    )
    args = parser.parse_args()

    # Configure logging so vision/extraction progress is visible in terminal
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key and not api_key.startswith("your-"):
        print(f"  API Key: loaded ({api_key[:12]}...)")
    else:
        print("  API Key: NOT SET (Vision Extract will not work)")
        print("           Add your key to .env file in project root")

    # Validate PDF path if provided
    if args.pdf_path:
        pdf_path = os.path.abspath(args.pdf_path)
        if not os.path.isfile(pdf_path):
            print(f"Error: File not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        if not pdf_path.lower().endswith(".pdf"):
            print(f"Warning: File may not be a PDF: {pdf_path}", file=sys.stderr)
    else:
        pdf_path = None

    # Import and create the Flask app
    from bookmark_generator.web.app import create_app

    app = create_app()

    if pdf_path:
        app.config["PRELOAD_PDF"] = pdf_path

    url = f"http://localhost:{args.port}"
    print(f"\n{'='*60}")
    print(f"  PDF Bookmark Editor")
    print(f"  Running at: {url}")
    if pdf_path:
        print(f"  Pre-loading: {os.path.basename(pdf_path)}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    # Auto-open browser after a short delay
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
