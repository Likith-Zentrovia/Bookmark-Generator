"""Flask web application for the PDF Bookmark Editor UI."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import fitz
from flask import Flask, jsonify, make_response, render_template, request, send_file

from bookmark_generator.bookmark_extractor import inject_bookmarks
from bookmark_generator.models import BookmarkEntry
from bookmark_generator.pipeline import extract_bookmarks

logger = logging.getLogger(__name__)

# ── In-memory session store (single-user local tool) ──────────────────────
sessions: dict[str, dict[str, Any]] = {}


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    # ── Routes ────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/config")
    def get_config():
        """Return startup config (e.g. preloaded PDF path)."""
        preload = app.config.get("PRELOAD_PDF")
        return jsonify({"preload_pdf": preload})

    @app.route("/api/upload", methods=["POST"])
    def upload_pdf():
        """Accept a PDF via file upload or local path. Returns file_id + metadata."""
        file_id = str(uuid.uuid4())

        try:
            # Check for local path in JSON body
            if request.is_json:
                data = request.get_json(force=True, silent=True) or {}

                local_path = data.get("local_path", "")
                if not local_path or not os.path.isfile(local_path):
                    return jsonify(error=f"File not found: {local_path}"), 400
                pdf_path = os.path.abspath(local_path)
                filename = os.path.basename(pdf_path)
            else:
                # File upload
                if "file" not in request.files:
                    return jsonify(error="No file provided"), 400
                f = request.files["file"]
                if not f.filename or not f.filename.lower().endswith(".pdf"):
                    return jsonify(error="Please upload a PDF file"), 400

                # Save to temp directory
                upload_dir = os.path.join(tempfile.gettempdir(), "bookmark_editor")
                os.makedirs(upload_dir, exist_ok=True)
                pdf_path = os.path.join(upload_dir, f"{file_id}_{f.filename}")
                f.save(pdf_path)
                filename = f.filename

            # Open the PDF
            doc = fitz.open(pdf_path)
            sessions[file_id] = {
                "pdf_path": pdf_path,
                "doc": doc,
                "total_pages": len(doc),
                "filename": filename,
                "bookmarks": [],
                "output_path": None,
            }

            return jsonify({
                "file_id": file_id,
                "filename": filename,
                "total_pages": len(doc),
            })

        except Exception as e:
            logger.exception("Upload failed")
            return jsonify(error=str(e)), 500

    @app.route("/api/extract", methods=["POST"])
    def run_extraction():
        """Run the bookmark extraction pipeline on a loaded PDF."""
        data = request.get_json()
        file_id = data.get("file_id")

        if not file_id or file_id not in sessions:
            return jsonify(error="Unknown file_id"), 404

        session = sessions[file_id]

        try:
            result = extract_bookmarks(
                pdf_path=session["pdf_path"],
                force_rebuild=data.get("force_rebuild", True),
                use_toc=data.get("use_toc", True),
                use_font_heuristics=data.get("use_font_heuristics", True),
                use_llm=data.get("use_llm", False),
                llm_model=data.get("llm_model", "claude-sonnet-4-20250514"),
                fuzzy_threshold=data.get("fuzzy_threshold", 70),
            )

            # Convert bookmarks to JSON-serializable format with unique IDs
            bookmarks_json = _bookmarks_to_json(result.bookmarks)
            session["bookmarks"] = bookmarks_json

            return jsonify({
                "bookmarks": bookmarks_json,
                "total_pages": result.total_pages,
                "method_used": result.method_used,
                "warnings": result.warnings,
                "bookmark_count": len(result.flat_bookmarks()),
            })

        except Exception as e:
            logger.exception("Extraction failed")
            return jsonify(error=str(e)), 500

    @app.route("/api/page/<file_id>/<int:page_index>")
    def get_page(file_id: str, page_index: int):
        """Render a single PDF page as PNG."""
        session = sessions.get(file_id)
        if not session:
            return jsonify(error="Unknown file_id"), 404

        doc = session["doc"]
        if page_index < 0 or page_index >= len(doc):
            return jsonify(error="Page index out of range"), 400

        width = request.args.get("width", 800, type=int)
        width = min(max(width, 200), 2400)

        try:
            page = doc[page_index]
            zoom = width / page.rect.width
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")

            response = make_response(img_bytes)
            response.headers["Content-Type"] = "image/png"
            response.headers["Cache-Control"] = "public, max-age=300"
            return response

        except Exception as e:
            logger.exception("Page render failed")
            return jsonify(error=str(e)), 500

    @app.route("/api/inject", methods=["POST"])
    def inject_into_pdf():
        """Inject edited bookmarks into the PDF and save."""
        data = request.get_json()
        file_id = data.get("file_id")
        bookmarks_data = data.get("bookmarks", [])
        output_filename = data.get("output_filename", "output_bookmarked.pdf")

        if not file_id or file_id not in sessions:
            return jsonify(error="Unknown file_id"), 404

        session = sessions[file_id]

        try:
            # Convert JSON tree to flat BookmarkEntry list
            flat_entries = _json_tree_to_flat_entries(bookmarks_data)

            if not flat_entries:
                return jsonify(error="No bookmarks to inject"), 400

            # Open a fresh copy of the PDF for injection
            doc = fitz.open(session["pdf_path"])
            inject_bookmarks(doc, flat_entries)

            # Save to output directory
            output_dir = os.path.join(tempfile.gettempdir(), "bookmark_editor", "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{file_id}_{output_filename}")
            doc.save(output_path, encryption=0)
            doc.close()

            session["output_path"] = output_path

            return jsonify({
                "success": True,
                "output_path": output_path,
                "bookmark_count": len(flat_entries),
            })

        except Exception as e:
            logger.exception("Injection failed")
            return jsonify(error=str(e)), 500

    @app.route("/api/download/<file_id>")
    def download_pdf(file_id: str):
        """Download the exported PDF."""
        session = sessions.get(file_id)
        if not session:
            return jsonify(error="Unknown file_id"), 404

        output_path = session.get("output_path")
        if not output_path or not os.path.isfile(output_path):
            return jsonify(error="No exported PDF available. Run export first."), 404

        return send_file(
            output_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=session.get("filename", "bookmarked.pdf").replace(
                ".pdf", "_bookmarked.pdf"
            ),
        )

    return app


# ── Helper functions ──────────────────────────────────────────────────────


def _bookmarks_to_json(bookmarks: list[BookmarkEntry]) -> list[dict]:
    """Convert BookmarkEntry tree to JSON-friendly dicts with unique IDs."""
    result = []
    for entry in bookmarks:
        node = {
            "id": str(uuid.uuid4()),
            "title": entry.title,
            "page_number": entry.page_number,
            "pdf_page_index": entry.pdf_page_index,
            "level": entry.level,
            "confidence": entry.confidence,
            "source": entry.source,
            "children": _bookmarks_to_json(entry.children),
        }
        result.append(node)
    return result


def _json_tree_to_flat_entries(
    nodes: list[dict], level: int = 1
) -> list[BookmarkEntry]:
    """Convert JSON bookmark tree to a flat list of BookmarkEntry objects."""
    result = []
    for node in nodes:
        entry = BookmarkEntry(
            title=node.get("title", "Untitled"),
            page_number=node.get("page_number", 1),
            pdf_page_index=node.get("pdf_page_index", 0),
            level=level,
            confidence=node.get("confidence", 1.0),
            source=node.get("source", "manual"),
        )
        result.append(entry)
        result.extend(
            _json_tree_to_flat_entries(node.get("children", []), level + 1)
        )
    return result
