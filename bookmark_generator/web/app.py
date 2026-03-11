"""Flask web application for the PDF Bookmark Editor UI."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import fitz
from flask import Flask, jsonify, make_response, render_template, request, send_file

from bookmark_generator.bookmark_extractor import inject_bookmarks
from bookmark_generator.models import BookmarkEntry
from bookmark_generator.pdf_utils import render_page_to_png
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
        has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        return jsonify({"preload_pdf": preload, "has_api_key": has_api_key})

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
        """Run the bookmark extraction pipeline on a loaded PDF.

        For vision extraction, this starts the pipeline in a background thread
        and returns immediately with a task_id for progress polling.
        For non-vision extraction, runs synchronously as before.
        """
        data = request.get_json()
        file_id = data.get("file_id")

        if not file_id or file_id not in sessions:
            return jsonify(error="Unknown file_id"), 404

        session = sessions[file_id]
        use_vision = data.get("use_vision", False)

        if use_vision:
            # ── Async vision extraction ─────────────────────────────────
            task_id = str(uuid.uuid4())

            # Initialize progress tracking in session
            session["extract_task"] = {
                "task_id": task_id,
                "status": "running",
                "phase": "starting",
                "step": 0,
                "total_steps": 1,
                "message": "Starting vision extraction...",
                "started_at": time.time(),
                "result": None,
                "error": None,
            }

            def progress_callback(phase: str, step: int, total_steps: int, message: str):
                """Called by vision_extract to report progress."""
                task = session.get("extract_task")
                if task and task["task_id"] == task_id:
                    task["phase"] = phase
                    task["step"] = step
                    task["total_steps"] = total_steps
                    task["message"] = message

            def run_extraction_thread():
                """Run extraction in background thread."""
                task = session.get("extract_task")
                try:
                    result = extract_bookmarks(
                        pdf_path=session["pdf_path"],
                        force_rebuild=data.get("force_rebuild", True),
                        use_toc=data.get("use_toc", True),
                        use_font_heuristics=data.get("use_font_heuristics", True),
                        use_llm=data.get("use_llm", False),
                        use_vision=True,
                        vision_model=data.get("vision_model", "claude-sonnet-4-20250514"),
                        llm_model=data.get("llm_model", "claude-sonnet-4-20250514"),
                        fuzzy_threshold=data.get("fuzzy_threshold", 70),
                        progress_callback=progress_callback,
                    )

                    bookmarks_json = _bookmarks_to_json(result.bookmarks)
                    session["bookmarks"] = bookmarks_json

                    if task:
                        task["status"] = "completed"
                        task["phase"] = "done"
                        task["message"] = f"Extracted {len(result.flat_bookmarks())} bookmarks"
                        task["result"] = {
                            "bookmarks": bookmarks_json,
                            "total_pages": result.total_pages,
                            "method_used": result.method_used,
                            "warnings": result.warnings,
                            "bookmark_count": len(result.flat_bookmarks()),
                        }

                except Exception as e:
                    logger.exception("Vision extraction failed")
                    if task:
                        task["status"] = "error"
                        task["error"] = str(e)
                        task["message"] = f"Error: {e}"

            thread = threading.Thread(target=run_extraction_thread, daemon=True)
            thread.start()

            return jsonify({
                "async": True,
                "task_id": task_id,
                "message": "Vision extraction started",
            })

        else:
            # ── Synchronous non-vision extraction ───────────────────────
            try:
                result = extract_bookmarks(
                    pdf_path=session["pdf_path"],
                    force_rebuild=data.get("force_rebuild", True),
                    use_toc=data.get("use_toc", True),
                    use_font_heuristics=data.get("use_font_heuristics", True),
                    use_llm=data.get("use_llm", False),
                    use_vision=False,
                    vision_model=data.get("vision_model", "claude-sonnet-4-20250514"),
                    llm_model=data.get("llm_model", "claude-sonnet-4-20250514"),
                    fuzzy_threshold=data.get("fuzzy_threshold", 70),
                )

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

    @app.route("/api/extract/progress", methods=["POST"])
    def get_extraction_progress():
        """Poll for vision extraction progress."""
        data = request.get_json()
        file_id = data.get("file_id")
        task_id = data.get("task_id")

        if not file_id or file_id not in sessions:
            return jsonify(error="Unknown file_id"), 404

        session = sessions[file_id]
        task = session.get("extract_task")

        if not task or task.get("task_id") != task_id:
            return jsonify(error="Unknown task_id"), 404

        elapsed = time.time() - task.get("started_at", time.time())

        response = {
            "status": task["status"],
            "phase": task["phase"],
            "step": task["step"],
            "total_steps": task["total_steps"],
            "message": task["message"],
            "elapsed_seconds": round(elapsed, 1),
        }

        if task["status"] == "completed":
            response["result"] = task["result"]
        elif task["status"] == "error":
            response["error"] = task["error"]

        return jsonify(response)

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
            img_bytes = render_page_to_png(page, width=width)

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
