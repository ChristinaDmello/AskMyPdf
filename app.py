"""
Flask web application for RAG-based PDF Question Answering.

Routes:
- GET  /           : Main page (upload + ask question UI)
- POST /upload     : Upload and process a PDF
- POST /ask        : Ask a question about the uploaded PDF
"""

import os
import uuid

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from utils.rag_pipeline import answer_question, process_pdf

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Folder configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Return True if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render the main page with upload and question forms."""
    return render_template(
        "index.html",
        document_id=session.get("document_id"),
        chunk_count=session.get("chunk_count"),
        filename=session.get("filename"),
    )


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    Handle PDF upload: save file, run RAG ingestion pipeline,
    and store document_id in the user session.
    """
    if "pdf_file" not in request.files:
        flash("No file was selected. Please choose a PDF to upload.", "error")
        return redirect(url_for("index"))

    file = request.files["pdf_file"]

    if file.filename == "":
        flash("No file was selected. Please choose a PDF to upload.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only PDF files are allowed.", "error")
        return redirect(url_for("index"))

    try:
        safe_filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4()}_{safe_filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        result = process_pdf(save_path)

        session["document_id"] = result["document_id"]
        session["chunk_count"] = result["chunk_count"]
        session["filename"] = safe_filename

        flash(result["message"], "success")

    except Exception as exc:
        flash(f"Failed to process PDF: {exc}", "error")

    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handle a user question: retrieve context from FAISS and
    generate an answer with an LLM via OpenRouter.
    """
    document_id = session.get("document_id")
    question = request.form.get("question", "").strip()

    if not document_id:
        flash("Please upload a PDF before asking a question.", "error")
        return redirect(url_for("index"))

    if not question:
        flash("Please enter a question.", "error")
        return redirect(url_for("index"))

    try:
        result = answer_question(document_id, question, top_k=3)

        return render_template(
            "index.html",
            document_id=document_id,
            chunk_count=session.get("chunk_count"),
            filename=session.get("filename"),
            question=result["question"],
            answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"],
        )

    except Exception as exc:
        flash(f"Failed to generate answer: {exc}", "error")
        return redirect(url_for("index"))


@app.errorhandler(413)
def file_too_large(_error):
    """Show a friendly message instead of the default 413 error page."""
    flash("File is too large. Please upload a PDF under 16 MB.", "error")
    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear the current document from session so user can upload a new PDF."""
    session.pop("document_id", None)
    session.pop("chunk_count", None)
    session.pop("filename", None)
    flash("Session cleared. You can upload a new PDF.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
