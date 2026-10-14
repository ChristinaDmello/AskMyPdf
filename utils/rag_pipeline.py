"""
RAG (Retrieval-Augmented Generation) pipeline.

This module ties together all RAG steps:
1. Extract PDF text
2. Chunk the text
3. Generate embeddings
4. Store in FAISS
5. Retrieve relevant chunks for a question
6. Generate an answer with an LLM via OpenRouter
"""

import os
import uuid

from utils.chunker import split_text_into_chunks
from utils.embedder import generate_embedding, generate_embeddings
from utils.faiss_store import FaissStore
from utils.gemini_client import generate_answer
from utils.pdf_extractor import extract_text_from_pdf

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIMENSION = 384

# Where processed indexes are saved (relative to project root)
PROCESSED_DIR = "uploads/processed"


def _ensure_processed_dir() -> str:
    """Create the processed storage folder if it does not exist."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    return PROCESSED_DIR


def process_pdf(pdf_path: str) -> dict:
    """
    Full ingestion pipeline: extract, chunk, embed, and index a PDF.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        Dictionary with document_id, chunk_count, and a success message.
    """
    # Step 1: Extract text from the PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Step 2: Split into chunks suitable for embedding
    chunks = split_text_into_chunks(raw_text)

    if not chunks:
        raise ValueError("Could not create any text chunks from the PDF.")

    # Step 3: Generate embeddings for every chunk
    embeddings = generate_embeddings(chunks)

    # Step 4: Build and populate the FAISS index
    store = FaissStore(dimension=EMBEDDING_DIMENSION)
    store.add_embeddings(embeddings, chunks)

    # Step 5: Save index and metadata for later questions
    document_id = str(uuid.uuid4())
    processed_dir = _ensure_processed_dir()
    index_path = os.path.join(processed_dir, f"{document_id}.faiss")
    metadata_path = os.path.join(processed_dir, f"{document_id}.pkl")

    store.save(index_path, metadata_path)

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "message": f"PDF processed successfully. Created {len(chunks)} searchable chunks.",
    }


def answer_question(document_id: str, question: str, top_k: int = 3) -> dict:
    """
    RAG query pipeline: retrieve context and generate an answer.

    Args:
        document_id: ID returned when the PDF was uploaded.
        question: User's question about the document.
        top_k: Number of chunks to retrieve from FAISS (default: 3).

    Returns:
        Dictionary with answer, retrieved_chunks, and question.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    processed_dir = _ensure_processed_dir()
    index_path = os.path.join(processed_dir, f"{document_id}.faiss")
    metadata_path = os.path.join(processed_dir, f"{document_id}.pkl")

    # Load the saved FAISS index for this document
    store = FaissStore.load(index_path, metadata_path)

    # Step 7: Convert the question into an embedding
    question_embedding = generate_embedding(question)

    # Step 8: Retrieve the top-k most relevant chunks
    search_results = store.search(question_embedding, top_k=top_k)
    retrieved_chunks = [result["chunk"] for result in search_results]

    # Step 9: Send context + question to the LLM via OpenRouter
    answer = generate_answer(question, retrieved_chunks)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": search_results,
    }
