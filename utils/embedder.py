"""
Embedding generation using Sentence Transformers.

The all-MiniLM-L6-v2 model converts text into dense vectors that capture
semantic meaning, which FAISS uses for similarity search.
"""

from sentence_transformers import SentenceTransformer


# Load the model once when the module is imported (saves reload time).
MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings (document chunks or a single question).

    Returns:
        List of embedding vectors (each vector is a list of floats).
    """
    if not texts:
        return []

    # encode() returns a NumPy array; we convert to plain Python lists
    # so the result is easy to serialize and store.
    embeddings = _embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def generate_embedding(text: str) -> list[float]:
    """
    Generate a single embedding vector for one piece of text.

    Args:
        text: A question or short text snippet.

    Returns:
        One embedding vector as a list of floats.
    """
    return generate_embeddings([text])[0]
