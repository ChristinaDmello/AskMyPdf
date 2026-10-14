"""
Text chunking utilities.

Long documents are split into smaller chunks so embeddings stay focused
and retrieval can find the most relevant passages.
"""


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks of roughly equal size.

    We use character-based chunking because it is simple and works well
    for internship demos. Overlap helps avoid cutting sentences in half
    at chunk boundaries.

    Args:
        text: Full document text.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of characters repeated between consecutive chunks.

    Returns:
        List of text chunks ready for embedding.
    """
    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward, keeping overlap so context is not lost at boundaries.
        start += chunk_size - chunk_overlap

        if start >= text_length:
            break

    return chunks
