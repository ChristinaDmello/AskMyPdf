"""
FAISS vector store operations.

FAISS (Facebook AI Similarity Search) stores embeddings and quickly finds
the vectors closest to a query embedding.
"""

import os
import pickle

import faiss
import numpy as np


class FaissStore:
    """
    Wrapper around a FAISS index plus the original text chunks.

    Keeps vectors and chunk text together so search results can return
    human-readable context, not just numeric indices.
    """

    def __init__(self, dimension: int):
        """
        Create an empty FAISS index.

        Args:
            dimension: Size of each embedding vector (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        # IndexFlatL2 uses Euclidean distance — simple and good for learning.
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[str] = []

    def add_embeddings(self, embeddings: list[list[float]], chunks: list[str]) -> None:
        """
        Add embedding vectors and their corresponding text chunks to the index.

        Args:
            embeddings: List of embedding vectors.
            chunks: Original text for each embedding (same order and length).
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")

        if not embeddings:
            raise ValueError("Cannot add an empty list of embeddings.")

        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: Embedding vector for the user's question.
            top_k: How many nearest chunks to return.

        Returns:
            List of dicts with keys: rank, score, chunk.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for rank, (distance, chunk_index) in enumerate(
            zip(distances[0], indices[0]), start=1
        ):
            if chunk_index == -1:
                continue

            results.append(
                {
                    "rank": rank,
                    "score": float(distance),
                    "chunk": self.chunks[chunk_index],
                }
            )

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Args:
            index_path: File path for the FAISS index (.faiss).
            metadata_path: File path for chunk text and dimension (.pkl).
        """
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as file:
            pickle.dump(
                {"chunks": self.chunks, "dimension": self.dimension},
                file,
            )

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FaissStore":
        """
        Load a previously saved FAISS index and its chunk metadata.

        Args:
            index_path: Path to the saved FAISS index file.
            metadata_path: Path to the saved metadata pickle file.

        Returns:
            A FaissStore instance ready for search.
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Saved FAISS index or metadata file is missing.")

        with open(metadata_path, "rb") as file:
            metadata = pickle.load(file)

        store = cls(dimension=metadata["dimension"])
        store.index = faiss.read_index(index_path)
        store.chunks = metadata["chunks"]
        return store
