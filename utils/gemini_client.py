"""
OpenRouter API client.

Sends the user's question plus retrieved PDF context to an OpenRouter-hosted
model and returns a natural-language answer grounded in that context.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


def _get_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointed at OpenRouter.

    Returns:
        A configured OpenAI client instance.

    Raises:
        ValueError: If OPENROUTER_API_KEY is missing from the environment.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file before running the app."
        )

    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Ask the model to answer a question using only the retrieved PDF context.

    Args:
        question: The user's question about the PDF.
        context_chunks: Top relevant text chunks from FAISS retrieval.

    Returns:
        The generated answer as a string.
    """
    if not context_chunks:
        return "No relevant context was found in the PDF. Please upload a document first."

    # Number each chunk so the model can refer to sources clearly.
    formatted_context = "\n\n".join(
        f"Context {index + 1}:\n{chunk}"
        for index, chunk in enumerate(context_chunks)
    )

    prompt = f"""You are a helpful assistant that answers questions about a PDF document.

Use ONLY the context below to answer the question. If the answer is not in the context,
say clearly that the information is not available in the uploaded PDF.

Context:
{formatted_context}

Question: {question}

Answer in clear, beginner-friendly language. Be concise but complete."""

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"OpenRouter API request failed: {exc}") from exc
