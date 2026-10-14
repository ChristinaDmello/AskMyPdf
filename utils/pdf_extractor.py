"""
PDF text extraction using pdfplumber.

This module reads a PDF file and returns the full text content
as a single string, page by page.
"""

import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all readable text from a PDF file.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Combined text from all pages, separated by newlines.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text could be extracted from the PDF.
    """
    pages_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()

                if page_text:
                    # Add a page marker so chunks can reference source pages later.
                    pages_text.append(f"[Page {page_number}]\n{page_text.strip()}")

    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF: {exc}") from exc

    if not pages_text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be scanned/image-only or empty."
        )

    return "\n\n".join(pages_text)
