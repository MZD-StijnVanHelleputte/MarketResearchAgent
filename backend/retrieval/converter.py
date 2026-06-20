"""Converts uploaded industry-knowledge files into Markdown for the chunker.

Plain text/Markdown is passed through; binary office formats are converted
via MarkItDown so the existing header-aware chunker (chunker.chunk_document)
keeps working unchanged.
"""
import os
import tempfile

from markitdown import MarkItDown

_PASSTHROUGH_EXTS = {".md", ".txt"}
_CONVERTIBLE_EXTS = {".pdf", ".docx", ".pptx"}


def convert_to_markdown(filename: str, raw_bytes: bytes) -> str:
    """Returns Markdown text for a supported file, or raises ValueError."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in _PASSTHROUGH_EXTS:
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"'{filename}' is not valid UTF-8 text.") from exc

    if ext not in _CONVERTIBLE_EXTS:
        supported = sorted(_PASSTHROUGH_EXTS | _CONVERTIBLE_EXTS)
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {', '.join(supported)}"
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        result = MarkItDown().convert(tmp_path)
    except Exception as exc:
        raise ValueError(f"Failed to convert '{filename}': {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return result.text_content
