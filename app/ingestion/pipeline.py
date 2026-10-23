import io
from pathlib import Path

from pypdf import PdfReader


MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("File exceeds the 10 MB limit.")

    if suffix == ".txt":
        text = data.decode("utf-8-sig")

    elif suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    text = text.strip()

    if not text:
        raise ValueError(
            "No extractable text found. Scanned PDFs require OCR."
        )

    return text


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start

        while end < len(words):
            candidate = " ".join(words[start : end + 1])

            if len(candidate) > CHUNK_SIZE and end > start:
                break

            end += 1

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Advance by the chunk size minus the overlap.
        overlap_start = end

        while (
            overlap_start > start
            and len(" ".join(words[overlap_start:end]))
            < CHUNK_OVERLAP
        ):
            overlap_start -= 1

        start = max(start + 1, overlap_start)

    return chunks
