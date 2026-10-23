import os

from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    embeddings = []

    # Keep batches small to reduce request size.
    for start in range(0, len(texts), 16):
        batch = texts[start : start + 16]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")

        if len(response.embeddings) != len(batch):
            raise RuntimeError("Embedding response count mismatch.")

        embeddings.extend(
            embedding.values
            for embedding in response.embeddings
        )

    return embeddings