
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk


def retrieve_chunks(
    db: Session,
    query_embedding: list[float],
    top_k: int = 5,
    document_id: int | None = None,
) -> list[dict]:
    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(DocumentChunk, Document.filename, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(distance)
        .limit(top_k)
    )

    if document_id is not None:
        statement = statement.where(
            DocumentChunk.document_id == document_id
        )

    results = db.execute(statement).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "filename": filename,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "similarity": round(1.0 - float(distance_value), 4),
        }
        for chunk, filename, distance_value in results
    ]
