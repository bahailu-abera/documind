import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Base, Document, DocumentChunk
from app.db.session import SessionLocal, engine
from app.ingestion.embeddings import embed_texts
from app.ingestion.pipeline import chunk_text, extract_text

from pydantic import BaseModel, Field
from app.generation.answer import generate_answer
from app.retrieval.search import retrieve_chunks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MVP setup. We'll replace create_all with Alembic migrations
    # before treating this as production-ready.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="DocuMind",
    description="A RAG assistant with evaluation.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "documind"}




class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: int | None = Field(default=None, gt=0)
    top_k: int = Field(default=5, ge=1, le=10)


@app.post("/questions")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        query_embedding = embed_texts(
            [question],
            task_type="RETRIEVAL_QUERY",
        )[0]

        with SessionLocal() as db:
            sources = retrieve_chunks(
                db=db,
                query_embedding=query_embedding,
                top_k=request.top_k,
                document_id=request.document_id,
            )

        if not sources:
            return {
                "question": question,
                "answer": (
                    "I couldn't find any indexed passages to answer "
                    "this question. Upload a document first."
                ),
                "sources": [],
            }

        answer = generate_answer(question, sources)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source_id": f"S{index}",
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "chunk_index": source["chunk_index"],
                    "similarity": source["similarity"],
                    "excerpt": source["content"][:500],
                }
                for index, source in enumerate(sources, start=1)
            ],
        }

    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=502,
            detail="Question answering failed. Check server logs.",
        ) from exc


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files are supported.",
        )

    try:
        data = file.file.read(10 * 1024 * 1024 + 1)
        extracted_text = extract_text(filename, data)
        chunks = chunk_text(extracted_text)

        if not chunks:
            raise ValueError("No usable text chunks were extracted.")

        vectors = embed_texts(chunks)

        with SessionLocal.begin() as db:
            document = Document(
                filename=filename,
                sha256=hashlib.sha256(data).hexdigest(),
            )
            db.add(document)
            db.flush()

            db.add_all(
                [
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=index,
                        content=content,
                        embedding=vector,
                    )
                    for index, (content, vector) in enumerate(
                        zip(chunks, vectors, strict=True)
                    )
                ]
            )

            document_id = document.id

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "status": "indexed",
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=502,
            detail="Document ingestion failed. Check server logs.",
        ) from exc

    finally:
        file.file.close()