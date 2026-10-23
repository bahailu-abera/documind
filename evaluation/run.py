
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing the database session.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from app.db.session import SessionLocal  # noqa: E402
from app.ingestion.embeddings import embed_texts  # noqa: E402
from app.retrieval.search import retrieve_chunks  # noqa: E402
from evaluation.metrics import recall_at_k, reciprocal_rank  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "report.json"


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    results = []

    for item in dataset:
        relevant_indices = set(item["relevant_chunk_indices"])
        if not relevant_indices:
            raise ValueError(
                f"{item['id']} must have at least one relevant chunk."
            )

        top_k = int(item.get("top_k", 5))
        if top_k < 1 or top_k > 10:
            raise ValueError(f"{item['id']}: top_k must be between 1 and 10.")

        query_vector = embed_texts(
            [item["question"]],
            task_type="RETRIEVAL_QUERY",
        )[0]

        with SessionLocal() as db:
            sources = retrieve_chunks(
                db=db,
                query_embedding=query_vector,
                top_k=top_k,
                document_id=int(item["document_id"]),
            )

        retrieved_indices = [source["chunk_index"] for source in sources]

        result = {
            "id": item["id"],
            "question": item["question"],
            "document_id": item["document_id"],
            "top_k": top_k,
            "relevant_chunk_indices": sorted(relevant_indices),
            "retrieved_chunk_indices": retrieved_indices,
            f"recall_at_{top_k}": round(
                recall_at_k(retrieved_indices, relevant_indices, top_k), 4
            ),
            "reciprocal_rank": round(
                reciprocal_rank(retrieved_indices, relevant_indices), 4
            ),
        }
        results.append(result)

        print(
            f"{item['id']}: "
            f"Recall@{top_k}={result[f'recall_at_{top_k}']:.2f}, "
            f"RR={result['reciprocal_rank']:.2f}"
        )

    recall_keys = {f"recall_at_{item.get('top_k', 5)}" for item in dataset}
    summary = {
        "questions_evaluated": len(results),
        "mean_reciprocal_rank": round(
            sum(r["reciprocal_rank"] for r in results) / len(results), 4
        ),
        "mean_recall": {
            key: round(
                sum(r[key] for r in results if key in r)
                / sum(1 for r in results if key in r),
                4,
            )
            for key in sorted(recall_keys)
        },
        "results": results,
    }

    REPORT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nReport saved to {REPORT_PATH}")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
