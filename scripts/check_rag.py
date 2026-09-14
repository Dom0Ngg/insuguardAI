from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.vector_store import PostgresVectorStore, RAGUnavailableError  # noqa: E402


if __name__ == "__main__":
    store = PostgresVectorStore(ROOT / "data" / "policies")
    try:
        store.ensure_indexed()
        info = store.get_index_summary()
        results = store.search(
            "¿Está cubierto el robo del vehículo?",
            policy_id="AXA-CAR",
            top_k=3,
        )
    except RAGUnavailableError as exc:
        print(f"RAG ERROR: {exc}")
        raise SystemExit(1)

    print("RAG index:")
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print("\nTop semantic matches (AXA-CAR only):")
    for result in results:
        print(
            f"- {result['chunk_id']} score={result['semantic_score']} "
            f"source={result['source']}"
        )
        print(f"  {result['text'][:240]}...")
