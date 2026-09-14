from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.vector_store import PostgresVectorStore, RAGUnavailableError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index car policy PDFs into PostgreSQL + pgvector")
    parser.add_argument("--force", action="store_true", help="Re-embed all policies even if unchanged")
    args = parser.parse_args()

    store = PostgresVectorStore(ROOT / "data" / "policies")
    try:
        result = store.sync_policies(force=args.force, prune=True)
        summary = store.get_index_summary()
    except RAGUnavailableError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(json.dumps({"sync": result, "index": summary}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
