from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from backend.ocr.ocr_service import OCRService
from backend.services.policy_repository import PolicyRepository


class RAGConfigurationError(RuntimeError):
    """Raised when the semantic RAG stack is not correctly configured."""


class RAGUnavailableError(RuntimeError):
    """Raised when PostgreSQL/pgvector or the embedding model is unavailable."""


@dataclass(frozen=True)
class RAGSettings:
    database_url: str
    embedding_model: str
    embedding_dimension: int
    chunk_size_words: int
    chunk_overlap_words: int

    @classmethod
    def from_env(cls) -> "RAGSettings":
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://insurguard:insurguard@localhost:5432/insurguard",
            ),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            embedding_dimension=int(os.getenv("RAG_EMBEDDING_DIMENSION", "384")),
            chunk_size_words=int(os.getenv("RAG_CHUNK_SIZE_WORDS", "220")),
            chunk_overlap_words=int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "40")),
        )


class PostgresVectorStore:
    """Semantic policy retrieval backed by PostgreSQL + pgvector.

    Policy PDFs are extracted with the existing OCR service, split into overlapping
    chunks, embedded with Sentence Transformers and stored in PostgreSQL. Queries
    are embedded with the same model and ranked with pgvector cosine distance.
    """

    def __init__(
        self,
        policies_dir: Path,
        settings: Optional[RAGSettings] = None,
        policy_repository: Optional[PolicyRepository] = None,
        embedding_model_instance: Any = None,
    ):
        self.policies_dir = policies_dir
        self.settings = settings or RAGSettings.from_env()
        self.policy_repository = policy_repository or PolicyRepository()
        self.ocr_service = OCRService(policies_dir)
        self._embedding_model = embedding_model_instance
        self._schema_ready = False

        if self.settings.chunk_overlap_words >= self.settings.chunk_size_words:
            raise RAGConfigurationError(
                "RAG_CHUNK_OVERLAP_WORDS must be smaller than RAG_CHUNK_SIZE_WORDS"
            )

    @staticmethod
    def dependencies_available() -> bool:
        try:
            import psycopg  # noqa: F401
            import pgvector  # noqa: F401
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _external_dependencies():
        try:
            import psycopg
            from pgvector import Vector
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RAGUnavailableError(
                "RAG dependencies are not installed. Run: "
                "pip install sentence-transformers 'psycopg[binary]' pgvector"
            ) from exc
        return psycopg, Vector, register_vector

    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RAGUnavailableError(
                "sentence-transformers is not installed. Run pip install -r requirements.txt"
            ) from exc

        try:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        except Exception as exc:  # model download/cache errors
            raise RAGUnavailableError(
                f"Could not load embedding model '{self.settings.embedding_model}': {exc}"
            ) from exc

        dimension = int(self._embedding_model.get_sentence_embedding_dimension())
        if dimension != self.settings.embedding_dimension:
            raise RAGConfigurationError(
                f"Embedding model dimension is {dimension}, but "
                f"RAG_EMBEDDING_DIMENSION={self.settings.embedding_dimension}."
            )
        return self._embedding_model

    def _connect(self, register: bool = True):
        psycopg, _, register_vector = self._external_dependencies()
        try:
            conn = psycopg.connect(self.settings.database_url)
        except Exception as exc:
            raise RAGUnavailableError(
                "Cannot connect to PostgreSQL. Start the pgvector database with "
                "'docker compose up -d postgres' or check DATABASE_URL. "
                f"Original error: {exc}"
            ) from exc
        if register:
            try:
                register_vector(conn)
            except Exception:
                conn.close()
                raise
        return conn

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return

        # pgvector must exist before registering the vector type on a connection.
        conn = self._connect(register=False)
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        finally:
            conn.close()

        conn = self._connect(register=True)
        try:
            dim = int(self.settings.embedding_dimension)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_policy_documents (
                        policy_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_dimension INTEGER NOT NULL,
                        extraction_method TEXT,
                        native_pages INTEGER[] NOT NULL DEFAULT '{}',
                        ocr_pages INTEGER[] NOT NULL DEFAULT '{}',
                        unreadable_pages INTEGER[] NOT NULL DEFAULT '{}',
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute("ALTER TABLE rag_policy_documents ADD COLUMN IF NOT EXISTS native_pages INTEGER[] NOT NULL DEFAULT '{}'")
                cur.execute("ALTER TABLE rag_policy_documents ADD COLUMN IF NOT EXISTS ocr_pages INTEGER[] NOT NULL DEFAULT '{}'")
                cur.execute("ALTER TABLE rag_policy_documents ADD COLUMN IF NOT EXISTS unreadable_pages INTEGER[] NOT NULL DEFAULT '{}'")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_policy_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        policy_id TEXT NOT NULL REFERENCES rag_policy_documents(policy_id)
                            ON DELETE CASCADE,
                        source TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector({dim}) NOT NULL,
                        UNIQUE(policy_id, chunk_id)
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_policy_chunks_policy_id "
                    "ON rag_policy_chunks(policy_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_policy_chunks_embedding_hnsw "
                    "ON rag_policy_chunks USING hnsw (embedding vector_cosine_ops)"
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RAGUnavailableError(
                "Could not initialize pgvector schema. If the database was created "
                "with a different embedding dimension, run 'docker compose down -v' "
                "and start it again. "
                f"Original error: {exc}"
            ) from exc
        finally:
            conn.close()

        self._schema_ready = True

    def _split(self, policy_id: str, source: str, text: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks: List[Dict[str, Any]] = []
        step = self.settings.chunk_size_words - self.settings.chunk_overlap_words
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + self.settings.chunk_size_words]).strip()
            if not piece:
                continue
            index = len(chunks) + 1
            chunks.append(
                {
                    "policy_id": policy_id,
                    "source": source,
                    "chunk_id": f"{Path(source).stem}-{index}",
                    "chunk_index": index,
                    "text": piece,
                }
            )
        return chunks

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.settings.embedding_dimension), dtype=np.float32)
        model = self._get_embedding_model()
        embeddings = model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.settings.embedding_dimension:
            raise RAGConfigurationError(
                f"Expected {self.settings.embedding_dimension} embedding dimensions, "
                f"received {array.shape[1]}."
            )
        return array

    def _read_policy(self, policy: Dict[str, Any], source_hash: str) -> Dict[str, Any]:
        source = policy["file"]
        result = self.ocr_service.read_document(source)
        text = (result.get("text") or "").strip()
        if result.get("status") != "read" or not text:
            raise RAGUnavailableError(
                f"Policy {source} could not be extracted: "
                f"{result.get('error') or result.get('warning') or result.get('status')}"
            )
        return {
            "policy_id": policy["policy_id"],
            "source": source,
            "text": text,
            "content_hash": source_hash,
            "extraction_method": result.get("extraction_method"),
            "native_pages": result.get("native_pages", []),
            "ocr_pages": result.get("ocr_pages", []),
            "unreadable_pages": result.get("unreadable_pages", []),
        }

    def sync_policies(self, force: bool = False, prune: bool = True) -> Dict[str, Any]:
        """Index new/changed policy PDFs and return an indexing summary."""
        self.ensure_schema()
        policies = self.policy_repository.list_policies()
        if not policies:
            return {"status": "ok", "indexed": [], "skipped": [], "removed": [], "total_policies": 0}

        current_policy_ids = {p["policy_id"] for p in policies}
        indexed: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        removed: List[str] = []
        _, Vector, _ = self._external_dependencies()

        conn = self._connect(register=True)
        try:
            with conn.cursor() as cur:
                if prune:
                    cur.execute("SELECT policy_id FROM rag_policy_documents")
                    stale = [row[0] for row in cur.fetchall() if row[0] not in current_policy_ids]
                    for policy_id in stale:
                        cur.execute("DELETE FROM rag_policy_documents WHERE policy_id = %s", (policy_id,))
                        removed.append(policy_id)

                for policy in policies:
                    source_path = self.policies_dir / policy["file"]
                    if not source_path.exists():
                        raise RAGUnavailableError(f"Policy file does not exist: {source_path}")
                    source_hash = self._file_hash(source_path)
                    cur.execute(
                        """
                        SELECT content_hash, embedding_model, embedding_dimension
                        FROM rag_policy_documents
                        WHERE policy_id = %s
                        """,
                        (policy["policy_id"],),
                    )
                    existing = cur.fetchone()
                    unchanged = (
                        existing
                        and existing[0] == source_hash
                        and existing[1] == self.settings.embedding_model
                        and int(existing[2]) == self.settings.embedding_dimension
                    )
                    if unchanged and not force:
                        skipped.append({
                            "policy_id": policy["policy_id"],
                            "source": policy["file"],
                            "reason": "unchanged",
                        })
                        continue

                    doc = self._read_policy(policy, source_hash)
                    chunks = self._split(doc["policy_id"], doc["source"], doc["text"])
                    embeddings = self._embed([chunk["text"] for chunk in chunks])

                    cur.execute(
                        """
                        INSERT INTO rag_policy_documents (
                            policy_id, source, content_hash, embedding_model,
                            embedding_dimension, extraction_method, native_pages, ocr_pages,
                            unreadable_pages, chunk_count, indexed_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (policy_id) DO UPDATE SET
                            source = EXCLUDED.source,
                            content_hash = EXCLUDED.content_hash,
                            embedding_model = EXCLUDED.embedding_model,
                            embedding_dimension = EXCLUDED.embedding_dimension,
                            extraction_method = EXCLUDED.extraction_method,
                            native_pages = EXCLUDED.native_pages,
                            ocr_pages = EXCLUDED.ocr_pages,
                            unreadable_pages = EXCLUDED.unreadable_pages,
                            chunk_count = EXCLUDED.chunk_count,
                            indexed_at = NOW()
                        """,
                        (
                            doc["policy_id"],
                            doc["source"],
                            doc["content_hash"],
                            self.settings.embedding_model,
                            self.settings.embedding_dimension,
                            doc["extraction_method"],
                            doc["native_pages"],
                            doc["ocr_pages"],
                            doc["unreadable_pages"],
                            len(chunks),
                        ),
                    )
                    cur.execute("DELETE FROM rag_policy_chunks WHERE policy_id = %s", (doc["policy_id"],))

                    rows = [
                        (
                            chunk["policy_id"],
                            chunk["source"],
                            chunk["chunk_id"],
                            chunk["chunk_index"],
                            chunk["text"],
                            Vector(embedding),
                        )
                        for chunk, embedding in zip(chunks, embeddings)
                    ]
                    if rows:
                        cur.executemany(
                            """
                            INSERT INTO rag_policy_chunks (
                                policy_id, source, chunk_id, chunk_index, content, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            rows,
                        )
                    indexed.append(
                        {
                            "policy_id": doc["policy_id"],
                            "source": doc["source"],
                            "chunks": len(chunks),
                            "extraction_method": doc["extraction_method"],
                            "native_pages": doc["native_pages"],
                            "ocr_pages": doc["ocr_pages"],
                            "unreadable_pages": doc["unreadable_pages"],
                        }
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return {
            "status": "ok",
            "indexed": indexed,
            "skipped": skipped,
            "removed": removed,
            "total_policies": len(policies),
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": self.settings.embedding_dimension,
        }

    def ensure_indexed(self) -> Dict[str, Any]:
        """Keep the DB synchronized; unchanged PDFs do not get embedded again."""
        return self.sync_policies(force=False, prune=True)

    def search(self, query: str, policy_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        self.ensure_indexed()
        _, Vector, _ = self._external_dependencies()
        query_embedding = self._embed([query])[0]
        vector = Vector(query_embedding)

        conn = self._connect(register=True)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        policy_id,
                        source,
                        chunk_id,
                        chunk_index,
                        content,
                        1 - (embedding <=> %s) AS cosine_similarity
                    FROM rag_policy_chunks
                    WHERE policy_id = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (vector, policy_id, vector, int(top_k)),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            {
                "policy_id": row[0],
                "source": row[1],
                "chunk_id": row[2],
                "chunk_index": int(row[3]),
                "text": row[4],
                "semantic_score": round(float(row[5]), 4),
                "vector_score": round(float(row[5]), 4),
                "hybrid_score": round(float(row[5]), 4),
            }
            for row in rows
        ]

    def get_index_summary(self) -> Dict[str, Any]:
        self.ensure_schema()
        conn = self._connect(register=True)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rag_policy_documents")
                total_policies = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM rag_policy_chunks")
                total_chunks = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT policy_id, source, chunk_count, extraction_method, native_pages, ocr_pages, unreadable_pages, indexed_at
                    FROM rag_policy_documents
                    ORDER BY policy_id
                    """
                )
                documents = [
                    {
                        "policy_id": row[0],
                        "source": row[1],
                        "chunks": int(row[2]),
                        "extraction_method": row[3],
                        "native_pages": list(row[4] or []),
                        "ocr_pages": list(row[5] or []),
                        "unreadable_pages": list(row[6] or []),
                        "indexed_at": row[7].isoformat() if row[7] else None,
                    }
                    for row in cur.fetchall()
                ]
        finally:
            conn.close()

        return {
            "status": "ready",
            "backend": "PostgreSQL + pgvector",
            "retrieval_method": "Sentence Transformers embeddings + pgvector cosine similarity (HNSW)",
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": self.settings.embedding_dimension,
            "total_policies": total_policies,
            "total_chunks": total_chunks,
            "sources": sorted({document["source"] for document in documents}),
            "documents": documents,
        }
