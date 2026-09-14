from pathlib import Path

import numpy as np

from backend.rag.vector_store import PostgresVectorStore, RAGSettings


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        rows = []
        for index, _ in enumerate(texts):
            vector = np.zeros(384, dtype=np.float32)
            vector[index % 384] = 1.0
            rows.append(vector)
        return np.asarray(rows)


class EmptyPolicyRepository:
    def list_policies(self):
        return []


def build_store(tmp_path: Path) -> PostgresVectorStore:
    return PostgresVectorStore(
        tmp_path,
        settings=RAGSettings(
            database_url="postgresql://unused",
            embedding_model="test-model",
            embedding_dimension=384,
            chunk_size_words=10,
            chunk_overlap_words=2,
        ),
        policy_repository=EmptyPolicyRepository(),
        embedding_model_instance=FakeEmbeddingModel(),
    )


def test_chunking_has_overlap_and_stable_ids(tmp_path):
    store = build_store(tmp_path)
    text = " ".join(f"word{i}" for i in range(25))
    chunks = store._split("POL-1", "policy.pdf", text)

    assert len(chunks) == 4
    assert chunks[0]["chunk_id"] == "policy-1"
    assert chunks[1]["chunk_id"] == "policy-2"
    assert chunks[0]["policy_id"] == "POL-1"
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]


def test_embedding_contract_is_384_dimensions(tmp_path):
    store = build_store(tmp_path)
    vectors = store._embed(["one", "two"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32


def test_default_rag_settings_use_pgvector_and_multilingual_model(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    settings = RAGSettings.from_env()
    assert settings.database_url.startswith("postgresql://")
    assert "multilingual" in settings.embedding_model.lower()
    assert settings.embedding_dimension == 384


def test_policy_reader_keeps_page_level_ocr_metadata(tmp_path):
    store = build_store(tmp_path)

    class FakeOCR:
        def read_document(self, name):
            assert name == "policy.pdf"
            return {
                "status": "read",
                "text": "[Página 1] texto digital\n[Página 2] texto recuperado por OCR",
                "extraction_method": "pdf_hybrid_text_ocr",
                "native_pages": [1],
                "ocr_pages": [2],
                "unreadable_pages": [],
            }

    store.ocr_service = FakeOCR()
    result = store._read_policy(
        {"policy_id": "POL-1", "file": "policy.pdf"},
        source_hash="abc123",
    )
    assert result["native_pages"] == [1]
    assert result["ocr_pages"] == [2]
    assert result["unreadable_pages"] == []
    assert result["extraction_method"] == "pdf_hybrid_text_ocr"
