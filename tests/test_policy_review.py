from fastapi.testclient import TestClient

from backend.agents.rag_coverage_agent import RAGCoverageAgent
from backend.main import app
import backend.admin.router as admin_router


client = TestClient(app)


def test_rag_policy_review_returns_structured_coverage_evidence():
    chunks = [
        {
            "source": "policy.pdf",
            "chunk_id": "policy-10",
            "chunk_index": 10,
            "semantic_score": 0.62,
            "text": "[Página 18] We will cover accidental damage to your car, subject to the conditions of this section.",
        },
        {
            "source": "policy.pdf",
            "chunk_id": "policy-11",
            "chunk_index": 11,
            "semantic_score": 0.44,
            "text": "[Página 19] You must notify us as soon as reasonably possible after the incident.",
        },
    ]

    review = RAGCoverageAgent._review_result(chunks)

    assert review["coverage_result"] == "Probablemente cubierto"
    assert review["retrieval_quality"] == "alta"
    assert review["coverage_evidence"]
    assert review["coverage_evidence"][0]["pages"] == [18]
    assert review["condition_evidence"]


def test_rag_policy_review_refuses_to_overinterpret_low_similarity():
    chunks = [
        {
            "source": "policy.pdf",
            "chunk_id": "policy-1",
            "chunk_index": 1,
            "semantic_score": 0.18,
            "text": "[Página 4] We will not cover some unrelated circumstance.",
        }
    ]

    review = RAGCoverageAgent._review_result(chunks)

    assert review["coverage_result"] == "Evidencia insuficiente"
    assert review["retrieval_quality"] == "insuficiente"


def test_supervisor_policy_review_is_persisted_without_full_reanalysis(tmp_path, monkeypatch):
    claims_dir = tmp_path / "claims"
    monkeypatch.setattr(admin_router.claim_repository, "claims_dir", claims_dir)
    admin_router.claim_repository.save_claim(
        {
            "claim_id": "TG-POLICY-TEST",
            "source": {"type": "telegram"},
            "features": {},
            "documents": [],
        }
    )

    calls = []

    def fake_rag_run(payload):
        calls.append(payload)
        return {
            "status": "success",
            "output": {
                "coverage_result": "Revisión manual requerida",
                "policy_id": payload["policy_id"],
                "query": payload["coverage_query"],
                "retrieval_quality": "media",
                "top_semantic_score": 0.34,
                "coverage_evidence": [],
                "exclusion_evidence": [],
                "condition_evidence": [],
                "recommended_action": "Revisar cláusulas.",
                "sources": [],
            },
        }

    monkeypatch.setattr(admin_router.analysis_service.orchestrator.rag_agent, "run", fake_rag_run)

    response = client.post(
        "/admin/api/claims/TG-POLICY-TEST/policy-review",
        json={"policy_id": "AXA-CAR", "query": "¿Existe alguna exclusión aplicable?"},
    )

    assert response.status_code == 200
    assert calls and calls[0]["top_k"] == 5
    stored = admin_router.claim_repository.get_claim("TG-POLICY-TEST")
    assert stored["policy_id"] == "AXA-CAR"
    assert stored["coverage_query"] == "¿Existe alguna exclusión aplicable?"
    assert stored["policy_review"]["output"]["retrieval_quality"] == "media"
