from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_status"] == "trained"
    assert data["claims_available"] >= 1
    assert data["policies_available"] == 3


def test_dataset_info():
    data = client.get("/dataset/info").json()
    assert data["rows"] == 15420
    assert data["fraud_cases"] == 923
    assert len(data["model_features"]) == 30


def test_stored_claim_analysis_does_not_need_documents():
    claims = client.get("/claims").json()["claims"]
    claim_id = claims[0]["claim_id"]
    response = client.get(f"/claims/{claim_id}/analysis")
    assert response.status_code == 200
    data = response.json()
    assert data["fraud_score"] is not None
    assert data["risk_level"] in {"Bajo", "Medio", "Alto"}
    document_agent = data["agents_trace"][0]["output"]
    assert document_agent["documents_requested"] == []


def test_ground_truth_is_evaluation_only():
    claims = client.get("/claims").json()["claims"]
    claim_id = claims[0]["claim_id"]
    stored = client.get(f"/claims/{claim_id}").json()
    assert "FraudFound_P" not in stored["features"]
    assert "FraudFound_P" in stored["ground_truth"]
    evaluation = client.get(f"/claims/{claim_id}/comparison")
    assert evaluation.status_code == 200


def test_policy_query_is_strictly_filtered_to_selected_policy(monkeypatch):
    from backend.main import get_orchestrator

    store = get_orchestrator().rag_agent.vector_store

    def fake_search(query, policy_id, top_k=3):
        assert policy_id == "AXA-CAR"
        return [
            {
                "policy_id": "AXA-CAR",
                "source": "axa_car_policy.pdf",
                "chunk_id": "axa_car_policy-1",
                "chunk_index": 1,
                "text": "[Página 12]\nThe policy may cover theft subject to its terms.",
                "semantic_score": 0.82,
            }
        ]

    monkeypatch.setattr(store, "search", fake_search)
    response = client.post(
        "/policies/AXA-CAR/query",
        json={"query": "Is theft covered by the policy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["policy_id"] == "AXA-CAR"
    assert "pgvector" in data["retrieval_method"]
    for source in data["sources"]:
        assert source["source"] == "axa_car_policy.pdf"
        assert source["exact_text"].startswith("[Página 12]")
        assert source["pages"] == [12]
        assert source["page"] == 12
