from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.claim_repository import ClaimRepository
from backend.telegram.intake_agent import TelegramIntakeAgent
from backend.telegram.repository import TelegramRepository
from backend.telegram.service import TelegramService


client = TestClient(app)


class FakeAnalysisService:
    def analyze(self, claim_data, *, source_channel, record_production):
        assert source_channel == "telegram"
        assert record_production is True
        return {
            "claim_id": claim_data["claim_id"],
            "fraud_score": 0.71,
            "risk_level": "Alto",
            "manual_review_required": True,
            "coverage_result": "No evaluado",
            "explanation": "demo",
            "agents_trace": [],
        }


class FakeTelegramClient:
    enabled = True

    def __init__(self):
        self.sent = []
        self.answered_callbacks = []

    def send_text(self, chat_id, text, *, reply_markup=None):
        self.sent.append((str(chat_id), text, reply_markup))
        return {"message_id": len(self.sent), "chat": {"id": int(chat_id)}}

    def answer_callback_query(self, callback_query_id, *, text=None, show_alert=False):
        self.answered_callbacks.append(str(callback_query_id))
        return True


def test_admin_dashboard_is_served():
    response = client.get("/admin")
    assert response.status_code == 200
    assert "InsurGuard AI" in response.text
    assert "Flujo multiagente" in response.text
    assert "Telegram" in response.text
    assert "Ground truth" in response.text


def test_admin_dashboard_has_supervisor_status_filter_and_clean_evidence_label():
    response = client.get("/admin")
    assert response.status_code == 200
    assert 'id="claims-status-filter"' in response.text
    assert "Pendiente de análisis" in response.text
    assert "Ground truth pendiente" in response.text
    assert "Ground truth confirmado" in response.text
    js = client.get("/admin/static/admin.js")
    assert js.status_code == 200
    assert "Revisión documental" in js.text


def test_admin_claims_exposes_supervisor_ground_truth_fields():
    response = client.get("/admin/api/claims")
    assert response.status_code == 200
    payload = response.json()
    assert "decision_threshold" in payload
    assert payload["claims"]
    row = payload["claims"][0]
    assert "supervisor_status" in row
    assert "monitoring_recorded" in row
    assert "production_ground_truth" in row
    assert "dataset_ground_truth" in row


def test_telegram_info_exposes_long_polling_mode_without_webhook():
    response = client.get("/telegram/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["receive_mode"] == "getUpdates long polling"
    assert payload["requires_public_webhook"] is False
    assert "text" in payload["supported_inbound"]
    assert "document" in payload["supported_inbound"]
    assert "photo" in payload["supported_inbound"]


def test_telegram_intake_builds_exact_ml_contract(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    claims = ClaimRepository()
    claims.claims_dir = tmp_path / "claims"
    agent = TelegramIntakeAgent(
        repository=repository,
        claim_repository=claims,
        analysis_service=FakeAnalysisService(),
    )

    chat_id = "123456789"
    replies = agent.process_text(chat_id, "NUEVO")
    assert "Nuevo siniestro" in replies[0]

    answers = [
        "12/09/2026",
        "13/09/2026",
        "Ford",
        "urbana",
        "hombre",
        "soltero",
        "28",
        "asegurado",
        "sedan",
        "colisión",
        "32000",
        "400",
        "3",
        "120",
        "121",
        "1",
        "5",
        "30",
        "sí",
        "no",
        "externo",
        "1",
        "0",
        "1",
    ]
    for answer in answers:
        agent.process_text(chat_id, answer)

    session = repository.get_session(chat_id)
    assert session is not None
    assert session["current_step"] == len(agent.steps)
    claim = claims.get_claim(session["claim_id"])
    assert claim["claim_id"].startswith("TG-")
    assert claim["source"]["type"] == "telegram"
    assert len(claim["features"]) == 30
    assert claim["features"]["PolicyType"] == "Sedan - Collision"
    assert claim["features"]["Month"] == "Sep"
    assert claim["features"]["MonthClaimed"] == "Sep"
    assert "FraudFound_P" not in claim["features"]

    result_reply = agent.process_text(chat_id, "ANALIZAR")
    assert "ANÁLISIS COMPLETADO" in result_reply[0]
    assert repository.get_session(chat_id)["status"] == "completed"


def test_telegram_update_is_processed_and_audited(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    fake_client = FakeTelegramClient()
    claims = ClaimRepository()
    claims.claims_dir = tmp_path / "claims"
    intake = TelegramIntakeAgent(
        repository=repository,
        claim_repository=claims,
        analysis_service=FakeAnalysisService(),
    )
    service = TelegramService(repository=repository, client=fake_client, intake_agent=intake)

    result = service.process_update(
        {
            "update_id": 9001,
            "message": {
                "message_id": 44,
                "from": {"id": 123456789, "username": "demo_user", "first_name": "Demo"},
                "chat": {"id": 123456789, "type": "private"},
                "text": "NUEVO",
            },
        }
    )

    assert result["status"] == "ok"
    assert fake_client.sent
    session = repository.get_session("123456789")
    assert session is not None
    assert session["username"] == "demo_user"
    assert session["claim_id"].startswith("TG-")
    messages = repository.list_messages("123456789")
    assert any(m["direction"] == "inbound" and m["text"] == "NUEVO" for m in messages)
    assert any(m["direction"] == "outbound" for m in messages)


def test_telegram_make_accepts_dataset_brands():
    """Regression: Toyota/BMW must not be rejected by the conversational intake."""
    from backend.telegram.intake_agent import TelegramIntakeAgent

    agent = TelegramIntakeAgent()
    parser = next(step.parser for step in agent.steps if step.key == "Make")
    assert parser("Toyota", {}) == {"Make": "Toyota"}
    assert parser("BMW", {}) == {"Make": "BMW"}
    assert parser("volkswagen", {}) == {"Make": "VW"}


def test_telegram_catalog_supports_pandas_string_dtype(monkeypatch):
    """Pandas 3 may infer string dtype instead of object; catalog must still load."""
    import pandas as pd
    from backend.telegram.intake_agent import TelegramIntakeAgent

    original = pd.read_csv

    def fake_read_csv(*args, **kwargs):
        df = original(*args, **kwargs)
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = df[column].astype("string")
        return df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    agent = TelegramIntakeAgent()
    assert "Toyota" in agent._catalog["Make"]
    assert "BMW" in agent._catalog["Make"]


def test_telegram_questions_include_explanations_and_context_buttons(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    agent = TelegramIntakeAgent(repository=repository, analysis_service=FakeAnalysisService())
    chat_id = "12345"

    replies = agent.process_text(chat_id, "NUEVO")
    assert "ℹ️" in replies[-1]
    assert "Fecha del accidente" in replies[-1]

    # Advance to the Make question, which should expose common-brand buttons.
    agent.process_text(chat_id, "12/09/2026")
    replies = agent.process_text(chat_id, "13/09/2026")
    assert "Marca del vehículo" in replies[-1]
    markup = agent.reply_markup_for(chat_id)
    labels = [button["text"] for row in markup["inline_keyboard"] for button in row]
    assert "Toyota" in labels
    assert "BMW" in labels
    assert "📊 Estado" in labels
    assert "❌ Cancelar" in labels


def test_telegram_button_labels_map_to_commands(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    agent = TelegramIntakeAgent(repository=repository, analysis_service=FakeAnalysisService())
    chat_id = "22222"

    replies = agent.process_text(chat_id, "🆕 Nuevo siniestro")
    assert "Nuevo siniestro" in replies[0]
    status = agent.process_text(chat_id, "📊 Estado")
    assert "Estado del claim" in status[0]


def test_telegram_service_sends_context_keyboard(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    fake_client = FakeTelegramClient()
    intake = TelegramIntakeAgent(repository=repository, analysis_service=FakeAnalysisService())
    service = TelegramService(repository=repository, client=fake_client, intake_agent=intake)

    service.process_update(
        {
            "update_id": 99,
            "message": {
                "message_id": 1,
                "from": {"id": 9876, "username": "buttons"},
                "chat": {"id": 9876, "type": "private"},
                "text": "NUEVO",
            },
        }
    )
    assert fake_client.sent
    assert fake_client.sent[-1][2] is not None
    assert "inline_keyboard" in fake_client.sent[-1][2]


def test_telegram_inline_callback_advances_question(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    fake_client = FakeTelegramClient()
    intake = TelegramIntakeAgent(repository=repository, analysis_service=FakeAnalysisService())
    service = TelegramService(repository=repository, client=fake_client, intake_agent=intake)
    chat_id = "7777"

    intake.process_text(chat_id, "NUEVO")
    intake.process_text(chat_id, "12/09/2026")
    intake.process_text(chat_id, "13/09/2026")
    assert repository.get_session(chat_id)["current_step"] == 2

    result = service.process_update({
        "update_id": 123,
        "callback_query": {
            "id": "cb-1",
            "from": {"id": 7777, "username": "inline"},
            "data": "answer:Toyota",
            "message": {"message_id": 55, "chat": {"id": 7777, "type": "private"}},
        },
    })
    assert result["status"] == "ok"
    assert fake_client.answered_callbacks == ["cb-1"]
    session = repository.get_session(chat_id)
    assert session["current_step"] == 3
    assert session["data"]["features"]["Make"] == "Toyota"
    assert fake_client.sent[-1][2] is not None
    assert "inline_keyboard" in fake_client.sent[-1][2]


def test_supervisor_can_open_attached_document_but_claim_ocr_is_disabled(tmp_path: Path, monkeypatch):
    import json
    import backend.admin.router as admin_router_module

    claims_dir = tmp_path / "claims"
    claim_dir = claims_dir / "TG-DOC-001"
    docs_dir = claim_dir / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "evidence.txt").write_text("EVIDENCIA EXACTA DEL CLIENTE", encoding="utf-8")
    (claim_dir / "claim.json").write_text(
        json.dumps(
            {
                "claim_id": "TG-DOC-001",
                "source": {"type": "telegram"},
                "features": {},
                "documents": ["evidence.txt"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(admin_router_module.claim_repository, "claims_dir", claims_dir)

    opened = client.get("/admin/api/claims/TG-DOC-001/documents/evidence.txt")
    assert opened.status_code == 200
    assert opened.text == "EVIDENCIA EXACTA DEL CLIENTE"
    assert opened.headers["content-disposition"].startswith("inline;")

    # There is intentionally no claim-document OCR/extraction endpoint.
    extracted = client.get("/admin/api/claims/TG-DOC-001/documents/evidence.txt/extraction")
    assert extracted.status_code == 404

    missing = client.get("/admin/api/claims/TG-DOC-001/documents/not-attached.txt")
    assert missing.status_code == 404


def test_claim_document_agent_only_inventories_files(tmp_path: Path):
    from backend.agents.document_extraction_agent import DocumentExtractionAgent

    agent = DocumentExtractionAgent()
    agent.project_root = tmp_path
    docs = tmp_path / "data" / "claims" / "TG-1" / "documents"
    docs.mkdir(parents=True)
    (docs / "scan.pdf").write_bytes(b"not-a-real-pdf")

    result = agent.run({"claim_id": "TG-1", "documents": ["scan.pdf"]})
    out = result["output"]
    assert out["ocr_used"] is False
    assert out["content_processing"] == "disabled"
    assert out["documents_available"] == ["scan.pdf"]
    assert "raw_text" not in out
    assert "document_texts" not in out


def test_telegram_client_has_no_policy_or_coverage_rag_controls(tmp_path: Path):
    repository = TelegramRepository(tmp_path / "telegram.db")
    agent = TelegramIntakeAgent(repository=repository, analysis_service=FakeAnalysisService())
    help_text = agent.help_text().casefold()
    assert "rag" not in help_text
    assert "póliza" not in help_text
    assert "poliza" not in help_text
    assert "cobertura" not in help_text

    chat_id = "44444"
    agent.process_text(chat_id, "NUEVO")
    session = repository.get_session(chat_id)
    session["current_step"] = len(agent.steps)
    repository.save_session(session)
    labels = [b["text"] for row in agent.reply_markup_for(chat_id)["inline_keyboard"] for b in row]
    assert not any("AXA" in label or "Póliza" in label or "Cobertura" in label for label in labels)
    assert "🔎 Analizar" in labels


def test_supervisor_policy_question_is_separate_from_fraud(tmp_path: Path, monkeypatch):
    import json
    import backend.admin.router as admin_router_module

    claims_dir = tmp_path / "claims"
    claim_dir = claims_dir / "TG-RAG-001"
    claim_dir.mkdir(parents=True)
    (claim_dir / "claim.json").write_text(json.dumps({
        "claim_id": "TG-RAG-001",
        "source": {"type": "telegram"},
        "features": {},
        "documents": [],
    }), encoding="utf-8")
    monkeypatch.setattr(admin_router_module.claim_repository, "claims_dir", claims_dir)
    monkeypatch.setattr(admin_router_module, "_run_policy_question", lambda policy_id, query: {
        "policy_id": policy_id,
        "query": query,
        "coverage_result": "Revisión manual requerida",
        "sources": [{"chunk_id": "c1", "exact_text": "FRAGMENTO EXACTO", "semantic_score": 0.81, "pages": [7]}],
    })

    response = client.post("/admin/api/claims/TG-RAG-001/policy-review", json={
        "policy_id": "AXA-CAR",
        "query": "¿Qué exclusiones aplican?",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["output"]["sources"][0]["exact_text"] == "FRAGMENTO EXACTO"
    saved = json.loads((claim_dir / "claim.json").read_text(encoding="utf-8"))
    assert saved["policy_review"]["output"]["policy_id"] == "AXA-CAR"
    assert len(saved["policy_review_history"]) == 1
    assert "coverage_query" not in saved


def test_supervisor_can_open_policy_pdf_inline():
    response = client.get("/admin/api/policies/AXA-CAR/document")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline;")
