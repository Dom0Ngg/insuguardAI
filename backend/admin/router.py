from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import mimetypes

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.app.schemas import AdminAnalysisRequest, GroundTruthRequest, PolicyReviewRequest, TelegramSimulationRequest
from backend.ml.fraud_model import FraudDetectionModel
from backend.mlops.mlops_service import MLOpsService
from backend.services.analysis_repository import AnalysisRepository
from backend.services.analysis_service import AnalysisService
from backend.services.claim_repository import ClaimRepository
from backend.services.policy_repository import PolicyRepository
from backend.telegram.repository import TelegramRepository
from backend.telegram.service import TelegramService
from backend.telegram.settings import TelegramSettings


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
router = APIRouter(tags=["Admin dashboard"])

claim_repository = ClaimRepository()
policy_repository = PolicyRepository()
analysis_repository = AnalysisRepository()
mlops_service = MLOpsService()
telegram_repository = TelegramRepository()
telegram_service = TelegramService(repository=telegram_repository)
analysis_service = telegram_service.intake.analysis_service


def _safe_child(base: Path, name: str) -> Path:
    """Resolve a single file inside base without allowing path traversal."""
    safe_name = Path(name).name
    candidate = (base / safe_name).resolve()
    resolved_base = base.resolve()
    if candidate.parent != resolved_base:
        raise HTTPException(status_code=400, detail="Invalid file name")
    return candidate


def _inline_file_response(path: Path) -> FileResponse:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path=str(path),
        media_type=media_type or "application/octet-stream",
        filename=path.name,
        content_disposition_type="inline",
    )


def _safe_rag_info() -> Dict[str, Any]:
    try:
        return analysis_service.orchestrator.rag_agent.vector_store.get_index_summary()
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def _claim_source(claim: Dict[str, Any]) -> str:
    source = claim.get("source") or {}
    if source.get("type"):
        return str(source["type"])
    if source.get("dataset"):
        return "kaggle_demo"
    return "unknown"


def _monitoring_by_claim() -> Dict[str, Dict[str, Any]]:
    return {row["claim_id"]: row for row in mlops_service.monitoring.latest_predictions()}


def _supervisor_status(source: str, monitoring: Optional[Dict[str, Any]]) -> str:
    if source == "kaggle_demo":
        return "reference_data"
    if monitoring is None:
        return "pending_analysis"
    if monitoring.get("FraudFound_P") is None:
        return "pending_ground_truth"
    return "ground_truth_confirmed"


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"title": "InsurGuard AI · Administration"},
    )


@router.get("/admin/api/overview")
def admin_overview():
    monitoring = mlops_service.monitoring_summary()
    return {
        "claims": len(claim_repository.list_claims()),
        "policies": len(policy_repository.list_policies()),
        "analyses": analysis_repository.summary(),
        "telegram": {
            **telegram_repository.summary(),
            "configuration": TelegramSettings.from_env().public_status(),
        },
        "mlops": monitoring,
        "rag": _safe_rag_info(),
        "model": {
            "name": FraudDetectionModel().get_model_info().get("model_name"),
            "version": FraudDetectionModel.MODEL_VERSION,
            "decision_threshold": FraudDetectionModel().decision_threshold,
        },
    }


@router.get("/admin/api/claims")
def admin_claims():
    rows = []
    monitoring_map = _monitoring_by_claim()
    threshold = FraudDetectionModel().decision_threshold
    for item in claim_repository.list_claims():
        try:
            claim = claim_repository.get_claim(item["claim_id"])
        except FileNotFoundError:
            continue
        latest = analysis_repository.latest_for_claim(item["claim_id"])
        result = (latest or {}).get("result") or {}
        source = _claim_source(claim)
        monitoring = monitoring_map.get(item["claim_id"])
        score = result.get("fraud_score")
        rows.append(
            {
                **item,
                "source": source,
                "fraud_score": score,
                "risk_level": result.get("risk_level"),
                "manual_review_required": result.get("manual_review_required"),
                "analysis_status": (latest or {}).get("status"),
                "last_analysis_at": (latest or {}).get("completed_at"),
                "model_prediction": None if score is None else int(float(score) >= threshold),
                "monitoring_recorded": monitoring is not None,
                "production_ground_truth": None if monitoring is None else monitoring.get("FraudFound_P"),
                "ground_truth_updated_at": None if monitoring is None else monitoring.get("ground_truth_updated_at"),
                "dataset_ground_truth": claim.get("ground_truth", {}).get("FraudFound_P"),
                "supervisor_status": _supervisor_status(source, monitoring),
            }
        )
    return {"total": len(rows), "claims": rows, "decision_threshold": threshold}


@router.get("/admin/api/claims/{claim_id}")
def admin_claim_detail(claim_id: str):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    session = telegram_repository.get_session_by_claim(claim_id)
    messages = telegram_repository.list_messages(session["chat_id"], limit=300) if session else []
    source = _claim_source(claim)
    monitoring = _monitoring_by_claim().get(claim_id)
    return {
        "claim": claim,
        "source": source,
        "runs": analysis_repository.list_runs(limit=20, claim_id=claim_id),
        "telegram_session": session,
        "telegram_messages": messages,
        "monitoring": monitoring,
        "dataset_ground_truth": claim.get("ground_truth", {}).get("FraudFound_P"),
        "supervisor_status": _supervisor_status(source, monitoring),
        "decision_threshold": FraudDetectionModel().decision_threshold,
        "available_policies": policy_repository.list_policies(),
        "policy_review": claim.get("policy_review"),
        "policy_review_history": claim.get("policy_review_history", []),
    }


@router.post("/admin/api/claims/{claim_id}/analyze")
def admin_analyze_claim(claim_id: str, options: Optional[AdminAnalysisRequest] = None):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    # Fraud analysis is independent from policy review. Policy questions are
    # executed through /policy-review and do not create duplicate MLOps events.
    payload = {
        "claim_id": claim["claim_id"],
        "features": claim["features"],
        "documents": claim.get("documents", []),
    }
    source = _claim_source(claim)
    return analysis_service.analyze(
        payload,
        source_channel="admin",
        record_production=source == "telegram",
    )


@router.put("/admin/api/claims/{claim_id}/ground-truth")
def admin_set_ground_truth(claim_id: str, payload: GroundTruthRequest):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    source = _claim_source(claim)
    if source == "kaggle_demo":
        raise HTTPException(
            status_code=409,
            detail="Kaggle demo claims already contain reference labels and are not production ground truth.",
        )
    try:
        saved = mlops_service.set_ground_truth(claim_id, payload.FraudFound_P)
    except KeyError:
        raise HTTPException(
            status_code=409,
            detail="Analyze the claim first so a monitored production prediction exists before ground truth is recorded.",
        )
    return {
        "ground_truth": saved,
        "performance": mlops_service.production_performance(),
        "monitoring": mlops_service.monitoring_summary(),
    }


@router.get("/admin/api/claims/{claim_id}/documents/{document_name}", include_in_schema=False)
def admin_open_claim_document(claim_id: str, document_name: str):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")

    safe_name = Path(document_name).name
    if safe_name not in (claim.get("documents") or []):
        raise HTTPException(status_code=404, detail="Document is not attached to this claim")
    path = _safe_child(claim_repository.documents_dir(claim_id), safe_name)
    return _inline_file_response(path)


@router.get("/admin/api/policies")
def admin_policies():
    return {"policies": policy_repository.list_policies()}


def _run_policy_question(policy_id: str, query: str) -> Dict[str, Any]:
    policy = policy_repository.get_policy(policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    result = analysis_service.orchestrator.rag_agent.run(
        {"policy_id": policy_id, "coverage_query": query, "top_k": 5}
    )
    if result.get("status") == "error":
        raise HTTPException(
            status_code=503,
            detail=result.get("output", {}).get("rag_answer", "RAG unavailable"),
        )
    return result.get("output") or {}


@router.post("/admin/api/policies/query")
def admin_policy_query(payload: PolicyReviewRequest):
    """Supervisor-only free question over one selected policy."""
    from datetime import datetime, timezone
    return {
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": "supervisor",
        "output": _run_policy_question(payload.policy_id, payload.query),
    }


@router.post("/admin/api/claims/{claim_id}/policy-review")
def admin_policy_review(claim_id: str, payload: PolicyReviewRequest):
    """Ask a policy question in the context of a claim without re-running fraud."""
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")

    from datetime import datetime, timezone
    review = {
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": "supervisor",
        "output": _run_policy_question(payload.policy_id, payload.query),
    }
    history = list(claim.get("policy_review_history") or [])
    history.append(review)
    claim["policy_review"] = review
    claim["policy_review_history"] = history[-20:]
    claim_repository.save_claim(claim)
    return review


@router.get("/admin/api/policies/{policy_id}/document", include_in_schema=False)
def admin_open_policy_document(policy_id: str):
    policy = policy_repository.get_policy(policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    source = Path(str(policy.get("file") or "")).name
    if not source:
        raise HTTPException(status_code=404, detail="Policy document not configured")
    path = _safe_child(policy_repository.policies_dir, source)
    return _inline_file_response(path)


@router.get("/admin/api/runs")
def admin_runs(limit: int = 100):
    return {"runs": analysis_repository.list_runs(limit=min(max(limit, 1), 500))}


@router.get("/admin/api/runs/{run_id}")
def admin_run(run_id: str):
    run = analysis_repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return run


@router.get("/admin/api/telegram/sessions")
def admin_telegram_sessions(limit: int = 100):
    sessions = telegram_repository.list_sessions(limit=min(max(limit, 1), 500))
    for session in sessions:
        session["chat_id_masked"] = telegram_service.intake.mask_chat_id(session["chat_id"])
    return {
        "configuration": TelegramSettings.from_env().public_status(),
        "sessions": sessions,
    }


@router.get("/admin/api/telegram/sessions/{chat_id}/messages")
def admin_telegram_messages(chat_id: str):
    return {"messages": telegram_repository.list_messages(chat_id, limit=500)}


@router.post("/admin/api/telegram/simulate")
def admin_telegram_simulate(payload: TelegramSimulationRequest):
    return telegram_service.process_simulated_text(payload.chat_id, payload.text)


@router.get("/admin/api/mlops")
def admin_mlops():
    return {
        "monitoring": mlops_service.monitoring_summary(),
        "data_quality": mlops_service.production_data_quality(),
        "data_drift": mlops_service.drift_report(),
        "prediction_drift": mlops_service.prediction_drift(),
        "performance": mlops_service.production_performance(),
        "retraining": mlops_service.retraining_status(),
    }


@router.get("/admin/api/rag")
def admin_rag():
    return _safe_rag_info()
