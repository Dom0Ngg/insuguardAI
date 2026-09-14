from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import shutil
import tempfile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

from backend.agents.orchestrator_agent import OrchestratorAgent
from backend.admin.router import router as admin_router
from backend.app.schemas import (
    ClaimAnalysisRequest,
    ClaimAnalysisResponse,
    GroundTruthRequest,
    PolicyQueryRequest,
    StoredClaimAnalysisOptions,
)
from backend.ml.fraud_model import FraudDetectionModel
from backend.mlops.mlops_service import MLOpsService
from backend.ocr.ocr_service import OCRService
from backend.rag.vector_store import RAGUnavailableError
from backend.services.analysis_repository import AnalysisRepository
from backend.services.analysis_service import AnalysisService
from backend.services.claim_repository import ClaimRepository
from backend.services.policy_repository import PolicyRepository
from backend.telegram.client import TelegramBotClient
from backend.telegram.intake_agent import TelegramIntakeAgent
from backend.telegram.polling import TelegramPollingRunner
from backend.telegram.repository import TelegramRepository
from backend.telegram.service import TelegramService
from backend.telegram.settings import TelegramSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
claim_repository = ClaimRepository()
policy_repository = PolicyRepository()
mlops_service = MLOpsService()
analysis_repository = AnalysisRepository()
telegram_repository = TelegramRepository()
telegram_client = TelegramBotClient()
telegram_polling_runner = None


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    global telegram_polling_runner
    settings = TelegramSettings.from_env()
    telegram_polling_runner = TelegramPollingRunner(
        service=get_telegram_service(),
        client=telegram_client,
        settings=settings,
    )
    telegram_polling_runner.start()
    try:
        yield
    finally:
        if telegram_polling_runner is not None:
            telegram_polling_runner.stop()


app = FastAPI(
    title="InsurGuard AI",
    version="1.2.0",
    lifespan=app_lifespan,
    description=(
        "TFM para análisis de fraude en siniestros de automóvil. "
        "FastAPI expone la API, Telegram actúa como canal conversacional de entrada y /admin "
        "muestra la trazabilidad completa del sistema."
    ),
)

ADMIN_STATIC_DIR = PROJECT_ROOT / "backend" / "admin" / "static"
app.mount("/admin/static", StaticFiles(directory=str(ADMIN_STATIC_DIR)), name="admin-static")
app.include_router(admin_router)


@lru_cache(maxsize=1)
def get_orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent()


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService(
        orchestrator=get_orchestrator(),
        mlops_service=mlops_service,
        repository=analysis_repository,
    )


@lru_cache(maxsize=1)
def get_telegram_service() -> TelegramService:
    intake = TelegramIntakeAgent(
        repository=telegram_repository,
        claim_repository=claim_repository,
        analysis_service=get_analysis_service(),
    )
    return TelegramService(
        intake_agent=intake,
        repository=telegram_repository,
        client=telegram_client,
    )


def _stored_claim_to_request(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_id": claim["claim_id"],
        "features": claim["features"],
        "documents": claim.get("documents", []),
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "project": "InsurGuard AI",
        "scope": "Vehicle insurance claims only",
        "interfaces": {"admin": "/admin", "swagger": "/docs", "telegram": "/telegram/info"},
        "swagger": "/docs",
        "dataset": "data/kaggle/fraud_oracle.csv",
        "data_structure": ["data/kaggle", "data/policies", "data/claims"],
        "main_endpoints": {
            "admin": "/admin",
            "telegram": "/telegram/info",
            "claims": "/claims",
            "analyze_stored_claim": "/claims/{claim_id}/analysis",
            "analyze_payload": "/claim-analysis",
            "policies": "/policies",
            "rag_info": "/rag/info",
            "rag_reindex": "/rag/reindex",
            "model": "/mlops/model-info",
            "monitoring": "/mlops/monitoring-summary",
            "drift": "/mlops/drift-report",
            "production_performance": "/mlops/production-performance",
            "retraining_status": "/mlops/retraining-status",
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    model = FraudDetectionModel()
    return {
        "status": "ok",
        "model_status": model.training_summary.get("status"),
        "claims_available": len(claim_repository.list_claims()),
        "policies_available": len(policy_repository.list_policies()),
        "tesseract_available": OCRService.tesseract_available(),
        "rag_backend": "PostgreSQL + pgvector + Sentence Transformers",
        "rag_dependencies_available": get_orchestrator().rag_agent.vector_store.dependencies_available(),
        "telegram": {
            **TelegramSettings.from_env().public_status(),
            "poller": telegram_polling_runner.status() if telegram_polling_runner is not None else {"running": False},
        },
        "admin_dashboard": "/admin",
    }


@app.get("/dataset/info")
def dataset_info() -> Dict[str, Any]:
    path = PROJECT_ROOT / "data" / "kaggle" / "fraud_oracle.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Kaggle dataset not found")
    df = pd.read_csv(path)
    target = FraudDetectionModel.TARGET
    positives = int(df[target].sum())
    return {
        "path": "data/kaggle/fraud_oracle.csv",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "model_features": FraudDetectionModel.FEATURES,
        "target": target,
        "fraud_cases": positives,
        "non_fraud_cases": int(len(df) - positives),
        "fraud_prevalence": round(positives / len(df), 4),
        "note": "The target is used for training/evaluation and is never passed as an inference feature.",
    }


@app.get("/claims")
def list_claims():
    return {"total": len(claim_repository.list_claims()), "claims": claim_repository.list_claims()}


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    try:
        return claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")


@app.get("/claims/{claim_id}/analysis", response_model=ClaimAnalysisResponse)
def analyze_stored_claim(claim_id: str):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    return get_analysis_service().analyze(
        _stored_claim_to_request(claim), source_channel="swagger_demo", record_production=False
    )


@app.post("/claims/{claim_id}/analysis", response_model=ClaimAnalysisResponse)
def analyze_stored_claim_with_options(claim_id: str, options: StoredClaimAnalysisOptions):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    request_data = _stored_claim_to_request(claim)
    if options.policy_id is not None:
        request_data["policy_id"] = options.policy_id
    if options.coverage_query is not None:
        request_data["coverage_query"] = options.coverage_query
    return get_analysis_service().analyze(
        request_data, source_channel="swagger_demo", record_production=False
    )


@app.post("/claims/{claim_id}/documents")
async def upload_claim_document(claim_id: str, file: UploadFile = File(...)):
    try:
        claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in claim_repository.get_supported_document_extensions():
        raise HTTPException(status_code=400, detail=f"Unsupported document format: {suffix}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = Path(temp.name)
    try:
        return claim_repository.attach_document(claim_id, temp_path, file.filename or f"document{suffix}")
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/claims/{claim_id}/comparison")
def compare_stored_claim_with_ground_truth(claim_id: str):
    try:
        claim = claim_repository.get_claim(claim_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found")
    analysis = get_analysis_service().analyze(
        _stored_claim_to_request(claim), source_channel="comparison", record_production=False
    )
    truth = claim.get("ground_truth", {}).get("FraudFound_P")
    predicted = None if analysis["fraud_score"] is None else int(analysis["fraud_score"] >= FraudDetectionModel.DECISION_THRESHOLD)
    return {
        "claim_id": claim_id,
        "source_row": claim.get("source", {}).get("row_index"),
        "ground_truth_FraudFound_P": truth,
        "predicted_fraud_alert": predicted,
        "fraud_score": analysis["fraud_score"],
        "correct_at_decision_threshold": None if truth is None or predicted is None else bool(int(truth) == predicted),
        "independent_holdout_evaluation": False,
        "note": "Ground truth is not included in inference features. This endpoint is a demo comparison only; the independent holdout metrics are exposed by /mlops/model-metrics.",
    }


@app.post("/claim-analysis", response_model=ClaimAnalysisResponse)
def analyze_claim(payload: ClaimAnalysisRequest):
    result = get_analysis_service().analyze(
        payload.model_dump(), source_channel="api", record_production=False
    )
    # Keep the public endpoint explicitly bound to the current MLOps service so
    # tests/maintenance can swap its repository without rebuilding the orchestrator.
    mlops_service.record_prediction(
        claim_id=payload.claim_id,
        features=payload.features.model_dump(),
        analysis=result,
    )
    return result


@app.get("/telegram/info")
def telegram_info():
    settings = TelegramSettings.from_env()
    return {
        **settings.public_status(),
        "admin": "/admin",
        "commands": ["/start", "NUEVO", "ESTADO", "ANALIZAR", "CANCELAR"],
        "supported_inbound": ["text", "document", "photo"],
        "receive_mode": "getUpdates long polling",
        "poller": telegram_polling_runner.status() if telegram_polling_runner is not None else {"running": False},
        "setup": "Create the bot with @BotFather, set TELEGRAM_BOT_TOKEN, and recreate only insurguard-api.",
    }


@app.get("/policies")
def list_policies():
    return {"total": len(policy_repository.list_policies()), "policies": policy_repository.list_policies()}


@app.post("/policies/{policy_id}/query")
def query_policy(policy_id: str, payload: PolicyQueryRequest):
    if policy_repository.get_policy(policy_id) is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    result = get_orchestrator().rag_agent.run({"policy_id": policy_id, "coverage_query": payload.query})
    if result.get("status") == "error":
        raise HTTPException(status_code=503, detail=result["output"].get("rag_answer", "RAG unavailable"))
    return result["output"]


@app.get("/rag/info")
def rag_info():
    try:
        return get_orchestrator().rag_agent.vector_store.get_index_summary()
    except RAGUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/rag/reindex")
def rag_reindex(force: bool = Query(default=False, description="Re-embed policies even when unchanged")):
    try:
        store = get_orchestrator().rag_agent.vector_store
        sync = store.sync_policies(force=force, prune=True)
        return {"sync": sync, "index": store.get_index_summary()}
    except RAGUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/mlops/model-info")
def model_info():
    return mlops_service.model_info()


@app.get("/mlops/model-metrics")
def model_metrics():
    info = mlops_service.model_info()
    return info.get("metrics", {})


@app.get("/mlops/data-quality")
def data_quality():
    return mlops_service.data_quality()


@app.get("/mlops/monitoring-summary")
def monitoring_summary():
    return mlops_service.monitoring_summary()


@app.get("/mlops/production-data-quality")
def production_data_quality():
    return mlops_service.production_data_quality()


@app.get("/mlops/drift-report")
def drift_report():
    return mlops_service.drift_report()


@app.get("/mlops/prediction-drift")
def prediction_drift():
    return mlops_service.prediction_drift()


@app.get("/mlops/production-performance")
def production_performance():
    return mlops_service.production_performance()


@app.get("/mlops/retraining-status")
def retraining_status():
    return mlops_service.retraining_status()


@app.put("/mlops/production-claims/{claim_id}/ground-truth")
def set_production_ground_truth(claim_id: str, payload: GroundTruthRequest):
    try:
        return mlops_service.set_ground_truth(claim_id, payload.FraudFound_P)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Claim not found in production monitoring. Analyze it first through POST /claim-analysis.",
        )


@app.get("/system-info")
def system_info():
    return {
        "project": "InsurGuard AI",
        "scope": "Automobile insurance only",
        "frontend": {"admin_dashboard": "/admin", "swagger": "/docs"},
        "input_channels": ["FastAPI", "Telegram Bot API"],
        "data": {
            "kaggle": "data/kaggle/fraud_oracle.csv",
            "policies": "data/policies/",
            "claims": "data/claims/<claim_id>/claim.json",
            "claim_documents_optional": "data/claims/<claim_id>/documents/",
        },
        "model": FraudDetectionModel().get_model_info(),
        "ocr": {
            "tesseract_available": OCRService.tesseract_available(),
            "scope": "policy_pdfs_only",
            "note": "OCR is used while indexing policy PDFs when a page has insufficient native text. Customer attachments are stored as original evidence for supervisor review.",
        },
        "rag": {
            "backend": "PostgreSQL + pgvector",
            "embeddings": "Sentence Transformers multilingual MiniLM",
            "index": "HNSW cosine similarity",
            "strict_policy_filter": True,
            "info_endpoint": "/rag/info",
            "reindex_endpoint": "/rag/reindex",
        },
        "telegram": {
            **TelegramSettings.from_env().public_status(),
            "poller": telegram_polling_runner.status() if telegram_polling_runner is not None else {"running": False},
        },
        "mlops": mlops_service.monitoring_summary()["monitoring_policy"],
    }
