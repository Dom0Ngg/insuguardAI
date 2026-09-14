from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

from backend.agents.orchestrator_agent import OrchestratorAgent
from backend.mlops.mlops_service import MLOpsService
from backend.services.analysis_repository import AnalysisRepository


class AnalysisService:
    """Single entry point for audited analyses from Swagger, admin and Telegram."""

    def __init__(
        self,
        orchestrator: Optional[OrchestratorAgent] = None,
        mlops_service: Optional[MLOpsService] = None,
        repository: Optional[AnalysisRepository] = None,
    ):
        self.orchestrator = orchestrator or OrchestratorAgent()
        self.mlops_service = mlops_service or MLOpsService()
        self.repository = repository or AnalysisRepository()

    def analyze(
        self,
        claim_data: Dict[str, Any],
        *,
        source_channel: str,
        record_production: bool,
    ) -> Dict[str, Any]:
        started_wall = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        try:
            result = self.orchestrator.analyze_claim(claim_data)
            duration_ms = (time.perf_counter() - started) * 1000
            completed = datetime.now(timezone.utc).isoformat()
            self.repository.record_run(
                claim_id=str(claim_data.get("claim_id")),
                source_channel=source_channel,
                started_at=started_wall,
                completed_at=completed,
                duration_ms=duration_ms,
                status="ok",
                result=result,
            )
            if record_production and result.get("fraud_score") is not None:
                features = claim_data.get("features") or {}
                if hasattr(features, "model_dump"):
                    features = features.model_dump()
                self.mlops_service.record_prediction(
                    claim_id=str(claim_data.get("claim_id")),
                    features=features,
                    analysis=result,
                )
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            self.repository.record_run(
                claim_id=str(claim_data.get("claim_id")),
                source_channel=source_channel,
                started_at=started_wall,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration_ms,
                status="error",
                error=str(exc),
            )
            raise
