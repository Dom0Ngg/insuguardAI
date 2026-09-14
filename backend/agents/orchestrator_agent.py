from __future__ import annotations

import time
from typing import Any, Dict, List

from backend.agents.document_extraction_agent import DocumentExtractionAgent
from backend.agents.explanation_agent import ExplanationAgent
from backend.agents.fraud_detection_agent import FraudDetectionAgent
from backend.agents.rag_coverage_agent import RAGCoverageAgent
from backend.agents.validation_agent import ValidationAgent


class OrchestratorAgent:
    def __init__(self):
        self.document_agent = DocumentExtractionAgent()
        self.validation_agent = ValidationAgent()
        self.fraud_agent = FraudDetectionAgent()
        self.rag_agent = RAGCoverageAgent()
        self.explanation_agent = ExplanationAgent()

    @staticmethod
    def _trace(agent, result: Dict[str, Any], duration_ms: float) -> Dict[str, Any]:
        return {
            "agent_name": agent.name,
            "status": result["status"],
            "duration_ms": round(duration_ms, 2),
            "output": result["output"],
        }

    def _run(self, agent, payload: Dict[str, Any]):
        started = time.perf_counter()
        result = agent.run(payload)
        duration_ms = (time.perf_counter() - started) * 1000
        return result, duration_ms

    def analyze_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run claim pre-analysis.

        The customer-facing claim path does not execute policy RAG. Policy review
        is a supervisor action. RAG is only executed here when an explicit
        policy_id + coverage_query is supplied by an internal/API caller.
        """
        trace: List[Dict[str, Any]] = []

        document_result, elapsed = self._run(self.document_agent, claim_data)
        trace.append(self._trace(self.document_agent, document_result, elapsed))

        validation_input = {**claim_data, "document_extraction": document_result["output"]}
        validation_result, elapsed = self._run(self.validation_agent, validation_input)
        trace.append(self._trace(self.validation_agent, validation_result, elapsed))

        fraud_result, elapsed = self._run(self.fraud_agent, claim_data)
        trace.append(self._trace(self.fraud_agent, fraud_result, elapsed))

        rag_output: Dict[str, Any] = {
            "coverage_result": "No evaluado",
            "rag_answer": "La revisión de póliza corresponde al supervisor.",
            "sources": [],
        }
        if claim_data.get("policy_id") and str(claim_data.get("coverage_query") or "").strip():
            rag_result, elapsed = self._run(self.rag_agent, claim_data)
            trace.append(self._trace(self.rag_agent, rag_result, elapsed))
            rag_output = rag_result["output"]

        fraud = fraud_result["output"]
        explanation_result, elapsed = self._run(
            self.explanation_agent,
            {
                "fraud_score": fraud.get("fraud_score"),
                "risk_level": fraud.get("risk_level"),
                "manual_review_required": fraud.get("manual_review_required", False),
                "coverage_result": rag_output.get("coverage_result", "No evaluado"),
                "validation_issues": validation_result["output"].get("issues", []),
            },
        )
        trace.append(self._trace(self.explanation_agent, explanation_result, elapsed))

        return {
            "claim_id": claim_data.get("claim_id"),
            "fraud_score": fraud.get("fraud_score"),
            "risk_level": fraud.get("risk_level"),
            "manual_review_required": bool(fraud.get("manual_review_required", False)),
            "coverage_result": rag_output.get("coverage_result", "No evaluado"),
            "explanation": explanation_result["output"]["explanation"],
            "agents_trace": trace,
        }
