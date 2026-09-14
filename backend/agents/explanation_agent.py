from typing import Any, Dict

from backend.agents.base_agent import BaseAgent
from backend.ml.fraud_model import FraudDetectionModel


class ExplanationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente explicador")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        score = input_data.get("fraud_score")
        risk = input_data.get("risk_level") or "No calculado"
        review = bool(input_data.get("manual_review_required", False))
        coverage = input_data.get("coverage_result", "No evaluado")
        issues = input_data.get("validation_issues", []) or []

        if score is None:
            explanation = (
                "No se pudo calcular el riesgo de fraude porque el claim no cumple "
                f"el esquema de {len(FraudDetectionModel.FEATURES)} variables de entrada utilizado por el modelo."
            )
        else:
            explanation = (
                "La Regresión Logística con regularización L1 entrenada sobre el dataset de "
                f"siniestros de automóvil estima una probabilidad de fraude de {score:.1%}. "
                f"El umbral operativo calibrado del benchmark es {FraudDetectionModel.DECISION_THRESHOLD:.4f}; "
                f"el nivel de riesgo resultante es {risk}. "
                + (
                    "El expediente supera el umbral y se prioriza para revisión manual. "
                    if review
                    else "El expediente no supera el umbral de revisión manual. "
                )
            )
        if coverage and coverage != "No evaluado":
            explanation += f"Revisión de póliza del supervisor: {coverage}. "
        if issues:
            explanation += "Validación: " + "; ".join(issues) + "."
        else:
            explanation += "Las variables requeridas por el modelo son coherentes con el contrato de entrada."
        return {"status": "success", "output": {"explanation": explanation}}
