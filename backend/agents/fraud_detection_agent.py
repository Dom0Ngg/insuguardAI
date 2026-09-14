from typing import Any, Dict

from backend.agents.base_agent import BaseAgent
from backend.ml.fraud_model import FraudDetectionModel


class FraudDetectionAgent(BaseAgent):
    """Fraud agent backed by the TFM-selected L1 Logistic Regression."""

    def __init__(self):
        super().__init__("Agente de detección de fraude")
        self.ml_model = FraudDetectionModel()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = self.ml_model.predict(input_data.get("features", {}) or {})
        model_info = self.ml_model.get_model_info()
        return {
            "status": "success",
            "output": {
                "fraud_score": result.get("ml_fraud_score"),
                "risk_level": result.get("ml_risk_level"),
                "manual_review_required": bool(result.get("manual_review_required", False)),
                "prediction": result.get("ml_prediction"),
                "ml_status": result.get("ml_status"),
                "ml_status_reason": result.get("ml_status_reason"),
                "model_used": result.get("ml_model_used"),
                "model_version": result.get("ml_model_version"),
                "decision_threshold": result.get("ml_decision_threshold"),
                "feature_validation": result.get("feature_validation"),
                "selected_features": model_info.get("selected_features", []),
                "top_global_feature_importance": model_info.get("top_feature_importance", [])[:8],
                "local_shap": result.get("local_shap", []),
                "note": (
                    "La importancia global se basa en el valor absoluto de los coeficientes L1. "
                    "La explicación local usa valores SHAP y no debe interpretarse como causalidad."
                ),
            },
        }
