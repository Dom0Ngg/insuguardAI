from typing import Any, Dict

from backend.agents.base_agent import BaseAgent
from backend.ml.fraud_model import FraudDetectionModel


class ValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente de validación")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        features = input_data.get("features", {}) or {}
        validation = FraudDetectionModel.validate_feature_payload(features)
        issues = []

        if not input_data.get("claim_id"):
            issues.append("Falta claim_id.")
        if validation["missing_features"]:
            issues.append("Faltan variables del esquema Kaggle: " + ", ".join(validation["missing_features"]))

        policy_type = str(features.get("PolicyType", ""))
        base_policy = str(features.get("BasePolicy", ""))
        if policy_type and base_policy:
            # In the Kaggle dataset, VehicleCategory is not always the prefix of
            # PolicyType (e.g. "Sedan - Liability" can have VehicleCategory="Sport").
            # The consistent contractual relationship is the policy coverage suffix.
            policy_base = policy_type.rsplit(" - ", 1)[-1].strip()
            if policy_base != base_policy:
                issues.append(
                    "PolicyType no coincide con BasePolicy "
                    f"(PolicyType implica: {policy_base}; BasePolicy recibido: {base_policy})."
                )

        document_info = input_data.get("document_extraction", {}) or {}
        missing_documents = document_info.get("documents_missing", [])
        if missing_documents:
            issues.append("Hay adjuntos referenciados que no existen en almacenamiento: " + ", ".join(missing_documents))

        return {
            "status": "success",
            "output": {
                "validation_status": "valid" if not issues else "warning",
                "issues": issues,
                "feature_validation": validation,
            },
        }
