from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.ml.fraud_model import FraudDetectionModel
from backend.mlops.monitoring_repository import MonitoringRepository


class MLOpsService:
    """Monitoring, drift detection and production performance for InsurGuard."""

    MIN_DRIFT_SAMPLE = 500
    DRIFT_WINDOW_SIZE = 500
    MIN_PERFORMANCE_SAMPLE = 100
    MIN_CLASS_COUNT_FOR_PERFORMANCE = 10
    PSI_STABLE_MAX = 0.10
    PSI_MODERATE_MAX = 0.25
    ROC_AUC_RELATIVE_DROP_ALERT = 0.10
    RECALL_ABSOLUTE_DROP_ALERT = 0.10

    def __init__(self, monitoring_db_path: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parents[2]
        self.dataset_path = self.project_root / "data" / "kaggle" / "fraud_oracle.csv"
        self.monitoring = MonitoringRepository(monitoring_db_path)

    def model_info(self) -> Dict[str, Any]:
        return FraudDetectionModel().get_model_info()

    def _reference_frame(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        return pd.read_csv(self.dataset_path)

    def data_quality(self) -> Dict[str, Any]:
        if not self.dataset_path.exists():
            return {"status": "missing_dataset", "path": str(self.dataset_path)}
        df = self._reference_frame()
        target = FraudDetectionModel.TARGET
        positives = int(df[target].sum())
        result = {
            "status": "ok",
            "dataset": "Vehicle Insurance Claim Fraud Detection (Kaggle)",
            "path": "data/kaggle/fraud_oracle.csv",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values_total": int(df.isna().sum().sum()),
            "fraud_cases": positives,
            "non_fraud_cases": int(len(df) - positives),
            "fraud_prevalence": round(positives / len(df), 4),
            "age_zero_rows": int((pd.to_numeric(df["Age"], errors="coerce") == 0).sum()),
            "age_zero_handling": "Converted to missing inside the model pipeline and imputed with KNNImputer.",
            "required_model_columns_present": all(
                col in df.columns for col in FraudDetectionModel.FEATURES + [target]
            ),
        }
        result["production_monitoring"] = self.production_data_quality()
        return result

    def record_prediction(
        self,
        *,
        claim_id: str,
        features: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        fraud_score = analysis.get("fraud_score")
        if fraud_score is None:
            return None
        return self.monitoring.record_prediction(
            claim_id=claim_id,
            features=features,
            fraud_score=float(fraud_score),
            risk_level=analysis.get("risk_level"),
            manual_review_required=bool(analysis.get("manual_review_required", False)),
            model_version=FraudDetectionModel.MODEL_VERSION,
        )

    def set_ground_truth(self, claim_id: str, fraud_label: int) -> Dict[str, Any]:
        return self.monitoring.set_ground_truth(claim_id, fraud_label)

    def monitoring_summary(self) -> Dict[str, Any]:
        summary = self.monitoring.summary()
        summary.update(
            {
                "monitoring_policy": {
                    "monitor_from_first_external_claim": True,
                    "minimum_claims_for_drift": self.MIN_DRIFT_SAMPLE,
                    "drift_window_size": self.DRIFT_WINDOW_SIZE,
                    "minimum_labeled_claims_for_performance": self.MIN_PERFORMANCE_SAMPLE,
                    "minimum_examples_per_class_for_performance": self.MIN_CLASS_COUNT_FOR_PERFORMANCE,
                    "automatic_retraining": False,
                    "retraining_policy": (
                        "No volume-percentage trigger is used. Retraining is considered only "
                        "after evidence of production degradation; significant drift triggers review, "
                        "not automatic retraining."
                    ),
                },
                "drift_sample_progress": round(
                    min(1.0, summary["unique_claims"] / self.MIN_DRIFT_SAMPLE), 4
                ),
                "performance_sample_progress": round(
                    min(1.0, summary["labeled_claims"] / self.MIN_PERFORMANCE_SAMPLE), 4
                ),
            }
        )
        return summary

    def production_data_quality(self) -> Dict[str, Any]:
        rows = self.monitoring.latest_predictions()
        if not rows:
            return {
                "status": "no_production_data",
                "unique_claims": 0,
                "message": "No external claims have been recorded through POST /claim-analysis yet.",
            }

        reference = self._reference_frame()
        unknown_by_feature: Dict[str, Dict[str, Any]] = {}
        missing_by_feature: Dict[str, int] = {name: 0 for name in FraudDetectionModel.FEATURES}

        for feature in FraudDetectionModel.CATEGORICAL_FEATURES:
            reference_values = set(reference[feature].dropna().astype(str).unique())
            unknown = 0
            for row in rows:
                value = row["features"].get(feature)
                if value in (None, ""):
                    missing_by_feature[feature] += 1
                elif str(value) not in reference_values:
                    unknown += 1
            unknown_by_feature[feature] = {
                "unknown_count": unknown,
                "unknown_rate": round(unknown / len(rows), 4),
            }

        for feature in FraudDetectionModel.NUMERICAL_FEATURES:
            for row in rows:
                if row["features"].get(feature) in (None, ""):
                    missing_by_feature[feature] += 1

        missing_total = int(sum(missing_by_feature.values()))
        unknown_total = int(sum(v["unknown_count"] for v in unknown_by_feature.values()))
        return {
            "status": "ok",
            "unique_claims": len(rows),
            "missing_values_total": missing_total,
            "missing_by_feature": missing_by_feature,
            "unknown_categorical_values_total": unknown_total,
            "unknown_categories_by_feature": unknown_by_feature,
            "schema_note": "FastAPI validates required fields before scoring; unknown categories are still monitored because OneHotEncoder ignores unseen values.",
        }

    @staticmethod
    def _psi_from_counts(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
        epsilon = 1e-6
        ref = reference_counts.astype(float)
        cur = current_counts.astype(float)
        ref = ref / max(ref.sum(), 1.0)
        cur = cur / max(cur.sum(), 1.0)
        ref = np.clip(ref, epsilon, None)
        cur = np.clip(cur, epsilon, None)
        return float(np.sum((cur - ref) * np.log(cur / ref)))

    @classmethod
    def _psi_status(cls, value: float) -> str:
        if value < cls.PSI_STABLE_MAX:
            return "stable"
        if value <= cls.PSI_MODERATE_MAX:
            return "moderate"
        return "significant"

    @staticmethod
    def _categorical_psi(reference: pd.Series, current: pd.Series) -> float:
        ref = reference.fillna("__MISSING__").astype(str)
        cur = current.fillna("__MISSING__").astype(str)
        categories = sorted(set(ref.unique()).union(set(cur.unique())))
        ref_counts = np.array([(ref == value).sum() for value in categories], dtype=float)
        cur_counts = np.array([(cur == value).sum() for value in categories], dtype=float)
        return MLOpsService._psi_from_counts(ref_counts, cur_counts)

    @staticmethod
    def _numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        ref = pd.to_numeric(reference, errors="coerce")
        cur = pd.to_numeric(current, errors="coerce")
        clean_ref = ref.dropna()
        if clean_ref.empty:
            return 0.0
        quantiles = np.unique(np.quantile(clean_ref, np.linspace(0, 1, bins + 1)))
        if len(quantiles) <= 2:
            midpoint = float(clean_ref.median())
            edges = np.array([-np.inf, midpoint, np.inf])
        else:
            edges = quantiles.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
        ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
        cur_bins = pd.cut(cur, bins=edges, include_lowest=True)
        categories = ref_bins.cat.categories
        ref_counts = ref_bins.value_counts(sort=False).reindex(categories, fill_value=0).to_numpy()
        cur_counts = cur_bins.value_counts(sort=False).reindex(categories, fill_value=0).to_numpy()
        # Missing values are an explicit extra bucket.
        ref_counts = np.append(ref_counts, ref.isna().sum())
        cur_counts = np.append(cur_counts, cur.isna().sum())
        return MLOpsService._psi_from_counts(ref_counts, cur_counts)

    def _production_feature_frame(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            [{name: row["features"].get(name) for name in FraudDetectionModel.FEATURES} for row in rows],
            columns=FraudDetectionModel.FEATURES,
        )

    def drift_report(self) -> Dict[str, Any]:
        rows = self.monitoring.latest_predictions()
        sample_size = len(rows)
        if sample_size < self.MIN_DRIFT_SAMPLE:
            return {
                "status": "insufficient_data",
                "production_claims": sample_size,
                "minimum_required": self.MIN_DRIFT_SAMPLE,
                "remaining": self.MIN_DRIFT_SAMPLE - sample_size,
                "reason": "Data drift is not reported as statistically operational until the minimum independent production window is reached.",
            }

        # Once monitoring is active, evaluate the most recent independent
        # production window so recent changes are not diluted by old history.
        window_rows = rows[-self.DRIFT_WINDOW_SIZE :]
        reference = self._reference_frame()[FraudDetectionModel.FEATURES]
        current = self._production_feature_frame(window_rows)
        feature_results: List[Dict[str, Any]] = []
        for feature in FraudDetectionModel.FEATURES:
            if feature in FraudDetectionModel.NUMERICAL_FEATURES:
                psi = self._numeric_psi(reference[feature], current[feature])
                feature_type = "numerical"
            else:
                psi = self._categorical_psi(reference[feature], current[feature])
                feature_type = "categorical"
            feature_results.append(
                {
                    "feature": feature,
                    "type": feature_type,
                    "psi": round(psi, 4),
                    "status": self._psi_status(psi),
                }
            )

        feature_results.sort(key=lambda row: row["psi"], reverse=True)
        significant = sum(row["status"] == "significant" for row in feature_results)
        moderate = sum(row["status"] == "moderate" for row in feature_results)
        if significant:
            overall = "alert"
        elif moderate:
            overall = "watch"
        else:
            overall = "stable"
        return {
            "status": "evaluated",
            "overall_status": overall,
            "production_claims_total": sample_size,
            "production_claims_in_window": len(window_rows),
            "reference_claims": int(len(reference)),
            "method": "Population Stability Index (PSI)",
            "thresholds": {
                "stable": "PSI < 0.10",
                "moderate": "0.10 <= PSI <= 0.25",
                "significant": "PSI > 0.25",
            },
            "significant_features": significant,
            "moderate_features": moderate,
            "features": feature_results,
        }

    def _reference_prediction_scores(self) -> np.ndarray:
        # Use the independent holdout score distribution saved at training
        # time. This is faster than rescoring the full Kaggle corpus and
        # avoids using in-sample serving predictions as the drift baseline.
        model = FraudDetectionModel()
        path = model.holdout_scores_path
        if path.exists():
            return np.load(path).astype(float)
        return np.array([], dtype=float)

    def prediction_drift(self) -> Dict[str, Any]:
        rows = self.monitoring.latest_predictions()
        sample_size = len(rows)
        if sample_size < self.MIN_DRIFT_SAMPLE:
            return {
                "status": "insufficient_data",
                "production_claims": sample_size,
                "minimum_required": self.MIN_DRIFT_SAMPLE,
            }
        window_rows = rows[-self.DRIFT_WINDOW_SIZE :]
        reference_scores = self._reference_prediction_scores()
        current_scores = np.array([row["fraud_score"] for row in window_rows], dtype=float)
        psi = self._numeric_psi(pd.Series(reference_scores), pd.Series(current_scores))
        return {
            "status": "evaluated",
            "method": "PSI over model fraud-score distribution",
            "psi": round(psi, 4),
            "drift_status": self._psi_status(psi),
            "reference_mean_fraud_score": round(float(np.mean(reference_scores)), 4),
            "production_mean_fraud_score": round(float(np.mean(current_scores)), 4),
            "reference_alert_rate": round(
                float(np.mean(reference_scores >= FraudDetectionModel().decision_threshold)), 4
            ),
            "production_alert_rate": round(
                float(np.mean(current_scores >= FraudDetectionModel().decision_threshold)), 4
            ),
            "production_claims_total": sample_size,
            "production_claims_in_window": len(window_rows),
        }

    @staticmethod
    def _classification_metrics(
        labels: np.ndarray, scores: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        predictions = (scores >= threshold).astype(int)
        return {
            "threshold": threshold,
            "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(labels, predictions, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        }

    def production_performance(self) -> Dict[str, Any]:
        rows = [row for row in self.monitoring.latest_predictions() if row["FraudFound_P"] is not None]
        labels = np.array([row["FraudFound_P"] for row in rows], dtype=int)
        scores = np.array([row["fraud_score"] for row in rows], dtype=float)
        positives = int(labels.sum()) if len(labels) else 0
        negatives = int(len(labels) - positives)

        if len(rows) < self.MIN_PERFORMANCE_SAMPLE or min(positives, negatives) < self.MIN_CLASS_COUNT_FOR_PERFORMANCE:
            return {
                "status": "insufficient_labeled_data",
                "labeled_claims": len(rows),
                "positive_labels": positives,
                "negative_labels": negatives,
                "minimum_labeled_claims": self.MIN_PERFORMANCE_SAMPLE,
                "minimum_examples_per_class": self.MIN_CLASS_COUNT_FOR_PERFORMANCE,
                "reason": "Production performance requires enough delayed ground-truth labels and representation of both classes.",
            }

        roc_auc = float(roc_auc_score(labels, scores))
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(recall_curve, precision_curve))
        model = FraudDetectionModel()
        decision_metrics = self._classification_metrics(labels, scores, model.decision_threshold)

        baseline = model.get_model_info()["metrics"]["holdout"]
        baseline_roc = float(baseline["roc_auc"])
        baseline_recall = float(baseline["at_decision_threshold"]["recall"])
        roc_relative_drop = max(0.0, (baseline_roc - roc_auc) / baseline_roc) if baseline_roc else 0.0
        recall_absolute_drop = max(0.0, baseline_recall - float(decision_metrics["recall"]))
        degraded = (
            roc_relative_drop >= self.ROC_AUC_RELATIVE_DROP_ALERT
            or recall_absolute_drop >= self.RECALL_ABSOLUTE_DROP_ALERT
        )

        return {
            "status": "evaluated",
            "labeled_claims": len(rows),
            "positive_labels": positives,
            "negative_labels": negatives,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "pr_auc_definition": "trapezoidal AUC(recall, precision), matching the deployed benchmark",
            "at_decision_threshold": decision_metrics,
            "holdout_baseline": {
                "roc_auc": baseline_roc,
                "pr_auc": float(baseline["pr_auc"]),
                "decision_threshold_recall": baseline_recall,
            },
            "degradation": {
                "detected": degraded,
                "roc_auc_relative_drop": round(roc_relative_drop, 4),
                "recall_absolute_drop": round(recall_absolute_drop, 4),
                "alert_if_roc_auc_relative_drop_at_least": self.ROC_AUC_RELATIVE_DROP_ALERT,
                "alert_if_recall_absolute_drop_at_least": self.RECALL_ABSOLUTE_DROP_ALERT,
            },
        }

    def retraining_status(self) -> Dict[str, Any]:
        summary = self.monitoring_summary()
        drift = self.drift_report()
        prediction_drift = self.prediction_drift()
        performance = self.production_performance()

        drift_alert = drift.get("overall_status") == "alert"
        prediction_drift_alert = prediction_drift.get("drift_status") == "significant"
        performance_degraded = bool(performance.get("degradation", {}).get("detected", False))

        reasons: List[str] = []
        if drift_alert:
            reasons.append("Significant feature drift was detected in the production window.")
        if prediction_drift_alert:
            reasons.append("Significant prediction-score drift was detected in production.")
        if performance_degraded:
            reasons.append("Production performance degradation crossed an alert threshold.")

        # Drift alone is a reason to investigate. A retraining recommendation is only made
        # when delayed ground truth confirms that model performance has degraded.
        return {
            "status": "review_due" if reasons else "continue_monitoring",
            "automatic_retraining": False,
            "retraining_recommended": performance_degraded,
            "formal_review_due": bool(reasons),
            "reasons": reasons,
            "production_claims": summary["unique_claims"],
            "labeled_production_claims": summary["labeled_claims"],
            "drift_status": drift.get("overall_status", drift.get("status")),
            "prediction_drift_status": prediction_drift.get(
                "drift_status", prediction_drift.get("status")
            ),
            "performance_status": performance.get("status"),
            "decision_policy": {
                "feature_or_prediction_drift": "human review / investigation",
                "confirmed_performance_degradation": "retraining recommended",
                "volume_percentage_trigger": None,
            },
            "note": (
                "InsurGuard never retrains automatically and does not use a percentage of the "
                "training dataset as a retraining trigger. Drift prompts investigation; retraining "
                "is recommended only when production labels confirm performance degradation."
            ),
        }
