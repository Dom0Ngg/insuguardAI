from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.ml.feature_engineering import KaggleFraudFeatureEngineer


class FraudDetectionModel:
    """Exact deployment of the winning fraud model from the supplied TFM benchmark.

    The benchmark winner is a cost-sensitive Logistic Regression with L1
    regularisation. This class reproduces the supplied preprocessing pipeline,
    selected feature space, class weighting, hyperparameter ``C`` and OOF
    threshold calibration. The target ``FraudFound_P`` is never accepted as an
    inference feature.
    """

    MODEL_VERSION = "3.0-kaggle-logistic-l1-benchmark-exact"
    MODEL_TYPE = "LogisticRegressionL1"
    MODEL_NAME = "InsurGuardVehicleFraudLogisticL1"
    TARGET = "FraudFound_P"

    # Exact winner recovered from the benchmark artefacts supplied with the TFM.
    LOGISTIC_C = 0.001
    SELECTOR_C = 0.1
    VARIANCE_THRESHOLD = 0.01
    SELECTOR_MAX_FEATURES = 65
    EXPECTED_SELECTED_FEATURE_COUNT = 24
    EXPECTED_NONZERO_CLASSIFIER_COEFFICIENTS = 2

    # Exact OOF threshold stored by the benchmark. During retraining the code
    # recalculates it with the same 5-fold procedure and records both values.
    STUDY_DECISION_THRESHOLD = 0.5748224375777465
    DECISION_THRESHOLD = STUDY_DECISION_THRESHOLD
    REVIEW_THRESHOLD = DECISION_THRESHOLD
    HIGH_RISK_THRESHOLD = DECISION_THRESHOLD

    # Both sequential identifiers are excluded from serving. RepNumber was not
    # retained by the benchmark selector and removing it reproduces the exact
    # published 24 selected features and test metrics.
    IDENTIFIER_COLUMNS = ["PolicyNumber", "RepNumber"]

    # 30 raw explanatory fields used by the exact preprocessing/engineering
    # stage after removing identifiers and target.
    FEATURES = [
        "Month",
        "WeekOfMonth",
        "DayOfWeek",
        "Make",
        "AccidentArea",
        "DayOfWeekClaimed",
        "MonthClaimed",
        "WeekOfMonthClaimed",
        "Sex",
        "MaritalStatus",
        "Age",
        "Fault",
        "PolicyType",
        "VehicleCategory",
        "VehiclePrice",
        "Deductible",
        "DriverRating",
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "PastNumberOfClaims",
        "AgeOfVehicle",
        "AgeOfPolicyHolder",
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "NumberOfSuppliments",
        "AddressChange_Claim",
        "NumberOfCars",
        "Year",
        "BasePolicy",
    ]

    CATEGORICAL_FEATURES = [
        "Month",
        "DayOfWeek",
        "Make",
        "AccidentArea",
        "DayOfWeekClaimed",
        "MonthClaimed",
        "Sex",
        "MaritalStatus",
        "Fault",
        "PolicyType",
        "VehicleCategory",
        "VehiclePrice",
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "PastNumberOfClaims",
        "AgeOfVehicle",
        "AgeOfPolicyHolder",
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "NumberOfSuppliments",
        "AddressChange_Claim",
        "NumberOfCars",
        "BasePolicy",
    ]

    # Exact numerical feature space after the supplied feature-engineering code.
    MODEL_NUMERICAL_FEATURES = [
        "WeekOfMonth",
        "WeekOfMonthClaimed",
        "Age",
        "Deductible",
        "DriverRating",
        "Year",
        "Claim_Reporting_Lag_Months",
        "Is_Late_Claim_Report",
        "Is_Weekend_Accident",
        "Is_Weekend_Claimed",
        "Is_Month_Boundary",
        "PastClaims_Num",
        "Has_Past_Claims",
        "Cars_Involved_Num",
        "Vehicle_Price_Approx",
        "Deductible_to_Price_Ratio",
        "Is_Young_Driver",
        "Is_Senior_Driver",
        "Driver_vs_PolicyHolder_Age_Diff",
        "Driver_Is_Not_Main_Holder",
        "Young_Driver_Expensive_Car",
        "No_Police_No_Witness",
        "At_Fault_Without_Police",
        "Recent_Address_Change",
        "Accident_Right_After_Policy",
    ]

    # MLOps treats the original numeric fields as numerical and the rest as
    # categorical for drift calculations.
    NUMERICAL_FEATURES = [
        "WeekOfMonth",
        "WeekOfMonthClaimed",
        "Age",
        "Deductible",
        "DriverRating",
        "Year",
    ]

    def __init__(self, force_retrain: bool = False):
        self.project_root = Path(__file__).resolve().parents[2]
        self.training_file = self.project_root / "data" / "kaggle" / "fraud_oracle.csv"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_path = self.models_dir / "fraud_pipeline.joblib"
        self.metadata_path = self.models_dir / "model_metadata.json"
        self.evaluations_dir = self.project_root / "artifacts" / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        self.holdout_metrics_path = self.evaluations_dir / "holdout_metrics.json"
        self.holdout_scores_path = self.evaluations_dir / "holdout_scores.npy"

        self.pipeline: Optional[Pipeline] = None
        self.training_summary: Dict[str, Any] = {"status": "not_trained"}
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: List[Dict[str, Any]] = []
        self.selected_features: List[str] = []
        self.decision_threshold: float = self.STUDY_DECISION_THRESHOLD
        self._shap_explainer = None

        if not force_retrain and self._load_cached_artifacts():
            return
        self._train()

    def _dataset_sha256(self) -> Optional[str]:
        if not self.training_file.exists():
            return None
        digest = hashlib.sha256()
        with open(self.training_file, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_cached_artifacts(self) -> bool:
        if (
            not self.pipeline_path.exists()
            or not self.metadata_path.exists()
            or not self.holdout_scores_path.exists()
        ):
            return False
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("model_version") != self.MODEL_VERSION:
                return False
            if metadata.get("training_dataset_sha256") != self._dataset_sha256():
                return False
            self.pipeline = joblib.load(self.pipeline_path)
            self.training_summary = metadata.get("training_summary", {})
            self.metrics = metadata.get("metrics", {})
            self.feature_importance = metadata.get("top_feature_importance", [])
            self.selected_features = metadata.get("selected_features", [])
            self.decision_threshold = float(
                metadata.get("operational_thresholds", {}).get(
                    "decision_threshold", self.STUDY_DECISION_THRESHOLD
                )
            )
            return self.training_summary.get("status") == "trained"
        except Exception:
            return False

    @staticmethod
    def _class_weights(y_train: pd.Series) -> Tuple[Dict[int, float], float]:
        negatives = int((y_train == 0).sum())
        positives = int((y_train == 1).sum())
        scale_pos = negatives / positives
        return {0: 1.0, 1: float(scale_pos)}, float(scale_pos)

    @staticmethod
    def _logistic(*, C: float, class_weight: Dict[int, float]) -> LogisticRegression:
        return LogisticRegression(
            solver="liblinear",
            penalty="l1",
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )

    def _build_pipeline(self, class_weight: Dict[int, float]) -> Pipeline:
        # Exact numerical branch from Preprocessing.py:
        # FeatureUnion(StandardScaler -> KNNImputer, MissingIndicator).
        knn_branch = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ]
        )
        indicator_branch = Pipeline(
            [("indicator", MissingIndicator(features="missing-only"))]
        )
        numerical = FeatureUnion(
            [
                ("knn_imputed", knn_branch),
                ("missing_flags", indicator_branch),
            ]
        )

        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numerical, self.MODEL_NUMERICAL_FEATURES),
                ("cat", categorical, self.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

        selector_model = self._logistic(
            C=self.SELECTOR_C, class_weight=class_weight
        )
        classifier = self._logistic(
            C=self.LOGISTIC_C, class_weight=class_weight
        )

        return Pipeline(
            [
                ("feature_engineering", KaggleFraudFeatureEngineer()),
                ("preprocessor", preprocessor),
                (
                    "variance_filter",
                    VarianceThreshold(threshold=self.VARIANCE_THRESHOLD),
                ),
                (
                    "feature_selector",
                    SelectFromModel(
                        estimator=selector_model,
                        max_features=self.SELECTOR_MAX_FEATURES,
                        threshold="mean",
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    @staticmethod
    def _pr_auc(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
        """PR-AUC definition used by the supplied benchmark: trapezoidal AUC."""
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        return float(auc(recall, precision))

    @staticmethod
    def _threshold_metrics(
        y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        return {
            "threshold": round(float(threshold), 6),
            "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        }

    def _selected_feature_names(self, fitted: Optional[Pipeline] = None) -> List[str]:
        pipeline = fitted or self.pipeline
        if pipeline is None:
            return []
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            num_step = preprocessor.named_transformers_["num"]
            indicator = num_step.transformer_list[1][1].named_steps["indicator"]
            missing_indices = indicator.features_
            flag_cols = [
                f"{self.MODEL_NUMERICAL_FEATURES[idx]}_is_missing"
                for idx in missing_indices
            ]

            cat_step = preprocessor.named_transformers_["cat"]
            onehot = cat_step.named_steps["onehot"]
            onehot_cols = onehot.get_feature_names_out(self.CATEGORICAL_FEATURES).tolist()

            all_names = np.asarray(
                self.MODEL_NUMERICAL_FEATURES + flag_cols + onehot_cols,
                dtype=object,
            )
            variance = pipeline.named_steps["variance_filter"]
            selector = pipeline.named_steps["feature_selector"]
            names_after_variance = all_names[variance.get_support()]
            return [str(x) for x in names_after_variance[selector.get_support()].tolist()]
        except Exception:
            return []

    def _transform_for_classifier(
        self, frame: pd.DataFrame, fitted: Optional[Pipeline] = None
    ) -> np.ndarray:
        pipeline = fitted or self.pipeline
        if pipeline is None:
            raise RuntimeError("Fraud model is not trained")
        engineered = pipeline.named_steps["feature_engineering"].transform(frame)
        transformed = pipeline.named_steps["preprocessor"].transform(engineered)
        transformed = pipeline.named_steps["variance_filter"].transform(transformed)
        transformed = pipeline.named_steps["feature_selector"].transform(transformed)
        return np.asarray(transformed, dtype=float)

    def _calibrate_threshold(
        self, fitted_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
    ) -> float:
        """Reproduce the benchmark's 5-fold OOF F1 threshold calibration.

        The supplied benchmark calibrates only the already-selected classifier
        on the processed training matrix. We intentionally mirror that exact
        procedure instead of refitting preprocessing inside each fold.
        """
        processed_train = self._transform_for_classifier(X_train, fitted_pipeline)
        classifier = clone(fitted_pipeline.named_steps["classifier"])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probabilities = cross_val_predict(
            classifier,
            processed_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(
            y_train, oof_probabilities
        )
        f1_values = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        best_index = int(np.argmax(f1_values[:-1]))
        return float(thresholds[best_index])

    def _calculate_feature_importance(self) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            return []
        classifier = self.pipeline.named_steps["classifier"]
        coefficients = classifier.coef_[0]
        rows = [
            {
                "feature": str(name),
                "coefficient": round(float(coef), 6),
                "absolute_importance": round(abs(float(coef)), 6),
                "direction": "fraud" if coef > 0 else "legitimate",
            }
            for name, coef in zip(self.selected_features, coefficients)
        ]
        rows.sort(key=lambda row: row["absolute_importance"], reverse=True)
        return rows

    def _save_artifacts(self) -> None:
        if self.pipeline is None:
            return
        joblib.dump(self.pipeline, self.pipeline_path, compress=3)
        metadata = {
            "model_name": self.MODEL_NAME,
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "training_file": str(self.training_file.relative_to(self.project_root)),
            "training_dataset_sha256": self._dataset_sha256(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_summary": self.training_summary,
            "metrics": self.metrics,
            "operational_thresholds": {
                "decision_threshold": self.decision_threshold,
                "study_threshold": self.STUDY_DECISION_THRESHOLD,
                "source": "5-fold OOF Precision-Recall/F1 calibration from supplied benchmark",
            },
            "selected_features": self.selected_features,
            "top_feature_importance": self.feature_importance[:30],
            "artifacts": {
                "pipeline_path": str(self.pipeline_path.relative_to(self.project_root)),
                "metadata_path": str(self.metadata_path.relative_to(self.project_root)),
                "holdout_metrics_path": str(self.holdout_metrics_path.relative_to(self.project_root)),
                "holdout_scores_path": str(self.holdout_scores_path.relative_to(self.project_root)),
            },
        }
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.holdout_metrics_path.write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _train(self) -> None:
        if not self.training_file.exists():
            self.training_summary = {
                "status": "not_trained",
                "reason": "Kaggle training dataset not found",
                "expected_file": str(self.training_file.relative_to(self.project_root)),
            }
            return

        df = pd.read_csv(self.training_file)
        required = set(self.FEATURES + [self.TARGET])
        missing = sorted(required - set(df.columns))
        if missing:
            self.training_summary = {
                "status": "not_trained",
                "reason": "Training dataset is missing required columns",
                "missing_columns": missing,
            }
            return

        X_all = df[self.FEATURES].copy()
        y_all = df[self.TARGET].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.20,
            random_state=42,
            stratify=y_all,
        )

        class_weight, scale_pos = self._class_weights(y_train)
        pipeline = self._build_pipeline(class_weight)
        pipeline.fit(X_train, y_train)

        # Reproduce the exact OOF threshold-calibration procedure.
        calibrated_threshold = self._calibrate_threshold(pipeline, X_train, y_train)
        self.decision_threshold = calibrated_threshold

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        np.save(self.holdout_scores_path, np.asarray(probabilities, dtype=float))
        decision_metrics = self._threshold_metrics(
            y_test, probabilities, self.decision_threshold
        )

        self.metrics = {
            "holdout": {
                "test_size": int(len(y_test)),
                "fraud_cases": int(y_test.sum()),
                "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 6),
                "pr_auc": round(self._pr_auc(y_test, probabilities), 6),
                "pr_auc_definition": "trapezoidal AUC(recall, precision), matching the supplied TFM benchmark",
                "average_precision": round(
                    float(average_precision_score(y_test, probabilities)), 6
                ),
                "at_decision_threshold": decision_metrics,
            },
            "benchmark_reproduction": {
                "expected_pr_auc": 0.5264554156198319,
                "expected_roc_auc": 0.755226872267231,
                "expected_recall": 0.9243243243243243,
                "expected_precision": 0.12555066079295155,
                "expected_f1": 0.22107304460245636,
                "expected_confusion_matrix": [[1708, 1191], [14, 171]],
                "expected_threshold": self.STUDY_DECISION_THRESHOLD,
                "threshold_difference": round(
                    abs(self.decision_threshold - self.STUDY_DECISION_THRESHOLD), 12
                ),
            },
        }

        # IMPORTANT: keep the fitted 80% training pipeline as the serving
        # artefact. This is exactly how the supplied benchmark exports its
        # end-to-end model and preserves the 20% holdout as truly unseen data.
        self.pipeline = pipeline
        self.selected_features = self._selected_feature_names(self.pipeline)
        self.feature_importance = self._calculate_feature_importance()

        classifier = self.pipeline.named_steps["classifier"]
        positives = int(y_all.sum())
        self.training_summary = {
            "status": "trained",
            "dataset": "Vehicle Insurance Claim Fraud Detection (Kaggle)",
            "total_samples": int(len(df)),
            "training_samples": int(len(y_train)),
            "holdout_samples": int(len(y_test)),
            "positive_labels": positives,
            "negative_labels": int(len(df) - positives),
            "fraud_prevalence": round(positives / len(df), 6),
            "raw_features": list(self.FEATURES),
            "raw_feature_count": len(self.FEATURES),
            "target": self.TARGET,
            "excluded_identifiers": list(self.IDENTIFIER_COLUMNS),
            "age_zero_rows_treated_as_missing": int((df["Age"] == 0).sum()),
            "preprocessing": {
                "numerical": "FeatureUnion(StandardScaler -> KNNImputer(k=5, weights=distance), MissingIndicator(missing-only))",
                "categorical": "SimpleImputer(Missing) -> OneHotEncoder(drop=first, handle_unknown=ignore)",
                "variance_filter_threshold": self.VARIANCE_THRESHOLD,
                "supervised_selector": "SelectFromModel(LogisticRegression L1, C=0.1, threshold=mean, max_features=65)",
                "selected_encoded_features": len(self.selected_features),
            },
            "selected_features": list(self.selected_features),
            "class_weight": class_weight,
            "scale_positive_weight": scale_pos,
            "regularization": "L1",
            "solver": "liblinear",
            "C": self.LOGISTIC_C,
            "classifier_nonzero_coefficients": int(np.count_nonzero(classifier.coef_)),
            "decision_threshold": self.decision_threshold,
            "train_test_split": {
                "test_size": 0.20,
                "random_state": 42,
                "stratified": True,
            },
            "threshold_calibration": {
                "method": "5-fold stratified OOF predictions on processed training data; threshold maximizing F1",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
            },
        }
        self._save_artifacts()

    @classmethod
    def validate_feature_payload(cls, features: Dict[str, Any]) -> Dict[str, Any]:
        missing = [name for name in cls.FEATURES if features.get(name) in (None, "")]
        extra = sorted(set(features) - set(cls.FEATURES))
        return {
            "valid": not missing,
            "missing_features": missing,
            "extra_features": extra,
            "supplied_feature_count": len(cls.FEATURES) - len(missing),
            "required_feature_count": len(cls.FEATURES),
        }

    def _risk_level(self, probability: float) -> str:
        return "Alto" if probability >= self.decision_threshold else "Bajo"

    def local_explanation(
        self, features: Dict[str, Any], top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """Return local SHAP contributions for the exact L1 logistic model."""
        if self.pipeline is None:
            return []
        try:
            import shap

            frame = pd.DataFrame(
                [{name: features[name] for name in self.FEATURES}],
                columns=self.FEATURES,
            )
            row = self._transform_for_classifier(frame)
            if self._shap_explainer is None:
                reference = pd.read_csv(self.training_file)[self.FEATURES].iloc[:250].copy()
                background = self._transform_for_classifier(reference)
                classifier = self.pipeline.named_steps["classifier"]
                self._shap_explainer = shap.LinearExplainer(classifier, background)
            explanation = self._shap_explainer(row)
            values = np.asarray(explanation.values)[0]
            rows = [
                {
                    "feature": name,
                    "shap_value": round(float(value), 6),
                    "direction": "fraud" if value > 0 else "legitimate",
                }
                for name, value in zip(self.selected_features, values)
            ]
            rows.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
            return rows[:top_k]
        except Exception:
            return []

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        validation = self.validate_feature_payload(features)
        base = {
            "ml_model_used": self.MODEL_TYPE,
            "ml_model_version": self.MODEL_VERSION,
            "ml_decision_threshold": round(self.decision_threshold, 6),
            "feature_validation": validation,
        }
        if self.training_summary.get("status") != "trained" or self.pipeline is None:
            return {
                **base,
                "ml_status": "not_trained",
                "ml_fraud_score": None,
                "ml_prediction": None,
                "ml_risk_level": None,
            }
        if not validation["valid"]:
            return {
                **base,
                "ml_status": "invalid_features",
                "ml_status_reason": (
                    f"The claim must contain all {len(self.FEATURES)} raw Kaggle model fields."
                ),
                "ml_fraud_score": None,
                "ml_prediction": None,
                "ml_risk_level": None,
            }

        frame = pd.DataFrame(
            [{name: features[name] for name in self.FEATURES}],
            columns=self.FEATURES,
        )
        probability = float(self.pipeline.predict_proba(frame)[0][1])
        prediction = int(probability >= self.decision_threshold)
        return {
            **base,
            "ml_status": "scored",
            "ml_fraud_score": round(probability, 6),
            "ml_prediction": prediction,
            "ml_risk_level": self._risk_level(probability),
            "manual_review_required": bool(prediction),
            "local_shap": self.local_explanation(features),
        }

    def predict_probabilities(self, frame: pd.DataFrame) -> pd.Series:
        if self.training_summary.get("status") != "trained" or self.pipeline is None:
            raise RuntimeError("Fraud model is not trained")
        missing = [name for name in self.FEATURES if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        probabilities = self.pipeline.predict_proba(frame[self.FEATURES])[:, 1]
        return pd.Series(probabilities, index=frame.index)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": self.MODEL_TYPE,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "training_summary": self.training_summary,
            "metrics": self.metrics,
            "operational_thresholds": {
                "decision_threshold": round(self.decision_threshold, 6),
                "study_threshold": self.STUDY_DECISION_THRESHOLD,
            },
            "selected_features": self.selected_features,
            "top_feature_importance": self.feature_importance[:20],
            "training_file": str(self.training_file.relative_to(self.project_root)),
            "artifacts": {
                "pipeline_path": str(self.pipeline_path.relative_to(self.project_root)),
                "metadata_path": str(self.metadata_path.relative_to(self.project_root)),
                "holdout_metrics_path": str(self.holdout_metrics_path.relative_to(self.project_root)),
                "holdout_scores_path": str(self.holdout_scores_path.relative_to(self.project_root)),
            },
        }
