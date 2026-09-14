from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.ml.fraud_model import FraudDetectionModel
from backend.mlops.mlops_service import MLOpsService


client = TestClient(main_module.app)


def _record_row(service: MLOpsService, model: FraudDetectionModel, claim_id: str, row, score: float):
    service.monitoring.record_prediction(
        claim_id=claim_id,
        features={feature: row[feature] for feature in model.FEATURES},
        fraud_score=float(score),
        risk_level=(
            "Alto"
            if score >= model.HIGH_RISK_THRESHOLD
            else "Medio"
            if score >= model.REVIEW_THRESHOLD
            else "Bajo"
        ),
        manual_review_required=bool(score >= model.REVIEW_THRESHOLD),
        model_version=model.MODEL_VERSION,
    )


def test_monitoring_policy_uses_expected_windows(tmp_path: Path):
    service = MLOpsService(tmp_path / "monitoring.db")
    summary = service.monitoring_summary()
    assert summary["monitoring_policy"]["minimum_claims_for_drift"] == 500
    assert summary["monitoring_policy"]["drift_window_size"] == 500
    assert summary["monitoring_policy"]["minimum_labeled_claims_for_performance"] == 100
    assert summary["monitoring_policy"]["automatic_retraining"] is False
    assert "formal_retraining_review_labeled_claims" not in summary["monitoring_policy"]
    assert "formal_retraining_review_fraction" not in summary["monitoring_policy"]
    assert summary["monitoring_policy"]["retraining_policy"].startswith(
        "No volume-percentage trigger"
    )


def test_drift_waits_for_minimum_independent_window(tmp_path: Path):
    service = MLOpsService(tmp_path / "monitoring.db")
    model = FraudDetectionModel()
    frame = pd.read_csv(service.dataset_path).iloc[[0]]
    score = model.predict_probabilities(frame[model.FEATURES]).iloc[0]
    _record_row(service, model, "PROD-ONE", frame.iloc[0], score)
    report = service.drift_report()
    assert report["status"] == "insufficient_data"
    assert report["minimum_required"] == 500
    assert report["remaining"] == 499


def test_drift_engine_evaluates_when_window_is_reached(tmp_path: Path):
    service = MLOpsService(tmp_path / "monitoring.db")
    service.MIN_DRIFT_SAMPLE = 5
    service.DRIFT_WINDOW_SIZE = 5
    model = FraudDetectionModel()
    frame = pd.read_csv(service.dataset_path).iloc[:5].copy()
    scores = model.predict_probabilities(frame[model.FEATURES]).to_numpy()
    for index, (_, row) in enumerate(frame.iterrows()):
        _record_row(service, model, f"PROD-{index}", row, scores[index])
    report = service.drift_report()
    prediction_report = service.prediction_drift()
    assert report["status"] == "evaluated"
    assert len(report["features"]) == 30
    assert prediction_report["status"] == "evaluated"


def test_delayed_ground_truth_enables_production_performance(tmp_path: Path):
    service = MLOpsService(tmp_path / "monitoring.db")
    model = FraudDetectionModel()
    df = pd.read_csv(service.dataset_path)
    selected = pd.concat(
        [
            df[df[model.TARGET] == 1].head(60),
            df[df[model.TARGET] == 0].head(60),
        ],
        ignore_index=True,
    )
    scores = model.predict_probabilities(selected[model.FEATURES]).to_numpy()
    for index, (_, row) in enumerate(selected.iterrows()):
        claim_id = f"LABELED-{index:03d}"
        _record_row(service, model, claim_id, row, scores[index])
        service.set_ground_truth(claim_id, int(row[model.TARGET]))
    performance = service.production_performance()
    assert performance["status"] == "evaluated"
    assert performance["labeled_claims"] == 120
    assert performance["positive_labels"] == 60
    assert performance["negative_labels"] == 60
    assert "roc_auc" in performance
    assert "degradation" in performance


def test_external_analysis_is_logged_and_can_receive_ground_truth(tmp_path: Path, monkeypatch):
    service = MLOpsService(tmp_path / "monitoring.db")
    monkeypatch.setattr(main_module, "mlops_service", service)
    stored = client.get("/claims/KAGGLE-000001").json()
    response = client.post(
        "/claim-analysis",
        json={
            "claim_id": "PROD-API-001",
            "features": stored["features"],
            "documents": [],
        },
    )
    assert response.status_code == 200
    assert service.monitoring_summary()["unique_claims"] == 1

    ground_truth = client.put(
        "/mlops/production-claims/PROD-API-001/ground-truth",
        json={"FraudFound_P": 0},
    )
    assert ground_truth.status_code == 200
    assert service.monitoring_summary()["labeled_claims"] == 1


def test_drift_uses_most_recent_unique_claim_window(tmp_path: Path):
    service = MLOpsService(tmp_path / "monitoring.db")
    service.MIN_DRIFT_SAMPLE = 5
    service.DRIFT_WINDOW_SIZE = 5
    model = FraudDetectionModel()
    df = pd.read_csv(service.dataset_path)
    selected = df.iloc[:6].copy()
    scores = model.predict_probabilities(selected[model.FEATURES]).to_numpy()
    for index, (_, row) in enumerate(selected.iterrows()):
        _record_row(service, model, f"WINDOW-{index}", row, scores[index])
    report = service.drift_report()
    prediction = service.prediction_drift()
    assert report["production_claims_total"] == 6
    assert report["production_claims_in_window"] == 5
    assert prediction["production_claims_total"] == 6
    assert prediction["production_claims_in_window"] == 5


def test_confirmed_performance_degradation_recommends_retraining(tmp_path: Path):
    service = MLOpsService(tmp_path / "monitoring.db")
    model = FraudDetectionModel()
    df = pd.read_csv(service.dataset_path)
    selected = pd.concat(
        [
            df[df[model.TARGET] == 1].head(60),
            df[df[model.TARGET] == 0].head(60),
        ],
        ignore_index=True,
    )
    scores = model.predict_probabilities(selected[model.FEATURES]).to_numpy()
    # Deliberately adverse confirmed labels simulate a materially degraded
    # production relationship between scores and true fraud outcomes.
    adverse_labels = (scores < model.REVIEW_THRESHOLD).astype(int)
    # Ensure both classes satisfy the minimum representation requirement.
    assert int(adverse_labels.sum()) >= service.MIN_CLASS_COUNT_FOR_PERFORMANCE
    assert int((adverse_labels == 0).sum()) >= service.MIN_CLASS_COUNT_FOR_PERFORMANCE
    for index, (_, row) in enumerate(selected.iterrows()):
        claim_id = f"DEGRADED-{index:03d}"
        _record_row(service, model, claim_id, row, scores[index])
        service.set_ground_truth(claim_id, int(adverse_labels[index]))
    performance = service.production_performance()
    status = service.retraining_status()
    assert performance["status"] == "evaluated"
    assert performance["degradation"]["detected"] is True
    assert status["retraining_recommended"] is True
    assert status["automatic_retraining"] is False
