from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.fraud_model import FraudDetectionModel


if __name__ == "__main__":
    model = FraudDetectionModel()
    metrics = model.get_model_info()["metrics"]
    holdout = metrics["holdout"]
    expected = metrics["benchmark_reproduction"]
    decision = holdout["at_decision_threshold"]

    checks = {
        "PR-AUC": abs(holdout["pr_auc"] - expected["expected_pr_auc"]) < 1e-6,
        "ROC-AUC": abs(holdout["roc_auc"] - expected["expected_roc_auc"]) < 1e-6,
        "Recall": abs(decision["recall"] - expected["expected_recall"]) < 1e-4,
        "Precision": abs(decision["precision"] - expected["expected_precision"]) < 1e-4,
        "Threshold": abs(decision["threshold"] - expected["expected_threshold"]) < 1e-6,
        "Confusion matrix": decision["confusion_matrix"] == expected["expected_confusion_matrix"],
    }

    print("InsurGuard benchmark reproduction")
    print("-" * 40)
    print(f"PR-AUC:    {holdout['pr_auc']}")
    print(f"ROC-AUC:   {holdout['roc_auc']}")
    print(f"Threshold: {decision['threshold']}")
    print(f"Recall:    {decision['recall']}")
    print(f"Precision: {decision['precision']}")
    print(f"F1:        {decision['f1_score']}")
    print(f"CM:        {decision['confusion_matrix']}")
    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}")

    if not all(checks.values()):
        raise SystemExit(1)
