from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.main import app



if __name__ == "__main__":
    expected = [ROOT / "data" / name for name in ("kaggle", "policies", "claims")]
    print("Data folders:")
    for path in expected:
        print(f"  {'OK' if path.exists() else 'MISSING'} {path.relative_to(ROOT)}")

    client = TestClient(app)
    model_info = client.get("/mlops/model-info").json()
    print("Model:", model_info.get("model_type"), model_info.get("model_version"))
    print("Decision threshold:", model_info.get("operational_thresholds", {}).get("decision_threshold"))
    for endpoint in ["/health", "/telegram/info", "/dataset/info", "/claims", "/policies", "/mlops/model-metrics", "/mlops/monitoring-summary", "/mlops/drift-report", "/mlops/retraining-status"]:
        response = client.get(endpoint)
        print(f"{response.status_code} {endpoint}")

    claims = client.get("/claims").json().get("claims", [])
    if claims:
        claim_id = claims[0]["claim_id"]
        response = client.get(f"/claims/{claim_id}/analysis")
        print(f"{response.status_code} /claims/{claim_id}/analysis")
        if response.status_code == 200:
            body = response.json()
            print("  fraud_score:", body.get("fraud_score"))
            print("  risk_level:", body.get("risk_level"))
    print("Project check completed.")
