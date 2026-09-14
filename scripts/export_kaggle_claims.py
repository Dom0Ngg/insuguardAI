import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from backend.ml.fraud_model import FraudDetectionModel


DATASET = ROOT / "data" / "kaggle" / "fraud_oracle.csv"
CLAIMS_DIR = ROOT / "data" / "claims"


def export_claim(row_index: int, row: pd.Series) -> str:
    claim_id = f"KAGGLE-{row_index + 1:06d}"
    folder = CLAIMS_DIR / claim_id
    folder.mkdir(parents=True, exist_ok=True)
    features = {name: row[name] for name in FraudDetectionModel.FEATURES}
    # Convert numpy scalars to ordinary Python values through JSON round-trip helpers.
    features = {key: (value.item() if hasattr(value, "item") else value) for key, value in features.items()}
    payload = {
        "claim_id": claim_id,
        "source": {
            "dataset": "Vehicle Insurance Claim Fraud Detection (Kaggle)",
            "file": "data/kaggle/fraud_oracle.csv",
            "row_index": int(row_index),
            "PolicyNumber": int(row["PolicyNumber"]),
            "Year": int(row["Year"]),
        },
        "features": features,
        "ground_truth": {"FraudFound_P": int(row[FraudDetectionModel.TARGET])},
        "policy_id": None,
        "coverage_query": None,
        "documents": [],
        "notes": "No claim documents or contractual policy are supplied by Kaggle for this row.",
    }
    (folder / "claim.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return claim_id


def main():
    parser = argparse.ArgumentParser(description="Export real Kaggle rows as InsurGuard demo claims.")
    parser.add_argument("--fraud", type=int, default=5, help="Number of fraud rows to export")
    parser.add_argument("--non-fraud", type=int, default=5, help="Number of non-fraud rows to export")
    parser.add_argument("--clean", action="store_true", help="Delete current exported claims first")
    args = parser.parse_args()

    if args.clean and CLAIMS_DIR.exists():
        shutil.rmtree(CLAIMS_DIR)
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)
    selected = pd.concat([
        df[df[FraudDetectionModel.TARGET] == 1].head(args.fraud),
        df[df[FraudDetectionModel.TARGET] == 0].head(args.non_fraud),
    ]).sort_index()
    ids = [export_claim(index, row) for index, row in selected.iterrows()]
    print(f"Exported {len(ids)} claims to data/claims")
    for claim_id in ids:
        print(claim_id)


if __name__ == "__main__":
    main()
