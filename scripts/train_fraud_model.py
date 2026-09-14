from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from backend.ml.fraud_model import FraudDetectionModel


if __name__ == "__main__":
    model = FraudDetectionModel(force_retrain=True)
    print(json.dumps(model.get_model_info(), ensure_ascii=False, indent=2))
