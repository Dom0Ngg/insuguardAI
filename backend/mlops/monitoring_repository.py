import json
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class MonitoringRepository:
    """Persistent prediction log for production-style FastAPI requests.

    Prediction events are append-only. Ground truth is stored separately by
    claim_id so labels can arrive later without rewriting the prediction log.
    """

    def __init__(self, database_path: Optional[Path] = None):
        project_root = Path(__file__).resolve().parents[2]
        self.database_path = Path(database_path) if database_path is not None else (
            project_root / "artifacts" / "monitoring" / "monitoring.db"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open, commit/rollback and always close a SQLite connection."""
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _json_default(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS prediction_events (
                    event_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    fraud_score REAL NOT NULL,
                    risk_level TEXT,
                    manual_review_required INTEGER NOT NULL,
                    features_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_prediction_claim_id
                ON prediction_events(claim_id);

                CREATE INDEX IF NOT EXISTS idx_prediction_observed_at
                ON prediction_events(observed_at);

                CREATE TABLE IF NOT EXISTS ground_truth (
                    claim_id TEXT PRIMARY KEY,
                    fraud_label INTEGER NOT NULL CHECK (fraud_label IN (0, 1)),
                    updated_at TEXT NOT NULL
                );
                """
            )

    def record_prediction(
        self,
        *,
        claim_id: str,
        features: Dict[str, Any],
        fraud_score: float,
        risk_level: Optional[str],
        manual_review_required: bool,
        model_version: str,
    ) -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        observed_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO prediction_events (
                    event_id, claim_id, observed_at, model_version, fraud_score,
                    risk_level, manual_review_required, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    claim_id,
                    observed_at,
                    model_version,
                    float(fraud_score),
                    risk_level,
                    int(bool(manual_review_required)),
                    json.dumps(features, ensure_ascii=False, sort_keys=True, default=self._json_default),
                ),
            )
        return {
            "event_id": event_id,
            "claim_id": claim_id,
            "observed_at": observed_at,
        }

    def set_ground_truth(self, claim_id: str, fraud_label: int) -> Dict[str, Any]:
        if fraud_label not in (0, 1):
            raise ValueError("fraud_label must be 0 or 1")
        if not self.has_claim(claim_id):
            raise KeyError(claim_id)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO ground_truth (claim_id, fraud_label, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    fraud_label=excluded.fraud_label,
                    updated_at=excluded.updated_at
                """,
                (claim_id, int(fraud_label), updated_at),
            )
        return {
            "claim_id": claim_id,
            "FraudFound_P": int(fraud_label),
            "updated_at": updated_at,
        }

    def has_claim(self, claim_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM prediction_events WHERE claim_id = ? LIMIT 1", (claim_id,)
            ).fetchone()
        return row is not None

    def summary(self) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS event_count,
                    COUNT(DISTINCT claim_id) AS unique_claims,
                    MIN(observed_at) AS first_observed_at,
                    MAX(observed_at) AS last_observed_at
                FROM prediction_events
                """
            ).fetchone()
            labeled = connection.execute(
                """
                SELECT COUNT(*) AS labeled_claims
                FROM ground_truth g
                WHERE EXISTS (
                    SELECT 1 FROM prediction_events p WHERE p.claim_id = g.claim_id
                )
                """
            ).fetchone()
        return {
            "event_count": int(row["event_count"] or 0),
            "unique_claims": int(row["unique_claims"] or 0),
            "labeled_claims": int(labeled["labeled_claims"] or 0),
            "first_observed_at": row["first_observed_at"],
            "last_observed_at": row["last_observed_at"],
            "database": "artifacts/monitoring/monitoring.db",
        }

    def latest_predictions(self) -> List[Dict[str, Any]]:
        """Return the latest prediction event per claim_id."""
        query = """
            SELECT p.*, g.fraud_label, g.updated_at AS ground_truth_updated_at
            FROM prediction_events p
            JOIN (
                SELECT claim_id, MAX(rowid) AS max_rowid
                FROM prediction_events
                GROUP BY claim_id
            ) latest ON latest.max_rowid = p.rowid
            LEFT JOIN ground_truth g ON g.claim_id = p.claim_id
            ORDER BY p.rowid
        """
        with self._connection() as connection:
            rows = connection.execute(query).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "event_id": row["event_id"],
                    "claim_id": row["claim_id"],
                    "observed_at": row["observed_at"],
                    "model_version": row["model_version"],
                    "fraud_score": float(row["fraud_score"]),
                    "risk_level": row["risk_level"],
                    "manual_review_required": bool(row["manual_review_required"]),
                    "features": json.loads(row["features_json"]),
                    "FraudFound_P": (
                        None if row["fraud_label"] is None else int(row["fraud_label"])
                    ),
                    "ground_truth_updated_at": row["ground_truth_updated_at"],
                }
            )
        return result

    def clear(self) -> None:
        """Testing/maintenance helper; not exposed by the public API."""
        with self._connection() as connection:
            connection.execute("DELETE FROM ground_truth")
            connection.execute("DELETE FROM prediction_events")
