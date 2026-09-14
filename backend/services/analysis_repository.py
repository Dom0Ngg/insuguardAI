from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import uuid


class AnalysisRepository:
    """Audit store for end-to-end analyses shown by the admin dashboard."""

    def __init__(self, database_path: Optional[Path] = None):
        project_root = Path(__file__).resolve().parents[2]
        self.database_path = Path(database_path) if database_path else (
            project_root / "artifacts" / "operations" / "analysis.db"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
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

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    source_channel TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_claim
                ON analysis_runs(claim_id, completed_at);
                CREATE INDEX IF NOT EXISTS idx_analysis_completed
                ON analysis_runs(completed_at);
                """
            )

    def record_run(
        self,
        *,
        claim_id: str,
        source_channel: str,
        started_at: str,
        completed_at: str,
        duration_ms: float,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO analysis_runs (
                    run_id, claim_id, source_channel, started_at, completed_at,
                    duration_ms, status, result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    claim_id,
                    source_channel,
                    started_at,
                    completed_at,
                    float(duration_ms),
                    status,
                    json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
                    error,
                ),
            )
        return self.get_run(run_id) or {"run_id": run_id}

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        raw = data.pop("result_json", None)
        data["result"] = json.loads(raw) if raw else None
        return data

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._decode(row)

    def list_runs(self, limit: int = 100, claim_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            if claim_id:
                rows = connection.execute(
                    "SELECT * FROM analysis_runs WHERE claim_id = ? ORDER BY completed_at DESC LIMIT ?",
                    (claim_id, int(limit)),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM analysis_runs ORDER BY completed_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        return [self._decode(row) for row in rows if row is not None]

    def latest_for_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        runs = self.list_runs(limit=1, claim_id=claim_id)
        return runs[0] if runs else None

    def summary(self) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total_runs,
                       COUNT(DISTINCT claim_id) AS claims_analyzed,
                       SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS successful_runs,
                       SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS failed_runs,
                       AVG(duration_ms) AS average_duration_ms
                FROM analysis_runs
                """
            ).fetchone()
        data = dict(row)
        data["average_duration_ms"] = round(float(data["average_duration_ms"] or 0.0), 2)
        for key in ("total_runs", "claims_analyzed", "successful_runs", "failed_runs"):
            data[key] = int(data[key] or 0)
        return data
