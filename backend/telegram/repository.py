from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelegramRepository:
    """Persistent operational log for Telegram sessions and messages.

    PostgreSQL remains dedicated to the pgvector semantic store. Telegram
    conversational state is intentionally kept in a small SQLite database so
    it is easy to inspect during the TFM demonstration.
    """

    def __init__(self, database_path: Optional[Path] = None):
        project_root = Path(__file__).resolve().parents[2]
        self.database_path = Path(database_path) if database_path else (
            project_root / "artifacts" / "telegram" / "telegram.db"
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
                CREATE TABLE IF NOT EXISTS telegram_sessions (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    username TEXT,
                    first_name TEXT,
                    claim_id TEXT,
                    status TEXT NOT NULL,
                    current_step INTEGER NOT NULL DEFAULT 0,
                    data_json TEXT NOT NULL DEFAULT '{}',
                    policy_id TEXT,
                    coverage_query TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    last_error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tg_sessions_claim
                ON telegram_sessions(claim_id);

                CREATE INDEX IF NOT EXISTS idx_tg_sessions_updated
                ON telegram_sessions(updated_at);

                CREATE TABLE IF NOT EXISTS telegram_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_message_id TEXT UNIQUE,
                    update_id TEXT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    direction TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    text TEXT,
                    media_id TEXT,
                    file_name TEXT,
                    local_path TEXT,
                    delivery_status TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tg_messages_chat
                ON telegram_messages(chat_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_tg_messages_update
                ON telegram_messages(update_id);
                """
            )

    @staticmethod
    def _decode_session(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        try:
            result["data"] = json.loads(result.pop("data_json") or "{}")
        except json.JSONDecodeError:
            result["data"] = {}
            result.pop("data_json", None)
        return result

    def get_session(self, chat_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM telegram_sessions WHERE chat_id = ?", (str(chat_id),)
            ).fetchone()
        return self._decode_session(row) if row else None

    def get_session_by_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM telegram_sessions WHERE claim_id = ? ORDER BY updated_at DESC LIMIT 1",
                (claim_id,),
            ).fetchone()
        return self._decode_session(row) if row else None

    def save_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        now = _utcnow()
        created_at = session.get("created_at") or now
        updated_at = now
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO telegram_sessions (
                    chat_id, user_id, username, first_name, claim_id, status,
                    current_step, data_json, policy_id, coverage_query,
                    created_at, updated_at, completed_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    username=excluded.username,
                    first_name=excluded.first_name,
                    claim_id=excluded.claim_id,
                    status=excluded.status,
                    current_step=excluded.current_step,
                    data_json=excluded.data_json,
                    policy_id=excluded.policy_id,
                    coverage_query=excluded.coverage_query,
                    updated_at=excluded.updated_at,
                    completed_at=excluded.completed_at,
                    last_error=excluded.last_error
                """,
                (
                    str(session["chat_id"]),
                    str(session.get("user_id") or "") or None,
                    session.get("username"),
                    session.get("first_name"),
                    session.get("claim_id"),
                    session.get("status", "collecting"),
                    int(session.get("current_step", 0)),
                    json.dumps(session.get("data", {}), ensure_ascii=False, sort_keys=True),
                    session.get("policy_id"),
                    session.get("coverage_query"),
                    created_at,
                    updated_at,
                    session.get("completed_at"),
                    session.get("last_error"),
                ),
            )
        return self.get_session(str(session["chat_id"])) or session

    def reset_session(
        self,
        chat_id: str,
        claim_id: str,
        *,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = _utcnow()
        previous = self.get_session(str(chat_id)) or {}
        session = {
            "chat_id": str(chat_id),
            "user_id": str(user_id or previous.get("user_id") or "") or None,
            "username": username if username is not None else previous.get("username"),
            "first_name": first_name if first_name is not None else previous.get("first_name"),
            "claim_id": claim_id,
            "status": "collecting",
            "current_step": 0,
            "data": {"features": {}, "documents": []},
            "policy_id": None,
            "coverage_query": None,
            "created_at": now,
            "completed_at": None,
            "last_error": None,
        }
        return self.save_session(session)

    def update_identity(
        self,
        chat_id: str,
        *,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        session = self.get_session(str(chat_id))
        if not session:
            return None
        if user_id is not None:
            session["user_id"] = str(user_id)
        if username is not None:
            session["username"] = username
        if first_name is not None:
            session["first_name"] = first_name
        return self.save_session(session)

    def record_message(
        self,
        *,
        chat_id: str,
        direction: str,
        message_type: str,
        text: Optional[str] = None,
        external_message_id: Optional[str] = None,
        update_id: Optional[str] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        media_id: Optional[str] = None,
        file_name: Optional[str] = None,
        local_path: Optional[str] = None,
        delivery_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        created_at = _utcnow()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO telegram_messages (
                    external_message_id, update_id, chat_id, user_id, username,
                    direction, message_type, text, media_id, file_name,
                    local_path, delivery_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    external_message_id,
                    str(update_id) if update_id is not None else None,
                    str(chat_id),
                    str(user_id) if user_id is not None else None,
                    username,
                    direction,
                    message_type,
                    text,
                    media_id,
                    file_name,
                    local_path,
                    delivery_status,
                    created_at,
                ),
            )
            if external_message_id:
                row = connection.execute(
                    "SELECT * FROM telegram_messages WHERE external_message_id = ?",
                    (external_message_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM telegram_messages ORDER BY id DESC LIMIT 1"
                ).fetchone()
        return dict(row) if row else {}

    def message_seen(self, external_message_id: str) -> bool:
        if not external_message_id:
            return False
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM telegram_messages WHERE external_message_id = ? LIMIT 1",
                (external_message_id,),
            ).fetchone()
        return row is not None

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM telegram_sessions ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._decode_session(row) for row in rows]

    def list_messages(self, chat_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            if chat_id is not None:
                rows = connection.execute(
                    "SELECT * FROM telegram_messages WHERE chat_id = ? ORDER BY id ASC LIMIT ?",
                    (str(chat_id), int(limit)),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM telegram_messages ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> Dict[str, Any]:
        with self._connection() as connection:
            sessions = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status='collecting' THEN 1 ELSE 0 END) AS collecting,
                       SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                       SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled
                FROM telegram_sessions
                """
            ).fetchone()
            messages = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN direction='inbound' THEN 1 ELSE 0 END) AS inbound,
                       SUM(CASE WHEN direction='outbound' THEN 1 ELSE 0 END) AS outbound
                FROM telegram_messages
                """
            ).fetchone()
        return {
            "sessions": {key: int(value or 0) for key, value in dict(sessions).items()},
            "messages": {key: int(value or 0) for key, value in dict(messages).items()},
        }
