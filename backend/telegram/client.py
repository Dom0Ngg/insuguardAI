from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from backend.telegram.settings import TelegramSettings


class TelegramBotClient:
    """Small synchronous client for the official Telegram Bot API."""

    def __init__(self, settings: Optional[TelegramSettings] = None):
        self.settings = settings or TelegramSettings.from_env()

    @property
    def enabled(self) -> bool:
        return self.settings.configured

    def _method_url(self, method: str) -> str:
        if not self.settings.bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
        return f"{self.settings.api_base_url}/bot{self.settings.bot_token}/{method}"

    def _file_url(self, file_path: str) -> str:
        if not self.settings.bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
        return f"{self.settings.api_base_url}/file/bot{self.settings.bot_token}/{file_path}"

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            raise RuntimeError(payload.get("description") or "Telegram Bot API request failed")
        return payload.get("result")

    def get_me(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "not_configured"}
        with httpx.Client(timeout=20) as client:
            result = self._unwrap(client.get(self._method_url("getMe")))
        return result or {}

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        if not self.enabled:
            return False
        with httpx.Client(timeout=20) as client:
            result = self._unwrap(
                client.post(
                    self._method_url("deleteWebhook"),
                    json={"drop_pending_updates": bool(drop_pending_updates)},
                )
            )
        return bool(result)

    def get_updates(
        self,
        *,
        offset: Optional[int] = None,
        timeout: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        payload: Dict[str, Any] = {
            "timeout": int(timeout if timeout is not None else self.settings.polling_timeout),
            "limit": min(max(int(limit), 1), 100),
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = int(offset)
        request_timeout = max(35, payload["timeout"] + 10)
        with httpx.Client(timeout=request_timeout) as client:
            result = self._unwrap(client.post(self._method_url("getUpdates"), json=payload))
        return list(result or [])

    def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "status": "preview",
                "chat_id": str(chat_id),
                "text": text,
                "reply_markup": reply_markup,
                "reason": "Telegram credentials are not configured; message stored for admin demo only.",
            }
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]
        last: Dict[str, Any] = {}
        with httpx.Client(timeout=30) as client:
            for index, chunk in enumerate(chunks):
                payload: Dict[str, Any] = {"chat_id": chat_id, "text": chunk}
                if reply_markup is not None and index == len(chunks) - 1:
                    payload["reply_markup"] = reply_markup
                last = self._unwrap(
                    client.post(self._method_url("sendMessage"), json=payload)
                ) or {}
        return last


    def answer_callback_query(
        self,
        callback_query_id: str,
        *,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> bool:
        """Acknowledge a Telegram inline-keyboard callback."""
        if not self.enabled:
            return False
        payload: Dict[str, Any] = {
            "callback_query_id": str(callback_query_id),
            "show_alert": bool(show_alert),
        }
        if text:
            payload["text"] = str(text)[:200]
        with httpx.Client(timeout=20) as client:
            result = self._unwrap(
                client.post(self._method_url("answerCallbackQuery"), json=payload)
            )
        return bool(result)

    def set_my_commands(self) -> bool:
        """Expose the main InsurGuard actions in Telegram's slash-command menu."""
        if not self.enabled:
            return False
        commands = [
            {"command": "nuevo", "description": "Iniciar un nuevo siniestro"},
            {"command": "estado", "description": "Ver progreso del claim"},
            {"command": "analizar", "description": "Ejecutar el análisis InsurGuard"},
            {"command": "cancelar", "description": "Cancelar la captura actual"},
            {"command": "ayuda", "description": "Mostrar ayuda y opciones"},
        ]
        with httpx.Client(timeout=20) as client:
            result = self._unwrap(
                client.post(self._method_url("setMyCommands"), json={"commands": commands})
            )
        return bool(result)

    def get_file(self, file_id: str) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required to download Telegram media")
        with httpx.Client(timeout=30) as client:
            result = self._unwrap(client.get(self._method_url("getFile"), params={"file_id": file_id}))
        return result or {}

    def download_file(self, file_id: str, destination: Path) -> Dict[str, Any]:
        metadata = self.get_file(file_id)
        file_path = metadata.get("file_path")
        if not file_path:
            raise RuntimeError("Telegram getFile did not return file_path")
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(self._file_url(file_path))
            response.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)
        return {
            "path": str(destination),
            "file_path": file_path,
            "file_size": metadata.get("file_size"),
            "file_unique_id": metadata.get("file_unique_id"),
        }
