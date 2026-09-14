from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from backend.telegram.client import TelegramBotClient
from backend.telegram.service import TelegramService
from backend.telegram.settings import TelegramSettings


logger = logging.getLogger(__name__)


class TelegramPollingRunner:
    """Background long-polling runner for Telegram updates.

    It deliberately uses getUpdates rather than a webhook so the local Docker
    project can receive real Telegram messages without ngrok or a public URL.
    """

    def __init__(
        self,
        service: TelegramService,
        client: Optional[TelegramBotClient] = None,
        settings: Optional[TelegramSettings] = None,
    ):
        self.service = service
        self.settings = settings or TelegramSettings.from_env()
        self.client = client or service.client
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_update_id: Optional[int] = None
        self.processed_updates = 0
        self.last_error: Optional[str] = None
        self.bot_identity: Dict[str, Any] = {}

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        if not self.settings.live_enabled or self.running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telegram-polling", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "processed_updates": self.processed_updates,
            "last_update_id": self.last_update_id,
            "last_error": self.last_error,
            "bot": {
                "id": self.bot_identity.get("id"),
                "username": self.bot_identity.get("username"),
                "first_name": self.bot_identity.get("first_name"),
            }
            if self.bot_identity
            else None,
        }

    def _run(self) -> None:
        offset: Optional[int] = None
        try:
            # getUpdates and webhooks are mutually exclusive in Telegram. Remove
            # any old webhook before polling; pending updates are kept by default.
            self.client.delete_webhook(drop_pending_updates=self.settings.drop_pending_updates)
            self.bot_identity = self.client.get_me()
            self.client.set_my_commands()
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__
            logger.warning("Telegram startup failed: %s", type(exc).__name__)
            return

        while not self._stop.is_set():
            try:
                updates = self.client.get_updates(
                    offset=offset,
                    timeout=self.settings.polling_timeout,
                )
                for update in updates:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        self.last_update_id = update_id
                        offset = update_id + 1
                    try:
                        self.service.process_update(update)
                    except Exception as exc:
                        # Keep the poller alive if one claim/document fails.
                        self.last_error = f"process_update:{type(exc).__name__}"
                        logger.exception("Telegram update processing failed")
                    finally:
                        self.processed_updates += 1
                if updates:
                    self.last_error = None
            except Exception as exc:
                self.last_error = type(exc).__name__
                logger.warning("Telegram polling request failed: %s", type(exc).__name__)
                # Avoid a tight failure loop if the token/network is invalid.
                self._stop.wait(3.0)
