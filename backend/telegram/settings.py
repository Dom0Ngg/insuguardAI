from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}


@dataclass(frozen=True)
class TelegramSettings:
    """Runtime configuration for the Telegram Bot API.

    With no bot token, InsurGuard stays fully usable through the local admin
    simulator. When a token is configured, long polling is enabled by default,
    so no public webhook, ngrok tunnel or business phone number is required.
    """

    bot_token: str
    api_base_url: str
    polling_enabled: bool
    polling_timeout: int
    drop_pending_updates: bool

    @classmethod
    def from_env(cls) -> "TelegramSettings":
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            api_base_url=os.getenv("TELEGRAM_API_BASE_URL", "https://api.telegram.org").rstrip("/"),
            polling_enabled=_env_bool("TELEGRAM_POLLING_ENABLED", True),
            polling_timeout=max(1, min(50, int(os.getenv("TELEGRAM_POLL_TIMEOUT", "25")))),
            drop_pending_updates=_env_bool("TELEGRAM_DROP_PENDING_UPDATES", False),
        )

    @property
    def configured(self) -> bool:
        return bool(self.bot_token)

    @property
    def live_enabled(self) -> bool:
        return self.configured and self.polling_enabled

    def public_status(self) -> dict:
        return {
            "configured": self.configured,
            "outbound_enabled": self.configured,
            "polling_enabled": self.polling_enabled,
            "live_enabled": self.live_enabled,
            "poll_timeout_seconds": self.polling_timeout,
            "mode": "telegram_long_polling" if self.live_enabled else "local_demo",
            "requires_public_webhook": False,
        }
