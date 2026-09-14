from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.telegram.client import TelegramBotClient
from backend.telegram.settings import TelegramSettings


if __name__ == "__main__":
    settings = TelegramSettings.from_env()
    print("Telegram configured:", settings.configured)
    print("Polling enabled:", settings.polling_enabled)
    print("Public webhook required:", False)
    if not settings.configured:
        print("Missing TELEGRAM_BOT_TOKEN. Configure it in .env and recreate insurguard-api.")
        raise SystemExit(1)

    client = TelegramBotClient(settings)
    me = client.get_me()
    print("Bot ID:", me.get("id"))
    print("Bot username:", f"@{me.get('username')}" if me.get("username") else "N/D")
    print("Bot name:", me.get("first_name"))
    print("OK: Telegram Bot API token is valid.")
