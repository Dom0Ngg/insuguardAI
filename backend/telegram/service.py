from __future__ import annotations

import mimetypes
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import uuid

from backend.services.claim_repository import ClaimRepository
from backend.telegram.client import TelegramBotClient
from backend.telegram.intake_agent import TelegramIntakeAgent
from backend.telegram.repository import TelegramRepository


class TelegramService:
    def __init__(
        self,
        intake_agent: Optional[TelegramIntakeAgent] = None,
        repository: Optional[TelegramRepository] = None,
        client: Optional[TelegramBotClient] = None,
    ):
        self.repository = repository or TelegramRepository()
        self.client = client or TelegramBotClient()
        self.intake = intake_agent or TelegramIntakeAgent(repository=self.repository)

    @staticmethod
    def _external_message_id(chat_id: str, message_id: Any) -> str:
        return f"tg:{chat_id}:{message_id}"

    def _store_outbound(self, chat_id: str, text: str, result: Dict[str, Any]) -> None:
        message_id = result.get("message_id") if isinstance(result, dict) else None
        external_id = (
            self._external_message_id(chat_id, message_id)
            if message_id is not None
            else f"preview-{uuid.uuid4()}"
        )
        self.repository.record_message(
            chat_id=chat_id,
            direction="outbound",
            message_type="text",
            text=text,
            external_message_id=external_id,
            delivery_status="sent" if message_id is not None else result.get("status", "preview"),
        )

    def send_replies(
        self,
        chat_id: str,
        replies: List[str],
        *,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        sent = []
        for index, text in enumerate(replies):
            markup = reply_markup if index == len(replies) - 1 else None
            result = self.client.send_text(chat_id, text, reply_markup=markup)
            self._store_outbound(chat_id, text, result)
            sent.append(result)
        return sent

    def process_simulated_text(self, chat_id: str, text: str) -> Dict[str, Any]:
        chat_id = str(chat_id)
        external_id = f"sim-tg-{uuid.uuid4()}"
        self.repository.record_message(
            chat_id=chat_id,
            direction="inbound",
            message_type="text",
            text=text,
            external_message_id=external_id,
            delivery_status="simulated",
        )
        replies = self.intake.process_text(chat_id, text)
        for reply in replies:
            self.repository.record_message(
                chat_id=chat_id,
                direction="outbound",
                message_type="text",
                text=reply,
                external_message_id=f"sim-tg-out-{uuid.uuid4()}",
                delivery_status="preview",
            )
        return {
            "chat_id": chat_id,
            "input": text,
            "replies": replies,
            "reply_markup": self.intake.reply_markup_for(chat_id),
            "session": self.repository.get_session(chat_id),
        }

    @staticmethod
    def _identity(message: Dict[str, Any]) -> Dict[str, Optional[str]]:
        sender = message.get("from") or {}
        return {
            "user_id": str(sender.get("id")) if sender.get("id") is not None else None,
            "username": sender.get("username"),
            "first_name": sender.get("first_name"),
        }

    def _refresh_identity(self, chat_id: str, message: Dict[str, Any]) -> None:
        self.repository.update_identity(chat_id, **self._identity(message))

    def _process_text_message(
        self, chat_id: str, message: Dict[str, Any], update_id: Optional[str]
    ) -> Dict[str, Any]:
        text = str(message.get("text") or "")
        identity = self._identity(message)
        external_id = self._external_message_id(chat_id, message.get("message_id"))
        self.repository.record_message(
            chat_id=chat_id,
            direction="inbound",
            message_type="text",
            text=text,
            external_message_id=external_id,
            update_id=update_id,
            user_id=identity["user_id"],
            username=identity["username"],
            delivery_status="received",
        )
        replies = self.intake.process_text(chat_id, text)
        self._refresh_identity(chat_id, message)
        sent = self.send_replies(
            chat_id,
            replies,
            reply_markup=self.intake.reply_markup_for(chat_id),
        )
        return {"type": "text", "replies": replies, "sent": sent}

    @staticmethod
    def _document_info(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        document = message.get("document")
        if isinstance(document, dict):
            return {
                "message_type": "document",
                "file_id": document.get("file_id"),
                "file_name": document.get("file_name") or "telegram_document",
                "mime_type": document.get("mime_type"),
                "file_size": document.get("file_size"),
            }
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            photo = photos[-1]
            return {
                "message_type": "image",
                "file_id": photo.get("file_id"),
                "file_name": f"telegram_photo_{message.get('message_id', uuid.uuid4().hex)}.jpg",
                "mime_type": "image/jpeg",
                "file_size": photo.get("file_size"),
            }
        return None

    def _process_media_message(
        self, chat_id: str, message: Dict[str, Any], update_id: Optional[str]
    ) -> Dict[str, Any]:
        info = self._document_info(message)
        if not info or not info.get("file_id"):
            return {"status": "ignored", "reason": "missing Telegram file_id"}

        original_name = Path(str(info["file_name"])).name
        suffix = Path(original_name).suffix.lower()
        if not suffix and info.get("mime_type"):
            suffix = mimetypes.guess_extension(str(info["mime_type"])) or ""
            original_name += suffix

        supported = ClaimRepository.get_supported_document_extensions()
        if suffix not in supported:
            reply = f"Formato de adjunto no soportado. Usa: {', '.join(supported)}."
            self.send_replies(chat_id, [reply])
            return {"status": "unsupported", "file_name": original_name, "reply": reply}

        file_size = info.get("file_size")
        if file_size and int(file_size) > 20 * 1024 * 1024:
            reply = "El archivo supera 20 MB, límite de descarga del Bot API estándar de Telegram."
            self.send_replies(chat_id, [reply])
            return {"status": "too_large", "file_name": original_name, "reply": reply}

        identity = self._identity(message)
        external_id = self._external_message_id(chat_id, message.get("message_id"))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / original_name
            metadata = self.client.download_file(str(info["file_id"]), temp_path)
            replies = self.intake.register_document(chat_id, temp_path, original_name)
            self._refresh_identity(chat_id, message)
            session = self.repository.get_session(chat_id)
            local_path = None
            if session:
                local_path = str(
                    self.intake.claim_repository.documents_dir(session["claim_id"]) / original_name
                )
            self.repository.record_message(
                chat_id=chat_id,
                direction="inbound",
                message_type=str(info["message_type"]),
                external_message_id=external_id,
                update_id=update_id,
                user_id=identity["user_id"],
                username=identity["username"],
                media_id=str(info["file_id"]),
                file_name=original_name,
                local_path=local_path,
                delivery_status="downloaded",
            )
            sent = self.send_replies(
                chat_id,
                replies,
                reply_markup=self.intake.reply_markup_for(chat_id),
            )
        return {"type": info["message_type"], "metadata": metadata, "replies": replies, "sent": sent}

    def _process_callback_query(
        self, callback: Dict[str, Any], update_id: Optional[str]
    ) -> Dict[str, Any]:
        callback_id = str(callback.get("id") or "")
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or "").strip()
        data = str(callback.get("data") or "")
        sender = callback.get("from") or {}
        if not callback_id or not chat_id or not data:
            return {"status": "ignored", "reason": "invalid callback_query"}

        # Telegram shows a spinner after an inline button is pressed until the
        # callback is acknowledged. Acknowledge first, then process normally.
        try:
            self.client.answer_callback_query(callback_id)
        except Exception:
            # Do not lose a valid claim answer just because the acknowledgement fails.
            pass

        if data.startswith("answer:"):
            text = data[len("answer:"):]
        elif data.startswith("cmd:"):
            text = data[len("cmd:"):]
        else:
            return {"status": "ignored", "reason": "unknown callback data"}

        external_id = f"tgcb:{callback_id}"
        if self.repository.message_seen(external_id):
            return {"status": "duplicate_ignored", "message_id": external_id}

        self.repository.record_message(
            chat_id=chat_id,
            direction="inbound",
            message_type="callback",
            text=text,
            external_message_id=external_id,
            update_id=update_id,
            user_id=str(sender.get("id")) if sender.get("id") is not None else None,
            username=sender.get("username"),
            delivery_status="received",
        )
        replies = self.intake.process_text(chat_id, text)
        self.repository.update_identity(
            chat_id,
            user_id=str(sender.get("id")) if sender.get("id") is not None else None,
            username=sender.get("username"),
            first_name=sender.get("first_name"),
        )
        sent = self.send_replies(
            chat_id,
            replies,
            reply_markup=self.intake.reply_markup_for(chat_id),
        )
        return {"type": "callback", "input": text, "replies": replies, "sent": sent}

    def process_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        update_id_value = update.get("update_id")
        update_id = str(update_id_value) if update_id_value is not None else None
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            result = self._process_callback_query(callback, update_id)
            return {
                "status": "ok" if result.get("status") not in {"ignored", "duplicate_ignored"} else result.get("status"),
                "update_id": update_id,
                "result": result,
            }

        message = update.get("message")
        if not isinstance(message, dict):
            return {"status": "ignored", "reason": "update does not contain a message or callback_query"}

        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or "").strip()
        message_id = message.get("message_id")
        if not chat_id or message_id is None:
            return {"status": "ignored", "reason": "message has no chat_id/message_id"}

        external_id = self._external_message_id(chat_id, message_id)
        if self.repository.message_seen(external_id):
            return {"status": "duplicate_ignored", "message_id": external_id}

        if "text" in message:
            result = self._process_text_message(chat_id, message, update_id)
        elif "document" in message or "photo" in message:
            result = self._process_media_message(chat_id, message, update_id)
        else:
            identity = self._identity(message)
            self.repository.record_message(
                chat_id=chat_id,
                direction="inbound",
                message_type="unsupported",
                external_message_id=external_id,
                update_id=update_id,
                user_id=identity["user_id"],
                username=identity["username"],
                delivery_status="unsupported",
            )
            replies = ["Formato no soportado. Envía texto, PDF o imagen."]
            result = {
                "type": "unsupported",
                "replies": replies,
                "sent": self.send_replies(
                    chat_id,
                    replies,
                    reply_markup=self.intake.reply_markup_for(chat_id),
                ),
            }

        return {
            "status": "ok",
            "update_id": update_id,
            "chat_id": chat_id,
            "result": result,
        }
