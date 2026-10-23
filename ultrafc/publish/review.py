"""Fila de revisão no Telegram: rascunho com botões → canal, só com aprovação de um operador.

Uma checagem pode vir com um card PNG, enviado como foto antes do texto (no rascunho e no canal).
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import time
from html import escape
from pathlib import Path

log = logging.getLogger(__name__)

PUBLISH, DISCARD = "pub", "drop"
IN_REVIEW, PUBLISHED, DISCARDED, IGNORED = "revisao", "publicado", "descartado", "ignorado"


def short_key(item_id: str) -> str:
    """O callback_data de um botão aceita só 64 bytes; os ids das checagens passam disso."""
    return hashlib.sha1(item_id.encode()).hexdigest()[:12]


def keyboard(key: str):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup([[InlineKeyboardButton("✅ Publicar", callback_data=f"{PUBLISH}:{key}"),
                                  InlineKeyboardButton("❌ Descartar", callback_data=f"{DISCARD}:{key}")]])


def _options() -> dict:
    from telegram import LinkPreviewOptions
    from telegram.constants import ParseMode

    return {"parse_mode": ParseMode.HTML, "link_preview_options": LinkPreviewOptions(is_disabled=True)}


class Reviewer:
    """Estado persistido em JSON: cada item é enviado para revisão uma vez e publicado no máximo uma vez."""

    def __init__(self, bot, state_path: Path, review_chat: int | str, channel: int | str, operators: set[int]):
        self.bot = bot
        self.state_path = Path(state_path)
        self.review_chat = review_chat
        self.channel = channel
        self.operators = operators
        self.state: dict[str, dict] = self._load()
        self._locks: dict[str, asyncio.Lock] = collections.defaultdict(asyncio.Lock)

    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(f".{self.state_path.name}.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def known(self, item_id: str) -> bool:
        return item_id in self.state

    def ignore(self, item_id: str, reason: str) -> None:
        self.state[item_id] = {"status": IGNORED, "motivo": reason}
        self._save()

    async def submit(self, item_id: str, kind: str, html: str, card: Path | None = None) -> None:
        if item_id in self.state:
            return
        key = short_key(item_id)
        entry: dict = {"tipo": kind, "status": IN_REVIEW, "key": key, "texto": html}
        if card is not None:
            photo = await self.bot.send_photo(chat_id=self.review_chat, photo=card.read_bytes())
            entry |= {"card": str(card), "card_msg_id": photo.message_id}
        msg = await self.bot.send_message(chat_id=self.review_chat, text=html, reply_markup=keyboard(key),
                                          **_options())
        self.state[item_id] = entry | {"review_msg_id": msg.message_id}
        self._save()
        log.info("📨 enviado para revisão: %s", item_id)

    async def _publish_card(self, item_id: str, entry: dict) -> None:
        """Foto antes do texto; marcada no estado para que uma nova tentativa não a repita."""
        if "card" not in entry or entry.get("card_publicado"):
            return
        try:
            photo = Path(entry["card"]).read_bytes()
        except OSError as exc:
            log.warning("card de %s indisponível (%s); publicando só o texto", item_id, exc)
            return
        await self.bot.send_photo(chat_id=self.channel, photo=photo)
        entry["card_publicado"] = True
        self._save()

    def _find(self, key: str) -> str | None:
        return next((i for i, e in self.state.items() if e.get("key") == key), None)

    async def decide(self, action: str, key: str, user_id: int, user_name: str) -> str:
        """Trata um clique; devolve o aviso curto mostrado a quem clicou."""
        if user_id not in self.operators:
            log.warning("clique ignorado de usuário não autorizado: %s (%s)", user_name, user_id)
            return "Você não está autorizado a aprovar publicações."
        item_id = self._find(key)
        if item_id is None:
            return "Item desconhecido."
        async with self._locks[item_id]:
            entry = self.state[item_id]
            if entry["status"] != IN_REVIEW:
                return f"Já {entry['status']} por {entry.get('por', '?')}."
            if action == PUBLISH:
                try:
                    await self._publish_card(item_id, entry)
                    await self.bot.send_message(chat_id=self.channel, text=entry["texto"], **_options())
                except Exception as exc:  # noqa: BLE001 - o item continua em revisão para nova tentativa
                    log.warning("falha ao publicar %s: %s", item_id, exc)
                    return f"Falha ao publicar: {exc}"
                entry["status"], mark = PUBLISHED, "✅ Publicado"
            else:
                entry["status"], mark = DISCARDED, "❌ Descartado"
            entry["por"], entry["em"] = user_name, time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()
            log.info("%s: %s (%s)", mark, item_id, user_name)
            try:
                await self.bot.edit_message_text(chat_id=self.review_chat, message_id=entry["review_msg_id"],
                                                 text=f"{entry['texto']}\n\n<b>{mark} por {escape(user_name)} às "
                                                      f"{entry['em'][11:16]}</b>", **_options())
            except Exception as exc:  # noqa: BLE001 - a decisão já valeu; só o rascunho não foi atualizado
                log.warning("não consegui atualizar o rascunho de %s: %s", item_id, exc)
            return mark
