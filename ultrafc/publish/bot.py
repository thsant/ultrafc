"""Partes do bot do Telegram comuns ao publish e ao saber: id do chat, conferência do canal e --whoami."""

from __future__ import annotations

import logging

log = logging.getLogger("ultrafc.publish")


def chat_id(value: str) -> int | str:
    return int(value) if value.lstrip("-").isdigit() else value


async def check_channel(bot, channel: int | str) -> int:
    """Confere, antes de começar, que o canal é o esperado e que o bot pode publicar nele.

    Devolve o id numérico do canal. Um @nome é global no Telegram: se o canal da equipe for privado,
    o @nome aponta para outro canal qualquer, e a publicação falharia só no primeiro clique.
    """
    from telegram.error import TelegramError

    try:
        chat = await bot.get_chat(channel)
    except TelegramError as exc:
        raise SystemExit(f"canal {channel} não encontrado: {exc}") from exc
    where = f"{chat.title!r} ({'@' + chat.username if chat.username else 'privado'}, id {chat.id})"
    try:
        member = await bot.get_chat_member(chat.id, bot.id)
        can_post = member.status == "creator" or (member.status == "administrator"
                                                  and getattr(member, "can_post_messages", True) is not False)
    except TelegramError:
        can_post = False  # "Member list is inaccessible": o bot não é administrador
    if not can_post:
        raise SystemExit(f"o bot não pode publicar no canal {where}. Se este não é o seu canal, use o id "
                         "numérico (-100…) do canal certo, que aparece no --whoami ao publicar algo nele; "
                         "se é, adicione o bot como administrador com permissão de publicar.")
    log.info("canal: %s", where)
    return chat.id


def whoami(token: str) -> None:
    from telegram import Update
    from telegram.ext import Application, MessageHandler, filters

    async def show(update: Update, _context) -> None:
        chat, user = update.effective_chat, update.effective_user
        log.info("chat: %s (%s, %s) · usuário: %s (%s)", chat.id, chat.type, chat.title or chat.full_name,
                 user.id if user else "-", user.full_name if user else "-")

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.ALL, show))
    log.info("mande uma mensagem ao bot (ou num grupo/canal onde ele está); Ctrl+C para sair")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
