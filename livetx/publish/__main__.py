"""CLI: `python -m livetx.publish` envia checagens e resumos para revisão no Telegram e publica no canal."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from rich_argparse import RichHelpFormatter

from ..console import make_consoles, setup_logging
from ..factcheck import usage
from ..factcheck.files import parse_verdict
from ..factcheck.tail import LogTail
from . import summary as summarizer
from .format import format_check, format_summary
from .review import Reviewer

log = logging.getLogger("livetx.publish")

SCAN_SECONDS = 5.0
MIN_SUMMARY_LINES = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="livetx.publish", description=__doc__, formatter_class=RichHelpFormatter)
    p.add_argument("--whoami", action="store_true",
                   help="só mostra o id do chat e do usuário de quem mandar mensagem ao bot")
    p.add_argument("--checking", type=Path, default=Path("checking"), help="pasta da checagem (padrão: checking)")
    p.add_argument("--channel", help="canal público: @nome ou id numérico (-100…)")
    p.add_argument("--review-chat", help="chat onde os rascunhos chegam para aprovação (id numérico)")
    p.add_argument("--operators", default="", help="ids dos usuários que podem aprovar, separados por vírgula")
    p.add_argument("--log-dir", type=Path, help="pasta da stream (logs/<stream>), para os resumos")
    p.add_argument("--summary-minutes", type=float, default=10.0, help="intervalo dos resumos (padrão: 10)")
    p.add_argument("--no-summary", action="store_true", help="publica só as checagens")
    p.add_argument("--summary-model", default=summarizer.DEFAULT_MODEL)
    p.add_argument("--from-start", action="store_true", help="resume o log desde o início")
    p.add_argument("--footer", help='rodapé das mensagens (ex.: "Checagem assistida por IA, revisada pela equipe")')
    p.add_argument("--color", default="auto", choices=["auto", "always", "never"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if not args.whoami:
        missing = [f for f in ("channel", "review_chat") if not getattr(args, f)]
        if missing:
            p.error("faltam: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        if not args.operators:
            p.error("informe --operators (use --whoami para descobrir seu id)")
    return args


def chat_id(value: str) -> int | str:
    return int(value) if value.lstrip("-").isdigit() else value


async def scan_checks(reviewer: Reviewer, checked: Path, footer: str | None) -> None:
    """Envia para revisão cada checagem nova; uma checagem com erro não tem o que publicar."""
    for path in sorted(checked.glob("*.md")):
        item_id = path.stem
        if reviewer.known(item_id):
            continue
        try:
            claim, verdict = parse_verdict(path.read_text(encoding="utf-8"))
        except (ValueError, KeyError) as exc:
            log.warning("arquivo ilegível %s: %s", path.name, exc)
            continue
        if verdict.resultado == "erro":
            log.warning("checagem com erro não vai para revisão: %s", item_id)
            reviewer.ignore(item_id, verdict.justificativa)
            continue
        await reviewer.submit(item_id, "checagem", format_check(claim, verdict, footer))


async def summaries(reviewer: Reviewer, client, args, tail: LogTail, out: Path) -> None:
    buffer = []
    while True:
        await asyncio.sleep(args.summary_minutes * 60)
        buffer += tail.read()
        if len(buffer) < MIN_SUMMARY_LINES:
            continue
        start, end = buffer[0].ts, buffer[-1].ts
        item_id = summarizer.summary_id(start, end)
        try:
            topics = await summarizer.summarize(client, args.summary_model, buffer)
        except Exception as exc:  # noqa: BLE001 - tenta de novo no próximo ciclo com o mesmo trecho
            log.warning("falha ao gerar resumo: %s", exc)
            continue
        if topics:
            summarizer.save(out, item_id, args.log_dir.name, start, end, topics)
            await reviewer.submit(item_id, "resumo", format_summary(start, end, topics, args.footer))
        tail.commit()
        buffer = []


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, stderr = make_consoles(args.color)
    setup_logging(stderr, args.verbose)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("defina TELEGRAM_BOT_TOKEN no ambiente (no topo do ~/.bashrc)")
        sys.exit(1)
    if args.whoami:
        whoami(token)
        return

    from telegram import Update
    from telegram.ext import Application, CallbackQueryHandler

    operators = {int(x) for x in args.operators.split(",") if x.strip()}
    tasks: list[asyncio.Task] = []

    # Antes de subir o bot: um SystemExit dentro do post_init seria tratado pela biblioteca como
    # parada normal, e o erro sumiria em silêncio.
    async def preflight() -> int:
        from telegram import Bot

        async with Bot(token) as bot:
            return await check_channel(bot, chat_id(args.channel))

    channel = asyncio.run(preflight())

    async def start(app: Application) -> None:
        reviewer = Reviewer(app.bot, args.checking / ".publish-state.json", chat_id(args.review_chat),
                            channel, operators)
        app.bot_data["reviewer"] = reviewer

        async def scan_loop() -> None:
            while True:
                try:
                    await scan_checks(reviewer, args.checking / "checked", args.footer)
                except Exception as exc:  # noqa: BLE001 - erro de rede/Telegram: tenta na próxima volta
                    log.warning("falha ao enviar para revisão: %s", exc)
                await asyncio.sleep(SCAN_SECONDS)

        tasks.append(asyncio.create_task(scan_loop()))
        if args.log_dir and not args.no_summary:
            import anthropic

            tail = LogTail(args.log_dir, args.checking / f".publish-tail-{args.log_dir.name}.json", args.from_start)
            tasks.append(asyncio.create_task(summaries(reviewer, anthropic.AsyncAnthropic(max_retries=4), args,
                                                       tail, args.checking / "summaries")))
        log.info("publicando em %s; rascunhos em %s; %d operador(es)", args.channel, args.review_chat,
                 len(operators))

    async def stop(_app: Application) -> None:
        for t in tasks:
            t.cancel()
        for line in usage.summary():
            log.info("uso: %s", line)

    async def on_click(update: Update, context) -> None:
        query = update.callback_query
        action, _, key = (query.data or "").partition(":")
        user = query.from_user
        notice = await context.bot_data["reviewer"].decide(action, key, user.id, user.full_name)
        await query.answer(notice)

    app = Application.builder().token(token).post_init(start).post_shutdown(stop).build()
    app.add_handler(CallbackQueryHandler(on_click))
    app.run_polling(allowed_updates=["callback_query"])


if __name__ == "__main__":
    main()
