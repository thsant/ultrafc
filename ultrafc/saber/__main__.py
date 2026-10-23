"""CLI: `python -m ultrafc.saber` recebe URLs de matérias, liga cada uma ao que foi dito na transmissão e
envia um card "Procure saber!" + texto para revisão no Telegram; publica no canal após aprovação.

Para cada URL, a CLI pergunta se é uma matéria ou a checagem de uma agência (formatada como checagem, com a
agência e a autoria). Se o site não deixa o Claude ler a página, o operador cola o texto no terminal."""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from rich_argparse import RichHelpFormatter

from ..console import make_consoles, setup_logging
from ..factcheck import usage
from ..factcheck.check import _norm
from ..factcheck.files import write_atomic
from ..publish.bot import chat_id, check_channel, whoami
from ..publish.card import save_saber_card
from ..publish.format import format_agency_check, format_saber
from ..publish.review import Reviewer, short_key
from .analyze import BLOCKED, DEFAULT_MODEL, FETCH_MODEL, analyze, fetch_article, latest_log, load_transcript, \
    pasted_document
from .files import render_materia

log = logging.getLogger("ultrafc.saber")

TOKEN_ENV = "TELEGRAM_SABER_BOT_TOKEN"
END_TEXT, SKIP = ".", "pular"  # fim do texto colado; desistir da matéria que espera texto
MIN_TEXT = 300  # caracteres: menos que isso não é uma matéria
MATERIA, CHECAGEM = "materia", "checagem"
ANSWERS = {"": MATERIA, "m": MATERIA, "materia": MATERIA, "matéria": MATERIA,
           "c": CHECAGEM, "checagem": CHECAGEM}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ultrafc.saber", description=__doc__, formatter_class=RichHelpFormatter)
    p.add_argument("transcript", type=Path, nargs="?",
                   help="log da transcrição (logs/<stream>/AAAA-MM-DD.log) ou a pasta da stream (usa o mais recente)")
    p.add_argument("--whoami", action="store_true",
                   help="só mostra o id do chat e do usuário de quem mandar mensagem ao bot")
    p.add_argument("--checking", type=Path, default=Path("checking"), help="pasta da checagem (padrão: checking)")
    p.add_argument("--channel", help="canal público: @nome ou id numérico (-100…)")
    p.add_argument("--review-chat", help="chat onde os rascunhos chegam para aprovação (id numérico)")
    p.add_argument("--operators", default="", help="ids dos usuários que podem aprovar, separados por vírgula")
    p.add_argument("--footer", help='rodapé das mensagens (ex.: "Conteúdo selecionado pela equipe")')
    p.add_argument("--card-brand", help="nome no rodapé dos cards (padrão: o @nome do canal)")
    p.add_argument("--no-cards", action="store_true", help="publica só o texto, sem o card PNG")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"modelo da análise (padrão: {DEFAULT_MODEL})")
    p.add_argument("--fetch-model", default=FETCH_MODEL,
                   help=f"modelo que só lê a página, sem a transcrição (padrão: {FETCH_MODEL})")
    p.add_argument("--max-parallel", type=int, default=2, help="matérias analisadas ao mesmo tempo (padrão: 2)")
    p.add_argument("--token-env", default=TOKEN_ENV,
                   help=f"variável com o token do bot (padrão: {TOKEN_ENV}); use um bot diferente do publish")
    p.add_argument("--color", default="auto", choices=["auto", "always", "never"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if not args.whoami:
        missing = [f for f in ("transcript", "channel", "review_chat") if not getattr(args, f)]
        if missing:
            p.error("faltam: " + ", ".join(m if m == "transcript" else "--" + m.replace("_", "-") for m in missing))
        if not args.operators:
            p.error("informe --operators (use --whoami para descobrir seu id)")
    return args


def item_id(url: str) -> str:
    return "saber-" + short_key(_norm(url))


class Saber:
    """Lê e analisa cada URL recebida e envia o resultado para revisão; uma URL já enviada não é refeita.

    Quando o site recusa a leitura, a URL entra na fila `awaiting` até o operador colar o texto.
    """

    def __init__(self, reviewer: Reviewer, client, args, brand: str | None, usage_path: Path):
        self.reviewer, self.client, self.args, self.brand = reviewer, client, args, brand
        self.usage_path = usage_path
        self.sem = asyncio.Semaphore(args.max_parallel)
        self.busy: set[str] = set()
        self.awaiting: collections.deque[tuple[str, str]] = collections.deque()  # (url, tipo)
        self.blocked_hosts: set[str] = set()  # sites que recusaram a leitura: nem tenta de novo

    def _waiting(self, key: str) -> bool:
        return any(item_id(u) == key for u, _ in self.awaiting)

    async def handle(self, url: str, kind: str = MATERIA, text: str | None = None) -> None:
        """Uma URL nova (`text` None: lê a página) ou o texto colado para uma URL que o esperava."""
        key = item_id(url)
        if self.reviewer.known(key):
            entry = self.reviewer.state[key]
            log.warning("já enviada (%s%s): %s", entry["status"],
                        f" por {entry['por']}" if entry.get("por") else "", url)
            return
        if key in self.busy or (text is None and self._waiting(key)):
            log.warning("já em análise ou esperando o texto: %s", url)
            return
        self.busy.add(key)
        try:
            await self._process(key, url, kind, text)
        except Exception as exc:  # noqa: BLE001 - erro de rede/API/Telegram: o operador pode colar de novo
            log.warning("falha com %s: %s", url, exc)
        finally:
            self.busy.discard(key)

    def ask_text(self, url: str, kind: str, reason: str) -> None:
        self.awaiting.append((url, kind))
        log.warning("%s: %s", reason, url)
        log.info("📋 cole o texto da matéria e termine com uma linha só com \"%s\" (ou digite \"%s\")",
                 END_TEXT, SKIP)

    async def _process(self, key: str, url: str, kind: str, text: str | None) -> None:
        args = self.args
        host = urlparse(url).hostname or ""
        if text is not None:
            article, origem = pasted_document(text), "colado"
            write_atomic(args.checking / "saber" / f"{key}.txt", text.strip() + "\n")
        elif host in self.blocked_hosts:
            self.ask_text(url, kind, f"{host} não permite a leitura por IA")
            return
        else:
            async with self.sem:
                log.info("📖 lendo %s", url)
                try:
                    article, error = await fetch_article(self.client, args.fetch_model, url)
                finally:
                    usage.save(self.usage_path)
            if article is None:
                if error == BLOCKED:
                    self.blocked_hosts.add(host)
                    self.ask_text(url, kind, f"{host} não permite a leitura por IA")
                else:
                    self.ask_text(url, kind, f"não consegui ler a página ({error})")
                return
            origem = "web"

        async with self.sem:
            lines = load_transcript(args.transcript)
            if not lines:
                log.warning("transcrição vazia ou inexistente em %s; ignorando %s", args.transcript, url)
                return
            log.info("🔎 analisando %s %s (%d falas na transcrição)",
                     "a checagem" if kind == CHECAGEM else "a matéria", url, len(lines))
            try:
                m = await analyze(self.client, args.model, key, url, lines, article, stream_name(args.transcript),
                                  kind)
            finally:
                usage.save(self.usage_path)
        m.origem = origem
        write_atomic(args.checking / "saber" / f"{key}.md", render_materia(m))
        if m.erro:
            log.warning("sem resultado para %s: %s", url, m.erro)
            return
        if not m.relacionada:
            log.warning("%s, não enviada: %s — %s", "não é uma checagem" if kind == CHECAGEM
                        else "matéria sem ligação com a transmissão", url, m.motivo)
            return
        if kind == CHECAGEM and not m.falas:
            log.info("checagem sem fala ligada na transmissão; vai com quem afirmou segundo a agência: %s",
                     m.quem_afirmou)
        card = None if args.no_cards else save_saber_card(args.checking / "cards" / f"{key}.png", m, self.brand,
                                                          args.footer)
        if kind == CHECAGEM:
            await self.reviewer.submit(key, "checagem-externa", format_agency_check(m, args.footer), card)
        else:
            await self.reviewer.submit(key, "procure-saber", format_saber(m, args.footer), card)


def stream_name(transcript: Path) -> str:
    path = latest_log(transcript)
    return path.parent.name if path else transcript.name


def read_lines(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    """Thread daemon: um input() dentro do asyncio prenderia o encerramento até a próxima linha."""
    for line in sys.stdin:
        loop.call_soon_threadsafe(queue.put_nowait, line.rstrip("\n"))
    loop.call_soon_threadsafe(queue.put_nowait, None)


def is_url(line: str) -> bool:
    return line.startswith(("http://", "https://")) and " " not in line


async def receive(saber: Saber, queue: asyncio.Queue) -> None:
    """Uma URL por linha, seguida da resposta à pergunta "matéria ou checagem?". Com uma matéria esperando
    texto, o que não é URL começa o texto colado, que vai até uma linha só com END_TEXT (ou o fim da entrada)."""
    pending: set[asyncio.Task] = set()
    asking: collections.deque[str] = collections.deque()  # URLs esperando a resposta, na ordem

    def spawn(url: str, kind: str, text: str | None = None) -> None:
        task = asyncio.create_task(saber.handle(url, kind, text))
        pending.add(task)
        task.add_done_callback(pending.discard)

    def ask() -> None:
        log.info("❓ %s\n   matéria ou checagem? [m/c] (Enter = matéria)", asking[0])

    def finish(lines: list[str]) -> None:
        (url, kind), text = saber.awaiting.popleft(), "\n".join(lines).strip()
        if len(text) < MIN_TEXT:
            saber.awaiting.appendleft((url, kind))
            log.warning("texto curto demais (%d caracteres); cole o texto inteiro de %s", len(text), url)
            return
        log.info("📋 texto recebido (%d caracteres): %s", len(text), url)
        spawn(url, kind, text)

    pasting: list[str] | None = None
    while (line := await queue.get()) is not None:
        if pasting is not None:
            if line.strip() == END_TEXT:
                finish(pasting)
                pasting = None
            else:
                pasting.append(line)
            continue
        line = line.strip()
        if is_url(line):
            asking.append(line)
            if len(asking) == 1:
                ask()
        elif asking:
            if (kind := ANSWERS.get(line.lower())) is None:
                log.warning("responda m (matéria) ou c (checagem)")
            else:
                spawn(asking.popleft(), kind)
            if asking:
                ask()
        elif not line:
            continue
        elif saber.awaiting and line.lower() == SKIP:
            log.info("sem texto, desisti de %s", saber.awaiting.popleft()[0])
        elif saber.awaiting:
            log.info("recebendo o texto de %s; termine com uma linha só com \"%s\"", saber.awaiting[0][0], END_TEXT)
            pasting = [line]
        else:
            log.warning("não parece uma URL (http:// ou https://): %s", line)
    if pasting:
        finish(pasting)
    for url in asking:
        log.warning("sem resposta, tratada como matéria: %s", url)
        spawn(url, MATERIA)
    log.info("fim da entrada; aguardando as análises e as decisões no chat de revisão (Ctrl+C para sair)")
    await asyncio.gather(*pending)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, stderr = make_consoles(args.color)
    setup_logging(stderr, args.verbose)
    token = os.environ.get(args.token_env)
    if not token:
        log.error("defina %s no ambiente (no topo do ~/.bashrc), com o token de um bot só para o saber",
                  args.token_env)
        sys.exit(1)
    if args.whoami:
        whoami(token)
        return
    if latest_log(args.transcript) is None:
        log.error("transcrição não encontrada: %s", args.transcript)
        sys.exit(1)

    import anthropic
    from telegram import Update
    from telegram.ext import Application, CallbackQueryHandler

    operators = {int(x) for x in args.operators.split(",") if x.strip()}
    brand = args.card_brand or (args.channel if args.channel.startswith("@") else None)
    usage_path = args.checking / "usage" / f"saber-{time.strftime('%Y%m%d-%H%M%S')}.json"
    tasks: list[asyncio.Task] = []

    async def preflight() -> int:
        from telegram import Bot

        async with Bot(token) as bot:
            return await check_channel(bot, chat_id(args.channel))

    channel = asyncio.run(preflight())

    async def start(app: Application) -> None:
        reviewer = Reviewer(app.bot, args.checking / ".saber-state.json", chat_id(args.review_chat), channel,
                            operators)
        app.bot_data["reviewer"] = reviewer
        saber = Saber(reviewer, anthropic.AsyncAnthropic(max_retries=4), args, brand, usage_path)
        queue: asyncio.Queue = asyncio.Queue()
        threading.Thread(target=read_lines, args=(asyncio.get_running_loop(), queue), daemon=True).start()
        tasks.append(asyncio.create_task(receive(saber, queue)))
        log.info("transcrição: %s; rascunhos em %s; %d operador(es)", latest_log(args.transcript),
                 args.review_chat, len(operators))
        log.info("cole a URL de uma matéria ou de uma checagem por linha e tecle Enter")

    async def stop(_app: Application) -> None:
        for t in tasks:
            t.cancel()
        for line in usage.summary():
            log.info("uso: %s", line)
        usage.save(usage_path)

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
