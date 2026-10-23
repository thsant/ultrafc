"""CLI: `python -m ultrafc.factcheck --log-dir logs/<stream>` acompanha a transcrição e checa afirmações."""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import time
from pathlib import Path

from rich.text import Text
from rich_argparse import RichHelpFormatter

from ..console import make_consoles, setup_logging
from ..logger import TIME_FORMAT, LogLine
from . import check as checker
from . import detect as detector
from . import review as reviewer
from . import usage
from .files import Claim, new_id, parse_claim, pending, render_claim, render_verdict, write_atomic
from .tail import LogTail

log = logging.getLogger("ultrafc.factcheck")

RESULT_STYLE = {"verdadeiro": "bold green", "falso": "bold red", "exagerado": "bold dark_orange",
                "omissão": "bold yellow", "inconclusivo": "bold blue", "erro": "bold magenta",
                "rejeitado": "bold bright_black"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ultrafc.factcheck", description=__doc__, formatter_class=RichHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-dir", type=Path, help="pasta da stream (logs/<stream>): segue o log do dia")
    src.add_argument("--log", type=Path, help="um arquivo de log específico")
    p.add_argument("--out", type=Path, default=Path("checking"), help="pasta de saída (padrão: checking)")
    p.add_argument("--detect-model", default=detector.DEFAULT_MODEL)
    p.add_argument("--check-model", default=checker.DEFAULT_MODEL)
    p.add_argument("--review-model", default=reviewer.DEFAULT_MODEL,
                   help="confere cada checagem só com as fontes citadas; reprovadas vão para rejected/")
    p.add_argument("--max-parallel", type=int, default=3, help="checagens simultâneas (padrão: 3)")
    p.add_argument("--window-lines", type=int, default=6, help="linhas novas que disparam a detecção")
    p.add_argument("--window-seconds", type=float, default=45.0,
                   help="tempo máximo até analisar linhas novas pendentes")
    p.add_argument("--context-lines", type=int, default=10, help="linhas anteriores enviadas como contexto")
    p.add_argument("--from-start", action="store_true",
                   help="analisa o log desde o início (padrão: só o que for escrito daqui em diante)")
    p.add_argument("--once", action="store_true", help="analisa o que já está no log, termina as checagens e sai")
    p.add_argument("--stop-after", type=int, help="para de detectar após N afirmações (testes com custo controlado)")
    p.add_argument("--color", default="auto", choices=["auto", "always", "never"])
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


class FactChecker:
    def __init__(self, args: argparse.Namespace, client, console):
        self.args = args
        self.client = client
        self.console = console
        self.todo_dir = args.out / "to_check"
        self.done_dir = args.out / "checked"
        self.rejected_dir = args.out / "rejected"
        source = args.log_dir or args.log
        self.stream = (args.log_dir or args.log.parent).name
        self.tail = LogTail(source, args.out / f".state-{self.stream}.json", args.from_start)
        self.queue: asyncio.Queue[Claim] = asyncio.Queue()
        self.known: list[str] = []
        self.registered = 0
        self.running = 0
        self.counts: collections.Counter = collections.Counter()
        self.started = time.strftime(TIME_FORMAT)
        self.usage_path = args.out / "usage" / f"{self.stream}-{time.strftime('%Y%m%d-%H%M%S')}.json"

    @property
    def stopped(self) -> bool:
        return self.args.stop_after is not None and self.registered >= self.args.stop_after

    # --- detecção -----------------------------------------------------------------------------
    async def watch(self) -> None:
        context: collections.deque[LogLine] = collections.deque(maxlen=self.args.context_lines)
        pending_lines: list[LogLine] = []
        last = time.monotonic()
        while not self.stopped:
            pending_lines += self.tail.read()
            due = time.monotonic() - last >= self.args.window_seconds
            while pending_lines and (len(pending_lines) >= self.args.window_lines or due or self.args.once):
                window, pending_lines = pending_lines[:self.args.window_lines], pending_lines[self.args.window_lines:]
                await self.analyze(list(context), window)
                context.extend(window)
                last, due = time.monotonic(), False
                if self.stopped:
                    break
            if not pending_lines:
                self.tail.commit()
            if self.args.once and not pending_lines:
                return
            await asyncio.sleep(1.0)

    async def analyze(self, context: list[LogLine], window: list[LogLine]) -> None:
        try:
            found = await detector.detect(self.client, self.args.detect_model, context, window, self.known)
        except Exception as exc:  # noqa: BLE001 - uma falha de detecção não derruba a checagem em curso
            log.warning("falha na detecção (%s); trecho de %d linhas ignorado", exc, len(window))
            return
        stamps = {time.strftime(TIME_FORMAT, time.localtime(line.ts)) for line in window}
        for a in found:
            if self.stopped:
                return
            horario = a.horario if a.horario in stamps else time.strftime(TIME_FORMAT, time.localtime(window[0].ts))
            ts = time.mktime(time.strptime(horario, TIME_FORMAT))
            claim = Claim(new_id(ts, a.afirmacao, self.todo_dir), self.stream, a.quem, horario, a.tipo,
                          a.afirmacao, a.trecho_original, a.rationale)
            write_atomic(self.todo_dir / f"{claim.id}.md", render_claim(claim))
            self.known.append(f"{a.quem}: {a.afirmacao}")
            self.registered += 1
            log.info("🔎 nova afirmação (%s): %s", a.quem, a.afirmacao)
            self.queue.put_nowait(claim)

    # --- checagem -----------------------------------------------------------------------------
    async def worker(self) -> None:
        while True:
            claim = await self.queue.get()
            self.running += 1
            log.info("checando (%d em andamento, %d na fila): %s", self.running, self.queue.qsize(),
                     claim.afirmacao[:80])
            try:
                verdict = await checker.check(self.client, self.args.check_model, claim)
            except Exception as exc:  # noqa: BLE001 - o erro vira um resultado registrado, não some
                verdict = checker._error(self.args.check_model, f"falha na checagem: {exc}")
            rev = None
            if verdict.resultado != "erro":  # erro não tem o que revisar e não é publicado
                try:
                    rev = await reviewer.review(self.client, self.args.review_model, claim, verdict)
                except Exception as exc:  # noqa: BLE001 - sem revisão, não publica
                    rev = reviewer._rejected(self.args.review_model, f"falha na revisão: {exc}")
            directory = self.done_dir if rev is None or rev.aprovado else self.rejected_dir
            path = directory / f"{claim.id}.md"
            write_atomic(path, render_verdict(claim, verdict, rev))
            self.counts["checagens"] += 1
            self.counts["erros" if rev is None else "aprovadas" if rev.aprovado else "rejeitadas"] += 1
            self.save_usage()
            self.show(claim, verdict.resultado, path, rev)
            self.running -= 1
            self.queue.task_done()

    def save_usage(self) -> None:
        usage.save(self.usage_path, stream=self.stream, inicio=self.started, **self.counts)

    def show(self, claim: Claim, resultado: str, path, rev=None) -> None:
        label = resultado if rev is None or rev.aprovado else "rejeitado"
        line = Text(f"{label.upper():>10}  ", style=RESULT_STYLE.get(label, "bold"))
        line.append(f"{claim.quem}: ", style="bold")
        line.append(claim.afirmacao)
        if label == "rejeitado":
            line.append(f" ({resultado}; {rev.motivo[:120]})", style="dim")
        line.append(f"  → {path}", style="dim")
        self.console.print(line)

    async def run(self) -> None:
        for path in sorted(self.todo_dir.glob("*.md")):  # evita registrar de novo o que já foi visto
            claim = parse_claim(path.read_text(encoding="utf-8"))
            if claim.stream == self.stream:
                self.known.append(f"{claim.quem}: {claim.afirmacao}")
        for claim in pending(self.args.out):
            if claim.stream == self.stream:
                self.queue.put_nowait(claim)
        if self.queue.qsize():
            log.info("retomando %d afirmação(ões) pendente(s)", self.queue.qsize())
        workers = [asyncio.create_task(self.worker()) for _ in range(self.args.max_parallel)]
        try:
            await self.watch()
            if self.args.once or self.stopped:
                await self.queue.join()
            else:
                await asyncio.Event().wait()
        finally:
            for w in workers:
                w.cancel()
            self.save_usage()


def main(argv: list[str] | None = None) -> None:
    import anthropic

    args = parse_args(argv)
    stdout, stderr = make_consoles(args.color)
    setup_logging(stderr, args.verbose)
    client = anthropic.AsyncAnthropic(max_retries=4)
    try:
        asyncio.run(FactChecker(args, client, stdout).run())
    except KeyboardInterrupt:
        log.info("encerrando; afirmações sem resultado continuam em %s e serão retomadas", args.out / "to_check")
    finally:
        for line in usage.summary():
            log.info("uso: %s", line)


if __name__ == "__main__":
    main()
