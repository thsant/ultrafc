"""Saída no terminal: falas coloridas por falante e mensagens do sistema em itálico."""

from __future__ import annotations

import logging
import re
import time

from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .logger import format_line

# Tableau 10 sem o cinza (reservado às mensagens): legível em fundo claro e escuro.
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#bcbd22", "#b07aa1", "#ff9da7", "#9c755f", "#17becf"]

NAME_WIDTH_MIN, NAME_WIDTH_MAX = 6, 20


def speaker_style(label: str | None) -> Style:
    """Cor pelo número do rótulo (P1 → 1ª cor), para continuar a mesma quando o falante ganha nome."""
    m = re.fullmatch(r"P(\d+)", label or "")
    if not m:
        return Style()
    return Style(color=PALETTE[(int(m.group(1)) - 1) % len(PALETTE)])


def make_consoles(color: str = "auto") -> tuple[Console, Console]:
    """(stdout para a transcrição, stderr para as mensagens)."""
    kw: dict = {}
    if color == "always":
        kw["force_terminal"] = True
    elif color == "never":
        kw["color_system"] = None
    return Console(highlight=False, **kw), Console(stderr=True, highlight=False, **kw)


class Display:
    def __init__(self, console: Console):
        self.console = console
        self._name_width = NAME_WIDTH_MIN

    def header(self, *parts: str) -> None:
        self.console.rule(Text(" · ".join(parts), style="bold"), style="dim")

    def status(self, message: str):
        return self.console.status(Text(message, style="italic"), spinner="dots")

    def utterance(self, ts: float, text: str, label: str | None = None, name: str | None = None) -> None:
        if not self.console.is_terminal:
            # Redirecionado/pipe: uma linha por fala, igual ao arquivo de log (bom para grep).
            self.console.print(format_line(ts, text, name), markup=False, soft_wrap=True)
            return

        style = speaker_style(label)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", no_wrap=True)
        row: list = [time.strftime("%H:%M:%S", time.localtime(ts))]
        if name:
            self._name_width = min(max(self._name_width, len(name)), NAME_WIDTH_MAX)
            grid.add_column(width=self._name_width, no_wrap=True, overflow="ellipsis")
            row.append(Text(name, style=style + Style(bold=True)))
        grid.add_column(ratio=1)
        row.append(Text(text, style=style))
        grid.add_row(*row)
        self.console.print(grid)


_LEVELS = {
    logging.DEBUG: ("·", "dim italic"),
    logging.INFO: ("ⓘ", "italic bright_black"),
    logging.WARNING: ("⚠", "italic yellow"),
    logging.ERROR: ("✖", "bold italic red"),
}


class MessageHandler(logging.Handler):
    """Mensagens que não são transcrição: em itálico, com ícone e cor pelo nível."""

    def __init__(self, console: Console, show_names: bool = False):
        super().__init__()
        self.console = console
        self.show_names = show_names

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = max((lv for lv in _LEVELS if lv <= record.levelno), default=logging.DEBUG)
            icon, style = _LEVELS[level]
            text = Text()
            if self.show_names:
                text.append(f"{record.name}: ", style="dim italic")
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + logging.Formatter().formatException(record.exc_info)
            text.append(message, style=style)
            # Ícone numa coluna própria: mensagens longas quebram alinhadas ao texto, não à margem.
            grid = Table.grid(padding=(0, 1))
            grid.add_column(width=3, justify="right", style=style)
            grid.add_column(ratio=1, overflow="fold")
            grid.add_row(icon, text)
            self.console.print(grid)
        except Exception:  # noqa: BLE001 - convenção do logging.Handler
            self.handleError(record)


# Clientes HTTP registram a URL de cada requisição; a da API do Telegram contém o token do bot.
_HTTP_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3")
_SECRETS = re.compile(r"(bot\d+):[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{10,}")


class RedactSecrets(logging.Filter):
    """Mascara tokens de bot do Telegram e chaves da Anthropic que apareçam numa mensagem."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        clean = _SECRETS.sub(lambda m: f"{m.group(1)}:<oculto>" if m.group(1) else "sk-ant-<oculto>", message)
        if clean != message:
            record.msg, record.args = clean, None
        return True


def setup_logging(console: Console, verbose: bool = False) -> None:
    """Mensagens do ultrafc em INFO (DEBUG com -v); bibliotecas só a partir de WARNING (INFO com -v).

    Os clientes HTTP ficam sempre em WARNING, mesmo com -v, para não expor tokens nas URLs.
    """
    root = logging.getLogger()
    handler = MessageHandler(console, show_names=verbose)
    handler.addFilter(RedactSecrets())
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger("ultrafc").setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in _HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
