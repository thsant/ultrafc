"""Mensagens do canal em HTML do Telegram: checagens e resumos."""

from __future__ import annotations

import time
from html import escape

from ..factcheck.files import Claim, Verdict
from ..logger import TIME_FORMAT

MAX_MESSAGE = 4096  # limite do Telegram para uma mensagem de texto
MAX_JUSTIFICATION = 900
MAX_SOURCES = 5

ICONS = {"verdadeiro": "✅", "falso": "❌", "exagerado": "🟠", "omissão": "🟡", "inconclusivo": "⚪"}


def _short_time(horario: str) -> str:
    """'2026-10-22 22:11:35' → '22/10 22:11'."""
    try:
        return time.strftime("%d/%m %H:%M", time.strptime(horario, TIME_FORMAT))
    except ValueError:
        return horario


def trim_paragraphs(text: str, limit: int) -> str:
    """Corta em parágrafos inteiros até `limit` caracteres; se nem o primeiro cabe, corta na palavra."""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    out: list[str] = []
    for p in paragraphs:
        if len("\n\n".join(out + [p])) > limit:
            break
        out.append(p)
    if len(out) == len(paragraphs):
        return "\n\n".join(out)
    if not out:
        return paragraphs[0][:limit].rsplit(" ", 1)[0] + "…"
    return "\n\n".join(out) + "…"


def _footer(footer: str | None, tag: str) -> str:
    return f"\n\n{tag}" + (f"\n<i>{escape(footer)}</i>" if footer else "")


def format_check(claim: Claim, verdict: Verdict, footer: str | None = None) -> str:
    fontes = [s for s in verdict.fontes if s.consultada][:MAX_SOURCES]
    links = " · ".join(f'<a href="{escape(s.url, quote=True)}">{escape(s.titulo or s.url)}</a>' for s in fontes)
    head = (f"{ICONS.get(verdict.resultado, '•')} <b>{escape(verdict.resultado.upper())}</b>\n"
            f"{escape(claim.quem)} · {_short_time(claim.horario)}\n"
            f"«{escape(claim.afirmacao)}»\n\n")
    tail = (f"\n\n<b>Fontes:</b> {links}" if links else "") + _footer(footer, "#checagem")
    budget = min(MAX_JUSTIFICATION, MAX_MESSAGE - len(head) - len(tail) - 10)
    return head + escape(trim_paragraphs(verdict.justificativa, budget)) + tail


def format_summary(start: float, end: float, bullets: list[str], footer: str | None = None) -> str:
    period = f"{time.strftime('%H:%M', time.localtime(start))}–{time.strftime('%H:%M', time.localtime(end))}"
    head, tail = f"🗒 <b>Resumo — {period}</b>\n\n", _footer(footer, "#resumo")
    items = [f"• {escape(b)}" for b in bullets]
    while items and len(head) + len("\n".join(items)) + len(tail) > MAX_MESSAGE:
        items.pop()  # corta tópicos inteiros, nunca no meio de uma tag HTML
    return head + "\n".join(items) + tail
