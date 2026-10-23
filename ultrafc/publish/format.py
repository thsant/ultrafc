"""Mensagens do canal em HTML do Telegram: checagens (nossas e de agências), resumos e Procure saber!."""

from __future__ import annotations

import time
from html import escape

from ..factcheck.files import Claim, Verdict
from ..logger import TIME_FORMAT
from ..saber.files import Materia, speaker_name

MAX_MESSAGE = 4096  # limite do Telegram para uma mensagem de texto
MAX_JUSTIFICATION = 900
MAX_SOURCES = 5
MAX_TEXT = 1500  # texto do Procure saber!

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


def _link(s) -> str:
    """Uma matéria da base sem URL (`base:<arquivo>`) sai só com o título: o Telegram recusa o link."""
    if not s.url.startswith(("http://", "https://")):
        return escape(s.titulo or s.url)
    return f'<a href="{escape(s.url, quote=True)}">{escape(s.titulo or s.url)}</a>'


def format_check(claim: Claim, verdict: Verdict, footer: str | None = None) -> str:
    fontes = [s for s in verdict.fontes if s.consultada][:MAX_SOURCES]
    links = " · ".join(_link(s) for s in fontes)
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


def format_saber(m: Materia, footer: str | None = None) -> str:
    """Procure saber!: a matéria, o que foi dito a que ela se liga e o link."""
    titulo, veiculo = escape(m.titulo.strip() or m.chamada.strip()), escape(m.veiculo.strip())
    head = f"🔎 <b>Procure saber!</b>\n<b>{titulo}</b>" + (f" — {veiculo}" if veiculo else "") + "\n\n"
    falas = "".join(f"\n🗣 <i>{escape(speaker_name(f.quem))}, {f.horario[11:16]}:</i> «{escape(f.trecho)}»"
                    for f in m.falas)
    link = (f'<a href="{escape(m.url, quote=True)}">Leia a matéria' + (f" em {veiculo}" if veiculo else "")
            + "</a>")
    tail = (f"\n{falas}" if falas else "") + f"\n\n{link}" + _footer(footer, "#procuresaber")
    budget = min(MAX_TEXT, MAX_MESSAGE - len(head) - len(tail) - 10)
    return head + escape(trim_paragraphs(m.texto, budget)) + tail


def format_agency_check(m: Materia, footer: str | None = None) -> str:
    """Checagem de uma agência, no formato das nossas, com o selo e o crédito da apuração."""
    agency, who = escape(m.veiculo.strip()), escape(m.quem_afirmou.strip())  # o autor segundo a agência
    head = (f"{ICONS.get(m.categoria, '•')} <b>{escape(m.selo.strip().upper())}</b> · checagem de {agency}\n"
            + (f"{who}\n" if who else "") + f"«{escape(m.afirmacao.strip())}»\n\n")
    falas = "".join(f"\n🗣 <i>{escape(speaker_name(f.quem))}, {f.horario[11:16]}:</i> «{escape(f.trecho)}»"
                    for f in m.falas)
    falas = f"\n<b>Na transmissão:</b>{falas}" if falas else ""
    credit = (f'<b>Apuração:</b> <a href="{escape(m.url, quote=True)}">{agency}</a>'
              + (f", por {escape(m.autoria.strip())}" if m.autoria.strip() else ""))
    tail = (f"\n{falas}" if falas else "") + f"\n\n{credit}" + _footer(footer, "#checagem")
    budget = min(MAX_JUSTIFICATION, MAX_MESSAGE - len(head) - len(tail) - 10)
    return head + escape(trim_paragraphs(m.texto, budget)) + tail
