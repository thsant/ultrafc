"""Card quadrado (1080×1080 PNG) de uma checagem, para repostar no Instagram e no WhatsApp."""

from __future__ import annotations

import io
import re
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

from ..factcheck.files import Claim, Verdict
from .format import _short_time

SIZE = 1080
PAD = 72
BAND = 230
FOOTER = 930  # início do rodapé
MAX_RESUMO = 200

# (fundo da faixa, texto da faixa)
COLORS = {"verdadeiro": ("#1E8E3E", "#FFFFFF"), "falso": ("#D93025", "#FFFFFF"),
          "exagerado": ("#E8710A", "#FFFFFF"), "omissão": ("#F2B600", "#1F1F1F"),
          "inconclusivo": ("#5F6368", "#FFFFFF")}
BACKGROUND, INK, MUTED, RULE = "#FAFAF7", "#1F1F1F", "#5F6368", "#DADCE0"

FONT_DIR = Path(sys.prefix) / "fonts"  # font-ttf-ubuntu, instalada pelo pixi


@lru_cache(maxsize=64)
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    for name in (f"Ubuntu-{weight}.ttf", "DejaVuSans.ttf"):
        if (FONT_DIR / name).exists():
            return ImageFont.truetype(str(FONT_DIR / name), size)
    return ImageFont.load_default(size)


def wrap(text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and f.getlength(f"{lines[-1]} {word}") <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines


def fit(text: str, weight: str, sizes: range, width: int, height: int,
        spacing: float = 1.22) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Maior fonte em que o texto cabe na caixa; na menor, corta as linhas que sobram com "…"."""
    for size in sizes:
        f, step = font(weight, size), round(size * spacing)
        lines = wrap(text, f, width)
        if len(lines) * step <= height:
            return f, lines, step
    keep = max(height // step, 1)
    if len(lines) > keep:
        lines = lines[:keep]
        while lines[-1] and f.getlength(lines[-1] + "…") > width:
            lines[-1] = lines[-1].rsplit(" ", 1)[0] if " " in lines[-1] else lines[-1][:-1]
        lines[-1] += "…"
    return f, lines, step


def summary_text(v: Verdict) -> str:
    """O resumo do checador; nas checagens antigas, a 1ª frase da justificativa."""
    text = " ".join((v.resumo or re.split(r"(?<=[.!?])\s", v.justificativa.strip(), maxsplit=1)[0]).split())
    return text if len(text) <= MAX_RESUMO else text[:MAX_RESUMO].rsplit(" ", 1)[0] + "…"


def domains(v: Verdict, limit: int = 3) -> list[str]:
    out: list[str] = []
    for s in v.fontes:
        host = urlparse(s.url).netloc.removeprefix("www.")
        if s.consultada and host and host not in out:
            out.append(host)
    return out[:limit]


def render_card(claim: Claim, verdict: Verdict, brand: str | None = None, footer: str | None = None) -> bytes:
    band, band_ink = COLORS.get(verdict.resultado, (MUTED, "#FFFFFF"))
    img = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    d = ImageDraw.Draw(img)
    width = SIZE - 2 * PAD

    d.rectangle((0, 0, SIZE, BAND), fill=band)
    d.text((PAD, 52), "CHECAGEM", font=font("M", 32), fill=band_ink)
    label, _, _ = fit(verdict.resultado.upper(), "B", range(112, 60, -4), width, 130)
    d.text((PAD, 92), verdict.resultado.upper(), font=label, fill=band_ink)

    y = BAND + 52
    d.text((PAD, y), f"{claim.quem} · {_short_time(claim.horario)}", font=font("M", 34), fill=MUTED)
    y += 64

    f, lines, step = fit(f"“{claim.afirmacao.strip()}”", "B", range(58, 33, -2), width, 330)
    for line in lines:
        d.text((PAD, y), line, font=f, fill=INK)
        y += step
    y += 40

    bar = 10
    f, lines, step = fit(summary_text(verdict), "R", range(40, 27, -2), width - bar - 28, FOOTER - 40 - y)
    d.rectangle((PAD, y + 4, PAD + bar, y + len(lines) * step - round(step * 0.15)), fill=band)
    for line in lines:
        d.text((PAD + bar + 28, y), line, font=f, fill=INK)
        y += step

    d.line((PAD, FOOTER, SIZE - PAD, FOOTER), fill=RULE, width=2)
    if hosts := domains(verdict):
        f, lines, _ = fit("Fontes: " + " · ".join(hosts), "R", range(28, 21, -2), width, 34)
        d.text((PAD, FOOTER + 26), lines[0], font=f, fill=MUTED)
    if mark := " · ".join(x for x in (brand, footer) if x):
        f, lines, _ = fit(mark, "M", range(28, 21, -2), width, 34)
        d.text((PAD, FOOTER + 72), lines[0], font=f, fill=INK)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def save_card(path: Path, claim: Claim, verdict: Verdict, brand: str | None = None,
              footer: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(render_card(claim, verdict, brand, footer))
    tmp.replace(path)
    return path
