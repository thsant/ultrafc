"""Cards quadrados (1080×1080 PNG) das checagens (nossas e de agências) e do Procure saber!, para repostar no
Instagram e no WhatsApp."""

from __future__ import annotations

import io
import re
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

from ..factcheck.files import Claim, Verdict
from ..saber.files import Materia, speaker_name
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
SABER_COLOR = "#0B6E77"

FONT_DIR = Path(sys.prefix) / "fonts"  # font-ttf-ubuntu, instalada pelo pixi
MASCOT = Path(__file__).with_name("assets") / "mascot.png"
MASCOT_SABER = Path(__file__).with_name("assets") / "mascot-research.png"
SABER_CROP = (60, 0, 460, 400)  # cabeça de óculos e o alto do jornal, na arte de 512 px
MASCOT_SIZE = 196


@lru_cache(maxsize=4)
def mascot(path: Path = MASCOT, crop: tuple[int, int, int, int] | None = None) -> Image.Image | None:
    """O carcará em círculo com borda branca, para o canto da faixa; None se o arquivo faltar."""
    try:
        art = Image.open(path).convert("RGB")
    except OSError:
        return None
    if crop is not None:
        art = art.crop(crop)
    ring, scale = 6, 4  # desenha 4× maior e reduz: borda do círculo sem serrilhado
    big = MASCOT_SIZE * scale
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
    inner = Image.new("L", (big, big), 0)
    ImageDraw.Draw(inner).ellipse((ring * scale,) * 2 + (big - 1 - ring * scale,) * 2, fill=255)
    out = Image.new("RGB", (big, big), "#FFFFFF")
    out.paste(art.resize((big, big), Image.LANCZOS), (0, 0), inner)
    out.putalpha(mask)
    return out.resize((MASCOT_SIZE, MASCOT_SIZE), Image.LANCZOS)


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
        if len(lines) * step <= height and all(f.getlength(line) <= width for line in lines):
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


def _band(img: Image.Image, color: str, ink: str, kicker: str, label: str, bird: Image.Image | None) -> None:
    """Faixa do topo: o tipo do card em letras pequenas, o rótulo grande e o mascote no canto."""
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, SIZE, BAND), fill=color)
    label_width = SIZE - 2 * PAD
    if bird is not None:
        img.paste(bird, (SIZE - PAD + 24 - MASCOT_SIZE, (BAND - MASCOT_SIZE) // 2), bird)
        label_width -= MASCOT_SIZE
    d.text((PAD, 52), kicker, font=font("M", 32), fill=ink)
    f, _, _ = fit(label, "B", range(112, 60, -4), label_width, 130)
    d.text((PAD, 92), label, font=f, fill=ink)


def _bar_text(d: ImageDraw.ImageDraw, y: int, text: str, color: str, sizes: range, height: int,
              weight: str = "R") -> int:
    """Parágrafo com uma barra vertical na cor da faixa à esquerda; devolve o y depois dele."""
    bar = 10
    f, lines, step = fit(text, weight, sizes, SIZE - 2 * PAD - bar - 28, height)
    d.rectangle((PAD, y + 4, PAD + bar, y + len(lines) * step - round(step * 0.15)), fill=color)
    for line in lines:
        d.text((PAD + bar + 28, y), line, font=f, fill=INK)
        y += step
    return y


def _bottom(d: ImageDraw.ImageDraw, sources: str | None, brand: str | None, footer: str | None) -> None:
    """Rodapé: as fontes (ou o site da matéria) e a marca do canal."""
    width = SIZE - 2 * PAD
    d.line((PAD, FOOTER, SIZE - PAD, FOOTER), fill=RULE, width=2)
    if sources:
        f, lines, _ = fit(sources, "R", range(28, 21, -2), width, 34)
        d.text((PAD, FOOTER + 26), lines[0], font=f, fill=MUTED)
    if mark := " · ".join(x for x in (brand, footer) if x):
        f, lines, _ = fit(mark, "M", range(28, 21, -2), width, 34)
        d.text((PAD, FOOTER + 72), lines[0], font=f, fill=INK)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def render_card(claim: Claim, verdict: Verdict, brand: str | None = None, footer: str | None = None) -> bytes:
    band, band_ink = COLORS.get(verdict.resultado, (MUTED, "#FFFFFF"))
    img = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    d = ImageDraw.Draw(img)
    width = SIZE - 2 * PAD
    _band(img, band, band_ink, "CHECAGEM", verdict.resultado.upper(), mascot())

    y = BAND + 52
    d.text((PAD, y), f"{claim.quem} · {_short_time(claim.horario)}", font=font("M", 34), fill=MUTED)
    y += 64

    f, lines, step = fit(f"“{claim.afirmacao.strip()}”", "B", range(58, 33, -2), width, 330)
    for line in lines:
        d.text((PAD, y), line, font=f, fill=INK)
        y += step
    y += 40

    _bar_text(d, y, summary_text(verdict), band, range(40, 27, -2), FOOTER - 40 - y)
    hosts = domains(verdict)
    _bottom(d, "Fontes: " + " · ".join(hosts) if hosts else None, brand, footer)
    return _png(img)


def render_saber_card(m: Materia, brand: str | None = None, footer: str | None = None) -> bytes:
    """Procure saber!: a matéria (veículo, chamada, resumo) e a fala da transmissão a que ela se liga."""
    img = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    d = ImageDraw.Draw(img)
    width = SIZE - 2 * PAD
    _band(img, SABER_COLOR, "#FFFFFF", "NA IMPRENSA", "PROCURE SABER!", mascot(MASCOT_SABER, SABER_CROP))

    y = BAND + 52
    if source := " · ".join(x for x in (m.veiculo.strip(), m.data_publicacao.strip()) if x):
        f, lines, _ = fit(source, "M", range(34, 25, -2), width, 44)
        d.text((PAD, y), lines[0], font=f, fill=MUTED)
        y += 64

    quote_space = 150 if m.falas else 0  # a fala é a ligação com a transmissão: não pode sumir
    f, lines, step = fit(m.chamada.strip() or m.titulo.strip(), "B", range(58, 33, -2), width,
                         230 - quote_space // 5)
    for line in lines:
        d.text((PAD, y), line, font=f, fill=INK)
        y += step
    y += 36

    if m.resumo.strip():
        height = min(190, FOOTER - 40 - y - quote_space)
        y = _bar_text(d, y, " ".join(m.resumo.split()), SABER_COLOR, range(38, 25, -2), height) + 36

    if m.falas:
        fala = m.falas[0]
        d.text((PAD, y), f"{speaker_name(fala.quem)} · {_short_time(fala.horario)}", font=font("M", 30), fill=MUTED)
        y += 46
        f, lines, step = fit(f"“{fala.trecho.strip()}”", "RI", range(40, 25, -2), width, FOOTER - 40 - y)
        for line in lines:
            d.text((PAD, y), line, font=f, fill=INK)
            y += step

    host = urlparse(m.url).netloc.removeprefix("www.")
    _bottom(d, f"Leia em {host}" if host else None, brand, footer)
    return _png(img)


def render_agency_card(m: Materia, brand: str | None = None, footer: str | None = None) -> bytes:
    """Checagem de uma agência: o selo dela na faixa (cor pela nossa categoria) e o crédito da apuração."""
    band, band_ink = COLORS.get(m.categoria, (MUTED, "#FFFFFF"))
    img = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    d = ImageDraw.Draw(img)
    width = SIZE - 2 * PAD
    _band(img, band, band_ink, f"CHECAGEM · {m.veiculo.strip().upper()}", m.selo.strip().upper(), mascot())

    y = BAND + 52
    if who := m.quem_afirmou.strip():  # o autor segundo a agência; a fala da transmissão vem à parte
        f, lines, _ = fit(who, "M", range(34, 25, -2), width, 44)
        d.text((PAD, y), lines[0], font=f, fill=MUTED)
        y += 64

    f, lines, step = fit(f"“{m.afirmacao.strip()}”", "B", range(58, 33, -2), width, 250)
    for line in lines:
        d.text((PAD, y), line, font=f, fill=INK)
        y += step
    y += 36

    credit = 108  # linha da apuração, acima do rodapé
    quote_space = 104 if m.falas else 0
    if m.resumo.strip():
        y = _bar_text(d, y, " ".join(m.resumo.split()), band, range(40, 25, -2),
                      FOOTER - credit - 20 - quote_space - y) + 24
    if m.falas:
        fala = m.falas[0]
        said = f"Na transmissão, {speaker_name(fala.quem)} · {_short_time(fala.horario)}: “{fala.trecho.strip()}”"
        f, lines, step = fit(said, "RI", range(28, 21, -2), width, FOOTER - credit - 16 - y)
        for line in lines:
            d.text((PAD, y), line, font=f, fill=MUTED)
            y += step
    agency = f"Apuração: {m.veiculo.strip()}" + (f" · por {m.autoria.strip()}" if m.autoria.strip() else "")
    f, lines, _ = fit(agency, "M", range(30, 21, -2), width, 80)
    for i, line in enumerate(lines[:2]):
        d.text((PAD, FOOTER - credit + i * round(f.size * 1.25)), line, font=f, fill=INK)

    host = urlparse(m.url).netloc.removeprefix("www.")
    _bottom(d, f"Leia em {host}" if host else None, brand, footer)
    return _png(img)


def _save(path: Path, png: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(png)
    tmp.replace(path)
    return path


def save_card(path: Path, claim: Claim, verdict: Verdict, brand: str | None = None,
              footer: str | None = None) -> Path:
    return _save(path, render_card(claim, verdict, brand, footer))


def save_saber_card(path: Path, m: Materia, brand: str | None = None, footer: str | None = None) -> Path:
    render = render_agency_card if m.tipo == "checagem" else render_saber_card
    return _save(path, render(m, brand, footer))
