"""Base local de matérias (base/*.md, base/*.txt): vai inteira no system prompt do checador, em cache.

Com 20–30 matérias o agente lê tudo de uma vez, sem uma recuperação que possa deixar a certa de fora;
a leitura do cache custa 1/10 da entrada. As matérias de sites que bloqueiam a IA (Folha, g1…) entram
coladas pelo operador e o revisor as recebe anexadas, já que não conseguiria relê-las.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag

from .files import _one_line

log = logging.getLogger(__name__)

SUFFIXES = (".md", ".txt")
WARN_TOKENS = 80_000  # com os resultados da web, o turno passa de 100 mil: tarifa 5× no Haiku
_FRONTMATTER = re.compile(r"---\n(.*?)\n---\n?(.*)", re.S)
_URL = re.compile(r"^https?://\S+$")


def norm(url: str) -> str:
    return urldefrag(url)[0].rstrip("/").lower()


@dataclass
class Article:
    arquivo: str
    url: str
    veiculo: str
    titulo: str
    data: str
    texto: str

    @property
    def label(self) -> str:
        return " — ".join(p for p in (self.veiculo, self.titulo) if p)


def parse(path: Path) -> Article:
    """Cabeçalho opcional (url, veiculo, titulo, data); sem ele, uma URL na 1ª linha vira a `url`."""
    text = path.read_text(encoding="utf-8").strip()
    fields: dict[str, str] = {}
    if m := _FRONTMATTER.match(text):
        fields = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
        text = m.group(2).strip()
    else:
        first, _, rest = text.partition("\n")
        if _URL.match(first.strip()):
            fields["url"], text = first.strip(), rest.strip()
    url = fields.get("url", "").strip()
    if not url:
        log.warning("base: %s sem url; será citado como base:%s", path.name, path.name)
        url = f"base:{path.name}"
    return Article(path.name, url, fields.get("veiculo", "").strip(), fields.get("titulo", path.stem).strip(),
                   fields.get("data", "").strip(), text)


def load(directory: Path) -> list[Article]:
    """Em ordem de nome: o prefixo do prompt fica igual entre as chamadas e o cache vale."""
    if not directory.is_dir():
        return []
    return [parse(p) for p in sorted(directory.iterdir()) if p.suffix in SUFFIXES and p.is_file()]


def _signature(directory: Path) -> tuple:
    if not directory.is_dir():
        return ()
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(directory.iterdir()) if p.suffix in SUFFIXES)


class Base:
    def __init__(self, directory: Path):
        self.directory = directory
        self.signature: tuple = ()
        self.articles: list[Article] = []
        self.refresh()

    def refresh(self) -> bool:
        """Relê a pasta se algum arquivo entrou, saiu ou mudou; devolve True se recarregou."""
        signature = _signature(self.directory)
        if signature == self.signature:
            return False
        self.signature, self.articles = signature, load(self.directory)
        return True

    def __bool__(self) -> bool:
        return bool(self.articles)

    def text(self) -> str:
        parts = []
        for i, a in enumerate(self.articles, 1):
            attrs = " ".join(f'{k}="{_one_line(v)}"' for k, v in
                             (("n", str(i)), ("url", a.url), ("veiculo", a.veiculo), ("titulo", a.titulo),
                              ("data", a.data)) if v)
            parts.append(f"<materia {attrs}>\n{a.texto}\n</materia>")
        return "<base>\n" + "\n\n".join(parts) + "\n</base>"

    def system(self, prompt: str) -> list[dict]:
        return [{"type": "text", "text": prompt},
                {"type": "text", "text": self.text(), "cache_control": {"type": "ephemeral"}}]

    def urls(self) -> set[str]:
        return {norm(a.url) for a in self.articles}

    def by_url(self, url: str) -> Article | None:
        key = norm(url)
        return next((a for a in self.articles if norm(a.url) == key), None)

    async def count_tokens(self, client, model: str) -> int:
        result = await client.beta.messages.count_tokens(
            model=model, system=[{"type": "text", "text": self.text()}], messages=[{"role": "user", "content": "."}])
        return result.input_tokens
