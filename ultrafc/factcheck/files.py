"""Arquivos .md da checagem: pendências em to_check/, resultados aprovados em checked/ e reprovados em rejected/."""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ..logger import TIME_FORMAT
from ..source import slugify

RESULTS = ("verdadeiro", "falso", "exagerado", "omissão", "inconclusivo")


@dataclass
class Claim:
    id: str
    stream: str
    quem: str
    horario: str  # "AAAA-MM-DD HH:MM:SS", como no log
    tipo: str
    afirmacao: str
    trecho: str
    rationale: str


@dataclass
class Source:
    titulo: str
    url: str
    o_que_sustenta: str
    consultada: bool = True


@dataclass
class Verdict:
    resultado: str  # um de RESULTS, ou "erro"
    justificativa: str
    fontes: list[Source] = field(default_factory=list)
    modelo: str = ""
    checado_em: str = ""
    resumo: str = ""  # 1 a 2 frases para o card; vazio em checagens antigas


@dataclass
class Review:
    aprovado: bool
    motivo: str
    modelo: str = ""
    revisado_em: str = ""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _ascii(text: str) -> str:
    """"milhões" → "milhoes": tira acentos em vez de perder a letra no slug."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def new_id(claim_ts: float, afirmacao: str, directory: Path) -> str:
    base = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(claim_ts))}-{slugify(_ascii(afirmacao), 50)}"
    candidate, n = base, 2
    while (directory / f"{candidate}.md").exists():
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _frontmatter(fields: dict[str, str]) -> str:
    return "---\n" + "".join(f"{k}: {_one_line(v)}\n" for k, v in fields.items()) + "---\n"


def _claim_body(c: Claim) -> str:
    return (f"# Afirmação\n{c.afirmacao.strip()}\n\n"
            f"> \"{_one_line(c.trecho)}\" — {c.quem}, {c.horario[-8:]}\n")


def render_claim(c: Claim) -> str:
    return (_frontmatter({"id": c.id, "stream": c.stream, "quem": c.quem, "horario": c.horario, "tipo": c.tipo})
            + "\n" + _claim_body(c) + f"\n## Por que checar\n{c.rationale.strip()}\n")


def render_verdict(c: Claim, v: Verdict, r: Review | None = None) -> str:
    fields = {"id": c.id, "stream": c.stream, "quem": c.quem, "horario": c.horario,
              "resultado": v.resultado, "checado_em": v.checado_em, "modelo": v.modelo}
    if r is not None:
        fields |= {"revisao": "aprovada" if r.aprovado else "reprovada", "revisor": r.modelo,
                   "revisado_em": r.revisado_em}
    fm = _frontmatter(fields)
    fontes = "\n".join(
        f"{i}. [{_one_line(s.titulo) or s.url}]({s.url}) — {_one_line(s.o_que_sustenta)}"
        + ("" if s.consultada else " *(não consultada pelo agente)*")
        for i, s in enumerate(v.fontes, 1)) or "Nenhuma."
    return (fm + "\n" + _claim_body(c)
            + f"\n## Resultado: {v.resultado.upper()}\n"
            + (f"\n## Resumo\n{v.resumo.strip()}\n" if v.resumo.strip() else "")
            + f"\n## Justificativa\n{v.justificativa.strip()}\n"
            + f"\n## Fontes consultadas\n{fontes}\n"
            + ("" if r is None else f"\n## Revisão\n{r.motivo.strip()}\n"))


def _sections(text: str) -> tuple[dict[str, str], dict[str, str]]:
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not m:
        raise ValueError("arquivo sem frontmatter")
    fields = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
    sections: dict[str, str] = {}
    for part in re.split(r"^#{1,2} ", m.group(2), flags=re.M)[1:]:
        title, _, body = part.partition("\n")
        sections[title.strip()] = body.strip()
    return fields, sections


def parse_claim(text: str) -> Claim:
    fields, sections = _sections(text)
    body = sections.get("Afirmação", "")
    afirmacao, _, quote = body.partition("\n\n> ")
    trecho = re.sub(r'^"(.*)" — .*$', r"\1", quote.strip(), flags=re.S)
    return Claim(fields["id"], fields.get("stream", ""), fields.get("quem", ""), fields.get("horario", ""),
                 fields.get("tipo", ""), afirmacao.strip(), trecho, sections.get("Por que checar", ""))


_SOURCE = re.compile(r"^\d+\. \[(.*?)\]\((.*?)\) — (.*?)( \*\(não consultada pelo agente\)\*)?$")


def parse_verdict(text: str) -> tuple[Claim, Verdict]:
    """Inverso de `render_verdict` (sem a revisão; ver `parse_review`)."""
    fields, sections = _sections(text)
    claim = parse_claim(text)
    fontes = [Source(m.group(1), m.group(2), m.group(3), m.group(4) is None)
              for line in sections.get("Fontes consultadas", "").splitlines()
              if (m := _SOURCE.match(line.strip()))]
    verdict = Verdict(fields.get("resultado", ""), sections.get("Justificativa", ""), fontes,
                      fields.get("modelo", ""), fields.get("checado_em", ""), sections.get("Resumo", ""))
    return claim, verdict


def parse_review(text: str) -> Review | None:
    fields, sections = _sections(text)
    if "revisao" not in fields:
        return None
    return Review(fields["revisao"] == "aprovada", sections.get("Revisão", ""), fields.get("revisor", ""),
                  fields.get("revisado_em", ""))


def pending(out: Path) -> list[Claim]:
    """Afirmações em to_check/ que ainda não têm resultado em checked/ nem em rejected/, em ordem de id."""
    todo, done = out / "to_check", (out / "checked", out / "rejected")
    if not todo.exists():
        return []
    return [parse_claim(p.read_text(encoding="utf-8")) for p in sorted(todo.glob("*.md"))
            if not any((d / p.name).exists() for d in done)]


def now() -> str:
    return time.strftime(TIME_FORMAT)
