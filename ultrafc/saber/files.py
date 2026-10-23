"""A matéria (ou checagem de agência) analisada, gravada em checking/saber/<id>.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..factcheck.files import _frontmatter, _one_line, _sections

_UNNAMED = re.compile(r"^(P\d+|\?)?$")


@dataclass
class Fala:
    quem: str
    horario: str  # "AAAA-MM-DD HH:MM:SS", como no log
    trecho: str


@dataclass
class Materia:
    id: str
    url: str
    stream: str = ""
    relacionada: bool = False
    motivo: str = ""  # por que a matéria se liga (ou não) ao que foi dito; só para o operador
    titulo: str = ""
    veiculo: str = ""
    data_publicacao: str = ""
    chamada: str = ""  # título curto para o card
    resumo: str = ""  # 1 a 2 frases para o card
    texto: str = ""
    falas: list[Fala] = field(default_factory=list)
    modelo: str = ""
    analisado_em: str = ""
    erro: str = ""
    origem: str = ""  # de onde veio o texto da matéria: "web" (web_fetch) ou "colado" (pelo operador)
    # Checagem de agência (tipo "checagem"): a agência fica em `veiculo` e a justificativa em `texto`.
    tipo: str = "materia"
    autoria: str = ""  # nomes na assinatura da checagem
    afirmacao: str = ""
    quem_afirmou: str = ""  # segundo a agência
    selo: str = ""  # o rótulo exato da agência
    categoria: str = ""  # um de factcheck.files.RESULTS, só para a cor


def speaker_name(quem: str | None) -> str:
    """Rótulo automático (P1, P2…, `?`) não é nome: vai para o canal como falante não identificado."""
    return "Falante não identificado" if _UNNAMED.match(quem or "") else quem


def render_materia(m: Materia) -> str:
    check = {"autoria": m.autoria, "quem_afirmou": m.quem_afirmou, "selo": m.selo,
             "categoria": m.categoria} if m.tipo == "checagem" else {}
    fm = _frontmatter({"id": m.id, "tipo": m.tipo, "url": m.url, "stream": m.stream,
                       "relacionada": "sim" if m.relacionada else "não", "veiculo": m.veiculo,
                       "data_publicacao": m.data_publicacao, "modelo": m.modelo, "analisado_em": m.analisado_em,
                       "origem": m.origem} | check | ({"erro": m.erro} if m.erro else {}))
    falas = "".join(f"- [{f.horario}] {_one_line(f.quem)}: \"{_one_line(f.trecho)}\"\n" for f in m.falas)
    return (fm + f"\n# {_one_line(m.titulo) or m.url}\n"
            + (f"\n## Afirmação\n{_one_line(m.afirmacao)}\n" if m.afirmacao.strip() else "")
            + f"\n## Chamada\n{_one_line(m.chamada)}\n"
            + f"\n## Resumo\n{_one_line(m.resumo)}\n"
            + f"\n## Texto\n{m.texto.strip()}\n"
            + f"\n## Falas\n{falas or 'Nenhuma.'}\n"
            + f"\n## Motivo\n{m.motivo.strip()}\n")


_FALA = re.compile(r'^- \[(.*?)\] (.*?): "(.*)"$')


def parse_materia(text: str) -> Materia:
    fields, sections = _sections(text)
    known = {"Afirmação", "Chamada", "Resumo", "Texto", "Falas", "Motivo"}
    titulo = next((t for t in sections if t not in known), "")
    falas = [Fala(m.group(2), m.group(1), m.group(3)) for line in sections.get("Falas", "").splitlines()
             if (m := _FALA.match(line.strip()))]
    return Materia(fields["id"], fields.get("url", ""), fields.get("stream", ""), fields.get("relacionada") == "sim",
                   sections.get("Motivo", ""), titulo, fields.get("veiculo", ""), fields.get("data_publicacao", ""),
                   sections.get("Chamada", ""), sections.get("Resumo", ""), sections.get("Texto", ""), falas,
                   fields.get("modelo", ""), fields.get("analisado_em", ""), fields.get("erro", ""),
                   fields.get("origem", ""), fields.get("tipo", "materia"), fields.get("autoria", ""),
                   sections.get("Afirmação", ""), fields.get("quem_afirmou", ""), fields.get("selo", ""),
                   fields.get("categoria", ""))
