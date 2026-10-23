"""Detecção de afirmações checáveis na transcrição (LLM com raciocínio + saída estruturada)."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from ..logger import LogLine
from . import usage

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

Tipo = Literal["número/estatística", "votação", "lei/norma", "evento", "citação/atribuição", "histórico", "outro"]


class Afirmacao(BaseModel):
    afirmacao: str = Field(description="A afirmação reescrita de forma autocontida: quem, o quê, quando, onde.")
    quem: str = Field(description="Quem fez a afirmação, exatamente como aparece no log (nome ou rótulo P1/P2/?).")
    horario: str = Field(description="Horário da linha onde a afirmação começa, copiado do log: AAAA-MM-DD HH:MM:SS.")
    trecho_original: str = Field(description="A fala literal da transcrição que contém a afirmação.")
    rationale: str = Field(description="Por que vale checar: o que é verificável e por que importa.")
    tipo: Tipo


class Deteccao(BaseModel):
    afirmacoes: list[Afirmacao]


SYSTEM = """\
Você faz a triagem de afirmações para checagem de fatos em transmissões ao vivo (debates, entrevistas, \
telejornais) em português. Você recebe a transcrição automática, em linhas no formato \
`[AAAA-MM-DD HH:MM:SS] Falante: texto`. O falante é um nome informado pelo operador ou um rótulo \
automático (P1, P2…; `?` quando desconhecido). A transcrição é automática: tolere erros de \
reconhecimento (nomes e números podem vir distorcidos) e não os trate como afirmações.

Extraia apenas afirmações que sejam fatos verificáveis em fontes públicas:
- números e estatísticas (valores, percentuais, rankings, recordes);
- votos, leis, normas, decisões judiciais, atos de governo;
- eventos com data, lugar ou protagonistas definidos;
- atribuições a terceiros (o que alguém disse, fez ou é);
- afirmações históricas.

Não extraia opiniões, promessas, intenções, perguntas, ataques genéricos sem fato concreto, \
metáforas, nem falas de procedimento (regras do debate, chamadas de bloco, agradecimentos).

Para cada afirmação:
- `afirmacao`: reescreva de forma autocontida e precisa, resolvendo pronomes com o contexto \
("você" → o nome do interlocutor, quando o contexto deixar claro);
- `quem`: o falante exatamente como está no log; uma afirmação feita por outra pessoa é outra afirmação;
- `horario`: copie o horário da linha em que a afirmação começa;
- `trecho_original`: a fala literal, sem corrigir;
- `rationale`: uma ou duas frases sobre o que torna a afirmação verificável e relevante.

Você recebe também as afirmações já registradas nesta sessão. Não devolva de novo uma afirmação \
equivalente feita pelo mesmo falante. Analise só o trecho marcado como novo; o contexto anterior serve \
apenas para entender o trecho novo. Se não houver nada checável, devolva a lista vazia: a maioria dos \
trechos não tem afirmações que mereçam checagem, e qualidade importa mais que quantidade.\
"""


def build_prompt(context: list[LogLine], new: list[LogLine], known: list[str]) -> str:
    parts = []
    if known:
        parts.append("<ja_registradas>\n" + "\n".join(f"- {k}" for k in known) + "\n</ja_registradas>")
    if context:
        parts.append("<contexto_anterior>\n" + "\n".join(map(str, context)) + "\n</contexto_anterior>")
    parts.append("<trecho_novo>\n" + "\n".join(map(str, new)) + "\n</trecho_novo>")
    return "\n\n".join(parts)


async def detect(client, model: str, context: list[LogLine], new: list[LogLine],
                 known: list[str]) -> list[Afirmacao]:
    response = await client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_prompt(context, new, known)}],
        output_format=Deteccao,
    )
    usage.record(model, response.usage)
    if response.stop_reason == "refusal":
        log.warning("detecção recusada pelo modelo (%s); trecho ignorado",
                    getattr(response.stop_details, "category", None))
        return []
    if response.parsed_output is None:
        log.warning("detecção sem saída estruturada (stop_reason=%s)", response.stop_reason)
        return []
    log.debug("detecção: %d afirmação(ões), %s tokens de entrada", len(response.parsed_output.afirmacoes),
              response.usage.input_tokens)
    return response.parsed_output.afirmacoes
