"""Resumo neutro de um trecho da transcrição, para publicação após revisão."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from ..factcheck import usage
from ..factcheck.detect import FALLBACK_BETA
from ..factcheck.files import _frontmatter, write_atomic
from ..logger import TIME_FORMAT, LogLine

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"


class Resumo(BaseModel):
    topicos: list[str] = Field(description="De 4 a 8 tópicos curtos, cada um atribuído a quem falou.")


SYSTEM = """\
Você resume, para um canal de notícias, um trecho da transcrição automática de uma transmissão ao vivo \
(debate, entrevista, telejornal) em português. As linhas vêm no formato \
`[AAAA-MM-DD HH:MM:SS] Falante: texto`.

Escreva de 4 a 8 tópicos curtos com o que foi dito de mais relevante no trecho:
- atribua cada ponto a quem falou, pelo nome ("Lula disse que…", "Flávio Bolsonaro afirmou que…");
  quando o falante for um rótulo automático (P1, P2…, `?`), escreva "falante não identificado";
- seja neutro: relate o que foi dito, sem avaliar quem tem razão, sem adjetivos e sem conclusões
  próprias; não diga se uma afirmação é verdadeira ou falsa;
- a transcrição tem erros de reconhecimento: não reproduza números, nomes ou siglas que pareçam
  distorcidos; prefira omitir o detalhe a publicar um erro;
- ignore falas de procedimento (regras, chamadas de bloco, agradecimentos).\
"""


async def summarize(client, model: str, lines: list[LogLine]) -> list[str]:
    response = await client.beta.messages.parse(
        model=model,
        max_tokens=8000,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=SYSTEM,
        messages=[{"role": "user", "content": "\n".join(map(str, lines))}],
        output_format=Resumo,
    )
    usage.record(model, response.usage)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        log.warning("resumo não gerado (stop_reason=%s)", response.stop_reason)
        return []
    return [t.strip() for t in response.parsed_output.topicos if t.strip()]


def summary_id(start: float, end: float) -> str:
    return f"resumo-{time.strftime('%Y%m%d-%H%M%S', time.localtime(start))}-{time.strftime('%H%M%S', time.localtime(end))}"


def save(directory, item_id: str, stream: str, start: float, end: float, topics: list[str]) -> None:
    fm = _frontmatter({"id": item_id, "stream": stream,
                       "inicio": time.strftime(TIME_FORMAT, time.localtime(start)),
                       "fim": time.strftime(TIME_FORMAT, time.localtime(end))})
    write_atomic(directory / f"{item_id}.md", fm + "\n# Resumo\n" + "".join(f"- {t}\n" for t in topics))
