"""Revisor: confere um veredito relendo só as fontes que o checador citou; reprovado não é publicado."""

from __future__ import annotations

import logging

from . import usage
from .base import Base
from .check import fallback_options, web_tools
from .files import Claim, Review, Verdict, now

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TURNS = 6
FETCH_TOKENS = 5000  # por página: reler as fontes é o que mais pesa no custo da revisão

REVIEW_TOOL = {
    "name": "registrar_revisao",
    "description": "Registra a conclusão da revisão. Chame uma única vez, ao terminar.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "aprovado": {"type": "boolean",
                         "description": "true se as fontes listadas sustentam o resultado e a justificativa."},
            "motivo": {"type": "string",
                       "description": "Em português, objetivo: o que foi conferido e, se reprovado, o problema."},
        },
        "required": ["aprovado", "motivo"],
        "additionalProperties": False,
    },
}

SYSTEM = """\
Você revisa o trabalho de um checador de fatos. Recebe uma afirmação feita numa transmissão ao vivo, o \
resultado que o checador atribuiu, a justificativa e o resumo dele e as fontes que ele citou. Sua tarefa é dizer se \
essa checagem pode ser publicada.

- Use só as fontes listadas. Você pode reler cada uma com `web_fetch`; não pesquise outras fontes. Fontes \
marcadas como não consultadas pelo checador não contam como evidência. Fontes da base da equipe vêm \
anexadas como documentos: confira-as pelo texto anexado, sem `web_fetch`.
- Confira se cada fonte diz o que o checador afirma que ela diz, e se o conjunto sustenta o resultado \
escolhido, inclusive a escolha entre `verdadeiro`, `falso`, `exagerado`, `omissão` e `inconclusivo`.
- O resumo vai sozinho num card para redes sociais: confira se ele, lido isoladamente, é fiel às fontes.
- Aprove se a conclusão se sustenta com essas fontes, ainda que a redação possa melhorar. Reprove se uma \
fonte foi mal lida, se a conclusão ou o resumo vão além do que as fontes mostram, se o resultado deveria ser \
outro, ou se não foi possível ler fontes suficientes para confirmar.
- Ao terminar, chame `registrar_revisao`. Não reescreva o veredito.\
"""


def build_prompt(c: Claim, v: Verdict, base: Base | None = None) -> str:
    def note(s) -> str:
        if base and base.by_url(s.url):
            return "\n   (da base da equipe: texto anexado)"
        return "" if s.consultada else "\n   (não consultada pelo checador)"

    fontes = "\n".join(f"{i}. {s.titulo} — {s.url}\n   O checador diz que mostra: {s.o_que_sustenta}" + note(s)
                       for i, s in enumerate(v.fontes, 1)) or "Nenhuma."
    return (f"Afirmação: {c.afirmacao}\n"
            f"Quem disse: {c.quem}\n"
            f"Quando: {c.horario}\n"
            f"Fala transcrita: \"{c.trecho}\"\n\n"
            f"Resultado do checador: {v.resultado}\n"
            f"Justificativa: {v.justificativa}\n"
            f"Resumo para o card: {v.resumo or '(nenhum)'}\n\n"
            f"Fontes citadas:\n{fontes}")


def _rejected(model: str, reason: str) -> Review:
    return Review(False, reason, model, now())


def base_documents(verdict: Verdict, base: Base | None) -> list[dict]:
    """As matérias da base citadas pelo checador; o revisor não conseguiria relê-las na web (Folha, g1…)."""
    found = {a.arquivo: a for s in verdict.fontes if base and (a := base.by_url(s.url))}
    return [{"type": "document", "title": f"{a.label or a.arquivo} ({a.url})"[:500],
             "source": {"type": "text", "media_type": "text/plain", "data": a.texto}} for a in found.values()]


async def review(client, model: str, claim: Claim, verdict: Verdict, base: Base | None = None) -> Review:
    docs = base_documents(verdict, base)
    prompt = build_prompt(claim, verdict, base)
    messages: list = [{"role": "user", "content": [*docs, {"type": "text", "text": prompt}] if docs else prompt}]
    web = sum(1 for s in verdict.fontes if not (base and base.by_url(s.url)))
    tools = [*web_tools(model, 0, max(web, 1), max_content_tokens=FETCH_TOKENS), REVIEW_TOOL]
    for _ in range(MAX_TURNS):
        response = await client.beta.messages.create(
            model=model,
            max_tokens=16000,
            **fallback_options(model),
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        usage.record(model, response.usage)
        if response.stop_reason == "refusal":
            return _rejected(model, "o revisor recusou a revisão "
                                    f"({getattr(response.stop_details, 'category', None)})")

        call = next((b for b in response.content
                     if b.type == "tool_use" and b.name == REVIEW_TOOL["name"]), None)
        if call is not None:
            return Review(bool(call.input["aprovado"]), call.input["motivo"], model, now())

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        messages.append({"role": "user", "content":
                         "Registre agora a conclusão chamando a ferramenta registrar_revisao."})
    return _rejected(model, f"nenhuma revisão registrada após {MAX_TURNS} turnos")
