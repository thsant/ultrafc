"""Agente de checagem: pesquisa a afirmação na web e registra um veredito com fontes."""

from __future__ import annotations

import logging
from urllib.parse import urldefrag

from . import usage
from .detect import FALLBACK_BETA
from .files import RESULTS, Claim, Source, Verdict, now

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TURNS = 8  # inclui retomadas de pause_turn e lembretes para registrar o veredito

VERDICT_TOOL = {
    "name": "registrar_veredito",
    "description": "Registra o resultado final da checagem. Chame uma única vez, ao terminar a pesquisa.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "resultado": {"type": "string", "enum": list(RESULTS)},
            "justificativa": {"type": "string",
                              "description": "Explicação objetiva, em português, apoiada nas fontes."},
            "fontes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "titulo": {"type": "string"},
                        "url": {"type": "string"},
                        "o_que_sustenta": {"type": "string",
                                           "description": "O que esta fonte mostra sobre a afirmação."},
                    },
                    "required": ["titulo", "url", "o_que_sustenta"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["resultado", "justificativa", "fontes"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 6},
    VERDICT_TOOL,
]

SYSTEM = """\
Você é um checador de fatos. Recebe uma afirmação feita numa transmissão ao vivo, transcrita \
automaticamente, e deve verificá-la pesquisando na web.

- Seu conhecimento interno pode estar desatualizado em relação à data da afirmação: a conclusão deve \
vir do que você encontrar na pesquisa, não da memória.
- Prefira fontes primárias (registros oficiais, diários oficiais, sites de casas legislativas e \
tribunais, bases de dados públicas) e imprensa profissional. Busque também versões contrárias.
- A transcrição pode ter erros de reconhecimento em nomes e números; considere a leitura mais \
plausível e diga na justificativa se precisou interpretar.
- Resultado:
  - `verdadeiro`: as fontes confirmam a afirmação no essencial;
  - `falso`: as fontes contradizem a afirmação no essencial;
  - `exagerado`: o fato existe, mas a afirmação infla número, intensidade, frequência ou alcance \
(ex.: diz que algo caiu 2 pontos quando caiu 1);
  - `omissão`: o que foi dito está correto, mas deixa de fora um contexto que muda a conclusão \
(ex.: o valor nominal é recorde, mas em proporção do PIB não é);
  - `inconclusivo`: depois de pesquisar, as fontes não bastam para confirmar nem refutar. Diga na \
justificativa o que faltou. Não use para casos em que a evidência aponta claramente para outro resultado.
  Se a afirmação está correta no essencial e só um detalhe secundário diverge, prefira `verdadeiro` e \
aponte o detalhe na justificativa.
- Ao terminar, chame `registrar_veredito` com o resultado, a justificativa e só as fontes que você \
de fato consultou nesta pesquisa (com a URL exata). Não escreva o veredito em texto: use a ferramenta.\
"""


def build_prompt(c: Claim) -> str:
    return (f"Afirmação a checar: {c.afirmacao}\n"
            f"Quem disse: {c.quem}\n"
            f"Quando: {c.horario}\n"
            f"Fala transcrita: \"{c.trecho}\"\n"
            f"Por que foi selecionada: {c.rationale}")


def _norm(url: str) -> str:
    return urldefrag(url)[0].rstrip("/").lower()


def _consulted_urls(content) -> set[str]:
    """URLs que o agente realmente viu: resultados de busca e páginas buscadas."""
    urls = set()
    for block in content:
        if block.type == "web_search_tool_result" and isinstance(block.content, list):
            urls.update(_norm(r.url) for r in block.content if getattr(r, "url", None))
        elif block.type == "web_fetch_tool_result" and getattr(block.content, "url", None):
            urls.add(_norm(block.content.url))
    return urls


def _error(model: str, reason: str) -> Verdict:
    return Verdict("erro", reason, [], model, now())


async def check(client, model: str, claim: Claim) -> Verdict:
    messages: list = [{"role": "user", "content": build_prompt(claim)}]
    consulted: set[str] = set()
    for _ in range(MAX_TURNS):
        response = await client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        usage.record(model, response.usage)
        consulted |= _consulted_urls(response.content)
        if response.stop_reason == "refusal":
            return _error(model, "o modelo recusou a checagem "
                                 f"({getattr(response.stop_details, 'category', None)})")

        call = next((b for b in response.content
                     if b.type == "tool_use" and b.name == VERDICT_TOOL["name"]), None)
        if call is not None:
            data = call.input
            fontes = [Source(f["titulo"], f["url"], f["o_que_sustenta"], _norm(f["url"]) in consulted)
                      for f in data.get("fontes", [])]
            return Verdict(data["resultado"], data["justificativa"], fontes, model, now())

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue  # o servidor retoma a pesquisa a partir do turno pausado
        messages.append({"role": "user", "content":
                         "Registre agora o resultado chamando a ferramenta registrar_veredito."})
    return _error(model, f"nenhum veredito registrado após {MAX_TURNS} turnos")
