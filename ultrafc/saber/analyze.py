"""Lê uma matéria jornalística e a liga ao que foi dito na transmissão.

Em dois passos: um modelo barato lê a página com web_fetch, sem a transcrição (uma recusa do site custa
quase nada); depois o Sonnet recebe a transcrição e o texto e escreve a análise. Quando o site recusa a
leitura, o texto pode vir colado pelo operador.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

from ..factcheck import usage
from ..factcheck.check import fallback_options, web_tools
from ..factcheck.files import RESULTS, now
from ..logger import TIME_FORMAT, LogLine, parse_line
from .files import Fala, Materia

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
FETCH_MODEL = "claude-haiku-5-5"
MAX_TURNS = 3
FETCH_TURNS = 3
BLOCKED = "url_not_allowed"  # o veículo não permite a leitura por IA (robots.txt ou pedido à Anthropic)
FETCH_TOKENS = 40000  # uma matéria longa cabe inteira
MAX_TRANSCRIPT_CHARS = 600_000  # acima disso, ficam as linhas finais
MAX_FALAS = 3
QUOTE_WINDOW = 3  # um trecho pode atravessar até 3 linhas seguidas do log

FALAS = {
    "type": "array",
    "description": "Até 3 falas da transcrição ligadas ao assunto; vazio se ninguém falou dele.",
    "items": {
        "type": "object",
        "properties": {
            "quem": {"type": "string", "description": "O falante, exatamente como aparece no log."},
            "horario": {"type": "string",
                        "description": "Horário da linha, copiado do log: AAAA-MM-DD HH:MM:SS."},
            "trecho": {"type": "string",
                       "description": "Trecho literal e contínuo da fala, copiado do log, sem cortes "
                                      "nem reticências; até 200 caracteres."},
        },
        "required": ["quem", "horario", "trecho"],
        "additionalProperties": False,
    },
}

SABER_TOOL = {
    "name": "registrar_materia",
    "description": "Registra a análise da matéria. Chame uma única vez, ao terminar.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "relacionada": {"type": "boolean",
                            "description": "true se a matéria tem ligação clara e direta com o que foi dito."},
            "motivo": {"type": "string",
                       "description": "Para o operador: qual é a ligação, ou por que não há ligação."},
            "titulo": {"type": "string", "description": "O título da matéria, como publicado."},
            "veiculo": {"type": "string", "description": "O nome do veículo (ex.: g1, Folha de S.Paulo)."},
            "data_publicacao": {"type": "string",
                                "description": "Data de publicação em DD/MM/AAAA, ou vazio se não aparece."},
            "chamada": {"type": "string",
                        "description": "Até 90 caracteres: o assunto da matéria, para o título de um card."},
            "resumo": {"type": "string",
                       "description": "1 a 2 frases, até 200 caracteres, que se entendam sozinhas: o que a "
                                      "matéria informa e por que importa para o que foi dito. Vai num card."},
            "texto": {"type": "string",
                      "description": "2 a 4 parágrafos curtos, separados por linha em branco: o que a matéria "
                                     "informa, atribuído ao veículo, e a ligação com o que foi dito na transmissão."},
            "falas": FALAS,
        },
        "required": ["relacionada", "motivo", "titulo", "veiculo", "data_publicacao", "chamada", "resumo",
                     "texto", "falas"],
        "additionalProperties": False,
    },
}

SYSTEM = """\
Você ajuda a equipe de um canal de notícias a divulgar matérias jornalísticas e conteúdos informativos \
que dão contexto ao que é dito numa transmissão ao vivo (debate, entrevista, telejornal) em português. \
Você recebe a transcrição automática da transmissão, em linhas no formato \
`[AAAA-MM-DD HH:MM:SS] Falante: texto`, e a URL e o texto de uma matéria. O texto foi extraído da página \
ou colado pelo operador e pode trazer menus, anúncios, legendas e chamadas para outras matérias: use só o \
corpo da matéria.

- Tire da matéria o título, o veículo e a data de publicação; não use outras fontes.
- Encontre na transcrição o que foi dito sobre o assunto da matéria: quem falou, quando e o trecho literal.
- Escreva para o canal um texto que divulgue a matéria e mostre a ligação com o que foi dito:
  - atribua ao veículo o que vem da matéria ("Segundo o g1, …") e aos falantes o que vem da transcrição;
  - seja neutro: não diga se uma fala é verdadeira ou falsa nem quem tem razão; isto não é uma checagem. \
Relate o que a matéria informa e deixe o leitor tirar as conclusões;
  - quando o falante for um rótulo automático (P1, P2…, `?`), escreva "falante não identificado";
  - a transcrição tem erros de reconhecimento: não reproduza números, nomes ou siglas que pareçam \
distorcidos; prefira omitir o detalhe a publicar um erro.
- Marque `relacionada` como false se a matéria não trata do que foi dito, se a ligação for fraca ou \
forçada, ou se o texto recebido não é a matéria (página de erro, de login ou só o começo de um \
texto fechado); explique em `motivo`.
- Ao terminar, chame `registrar_materia`. Não escreva a análise em texto: use a ferramenta.\
"""

CHECK_TOOL = {
    "name": "registrar_checagem",
    "description": "Registra a checagem da agência. Chame uma única vez, ao terminar.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "e_checagem": {"type": "boolean",
                           "description": "true se o texto é uma checagem de fatos com um selo/veredito."},
            "motivo": {"type": "string",
                       "description": "Para o operador: a ligação com a transmissão (ou a falta dela), ou por "
                                      "que o texto não é uma checagem."},
            "agencia": {"type": "string", "description": "O nome da agência (ex.: Aos Fatos, Lupa)."},
            "autoria": {"type": "string",
                        "description": "Os nomes na assinatura, como publicados (\"Ana e Bia\"); vazio se não há."},
            "titulo": {"type": "string", "description": "O título da checagem, como publicado."},
            "data_publicacao": {"type": "string",
                                "description": "Data de publicação em DD/MM/AAAA, ou vazio se não aparece."},
            "afirmacao": {"type": "string",
                          "description": "A afirmação checada, numa frase autocontida de até 160 caracteres, sem "
                                         "contar onde foi dita. Vai entre aspas no card."},
            "quem_afirmou": {"type": "string",
                             "description": "Até 60 caracteres: quem fez a afirmação segundo a agência (o nome da "
                                            "pessoa, \"post viral\", \"perfil no Instagram\")."},
            "selo": {"type": "string",
                     "description": "O selo/etiqueta exato que a agência deu a essa afirmação (ex.: DISTORCIDO)."},
            "categoria": {"type": "string", "enum": list(RESULTS),
                          "description": "A categoria mais próxima do selo da agência; só define a cor do card."},
            "resumo": {"type": "string",
                       "description": "1 a 2 frases, até 200 caracteres, que se entendam sozinhas, atribuídas à "
                                      "agência (\"Segundo o Aos Fatos, …\"), sem repetir a afirmação. "
                                      "Vai num card."},
            "justificativa": {"type": "string",
                              "description": "2 a 4 parágrafos curtos, separados por linha em branco: o que a "
                                             "agência apurou, com os dados e fontes que ela cita, atribuído a ela."},
            "falas": FALAS,
        },
        "required": ["e_checagem", "motivo", "agencia", "autoria", "titulo", "data_publicacao", "afirmacao",
                     "quem_afirmou", "selo", "categoria", "resumo", "justificativa", "falas"],
        "additionalProperties": False,
    },
}

CHECK_SYSTEM = """\
Você ajuda a equipe de um canal de notícias a divulgar checagens de fatos feitas por agências de checagem \
(Aos Fatos, Lupa, Estadão Verifica…), ligando-as, quando possível, ao que é dito numa transmissão ao vivo \
(debate, entrevista, telejornal) em português. Você recebe a transcrição automática da transmissão, em \
linhas no formato `[AAAA-MM-DD HH:MM:SS] Falante: texto`, e a URL e o texto de uma checagem. O texto foi \
extraído da página ou colado pelo operador e pode trazer menus, anúncios e chamadas para outros textos: use \
só o corpo da checagem.

- A apuração é da agência: relate o que ela concluiu e como, sem refazer a checagem, sem acrescentar \
conclusões suas e sem usar outras fontes. Atribua tudo à agência ("Segundo a Lupa, …").
- Copie o selo exatamente como a agência o usa; não o troque por outro. Se a página não tem selo à parte, \
use o termo do veredito no título (ex.: "mente", "engana") e diga isso em `motivo`. Em `categoria`, escolha a mais \
próxima dele: `verdadeiro`, `falso`, `exagerado` (número ou alcance inflado; também "distorcido"), \
`omissão` (correto, mas sem um contexto que muda a conclusão) ou `inconclusivo` (sem provas suficientes; \
também "insustentável", "ainda é cedo para dizer").
- Se a página checa várias afirmações, escolha a que mais se liga ao que foi dito na transmissão; se \
nenhuma se liga, a principal (a do título). Use o selo dessa afirmação.
- Procure na transcrição quem falou do assunto: quem, quando e o trecho literal. Se ninguém falou dele, \
deixe `falas` vazio e diga isso em `motivo`; não force a ligação. `quem_afirmou` é sempre o autor segundo \
a agência, mesmo que outra pessoa tenha repetido a afirmação na transmissão; diga em `motivo` se a fala \
repete a afirmação inteira ou só parte dela.
  - quando o falante for um rótulo automático (P1, P2…, `?`), escreva "falante não identificado";
  - a transcrição tem erros de reconhecimento: não reproduza números, nomes ou siglas que pareçam \
distorcidos.
- Marque `e_checagem` como false se o texto não é uma checagem com selo (matéria comum, página de erro, \
de login ou só o começo de um texto fechado); explique em `motivo`.
- Ao terminar, chame `registrar_checagem`. Não escreva a análise em texto: use a ferramenta.\
"""


def latest_log(source: Path) -> Path | None:
    """O próprio arquivo, ou o `AAAA-MM-DD.log` mais recente de uma pasta de stream."""
    if not source.is_dir():
        return source if source.exists() else None
    logs = sorted(source.glob("????-??-??.log"))
    return logs[-1] if logs else None


def load_transcript(source: Path) -> list[LogLine]:
    path = latest_log(Path(source))
    if path is None:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line for raw in text.splitlines() if (line := parse_line(raw)) is not None]


def transcript_text(lines: list[LogLine], limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    out, size = [], 0
    for line in reversed(lines):
        s = str(line)
        if size + len(s) + 1 > limit:
            break
        out.append(s)
        size += len(s) + 1
    return "\n".join(reversed(out))


def build_content(lines: list[LogLine], url: str, article: dict) -> list[dict]:
    """Transcrição primeiro e em cache: várias URLs na mesma transmissão reaproveitam o prefixo."""
    return [{"type": "text", "text": f"Transcrição da transmissão:\n\n{transcript_text(lines)}",
             "cache_control": {"type": "ephemeral"}},
            article,
            {"type": "text", "text": f"Matéria: {url}"}]


def pasted_document(text: str) -> dict:
    return {"type": "document", "title": "Texto da matéria, colado pelo operador",
            "source": {"type": "text", "media_type": "text/plain", "data": text.strip()}}


def _document(result, url: str) -> dict:
    """O documento devolvido pelo web_fetch (texto ou PDF), reenviado como está ao modelo da análise."""
    doc = result.content
    source = doc.source if isinstance(doc.source, dict) else doc.source.model_dump(exclude_none=True)
    return {"type": "document", "title": (getattr(doc, "title", None) or url)[:500], "source": source}


async def fetch_article(client, model: str, url: str) -> tuple[dict | None, str]:
    """Lê a página com web_fetch; devolve (documento, "") ou (None, código do erro)."""
    host = urlparse(url).hostname
    if not host:
        return None, "URL inválida"
    messages: list = [{"role": "user", "content": f"Leia esta página com web_fetch e responda só OK: {url}"}]
    tools = web_tools(model, 0, 1, allowed_domains=[host], max_content_tokens=FETCH_TOKENS)
    for _ in range(FETCH_TURNS):
        response = await client.beta.messages.create(
            model=model,
            max_tokens=2000,
            **fallback_options(model),
            output_config={"effort": "low"},
            tools=tools,
            messages=messages,
        )
        usage.record(model, response.usage)
        for block in response.content:
            if block.type == "web_fetch_tool_result":
                if getattr(block.content, "type", "") == "web_fetch_result":
                    return _document(block.content, url), ""
                return None, getattr(block.content, "error_code", None) or "erro desconhecido"
        if response.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": response.content})
    return None, "a página não foi lida"


def _norm_text(text: str) -> str:
    """Compara falas sem caixa, acentos, pontuação nem espaços extras."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def _clock(ts: float) -> str:
    return time.strftime(TIME_FORMAT, time.localtime(ts))


def match_fala(fala: Fala, lines: list[LogLine]) -> Fala | None:
    """Acha o trecho na transcrição e devolve a fala com quem e horário do log; None se não estiver lá.

    O trecho pode atravessar linhas seguidas; com mais de uma ocorrência, prefere a do horário informado.
    """
    quote = _norm_text(fala.trecho)
    if not quote:
        return None
    found = []
    for i, line in enumerate(lines):
        first = _norm_text(line.text)
        window = " ".join([first] + [_norm_text(l.text) for l in lines[i + 1:i + QUOTE_WINDOW]])
        if 0 <= window.find(quote) < len(first):  # o trecho começa nesta linha
            found.append(line)
    if not found:
        return None
    best = next((l for l in found if _clock(l.ts) == fala.horario.strip()), found[0])
    return Fala(best.speaker or "?", _clock(best.ts), " ".join(fala.trecho.strip().strip('"“”«»…').split()))


def _error(m: Materia, model: str, reason: str) -> Materia:
    m.erro, m.modelo, m.analisado_em = reason, model, now()
    return m


async def analyze(client, model: str, item_id: str, url: str, lines: list[LogLine], article: dict,
                  stream: str = "", kind: str = "materia") -> Materia:
    """`article` é o documento da matéria: o de `fetch_article` ou o de `pasted_document`.

    `kind` "checagem": o texto é a checagem de uma agência, registrada com `CHECK_TOOL`.
    """
    m = Materia(item_id, url, stream, tipo=kind)
    tool, system, finish = ((CHECK_TOOL, CHECK_SYSTEM, _finish_check) if kind == "checagem"
                            else (SABER_TOOL, SYSTEM, _finish))
    messages: list = [{"role": "user", "content": build_content(lines, url, article)}]
    for _ in range(MAX_TURNS):
        response = await client.beta.messages.create(
            model=model,
            max_tokens=16000,
            **fallback_options(model),
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=system,
            tools=[tool],
            messages=messages,
        )
        usage.record(model, response.usage)
        if response.stop_reason == "refusal":
            return _error(m, model, f"o modelo recusou a análise ({getattr(response.stop_details, 'category', None)})")

        call = next((b for b in response.content if b.type == "tool_use" and b.name == tool["name"]), None)
        if call is not None:
            return finish(m, model, call.input, lines)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content":
                         f"Registre agora a análise chamando a ferramenta {tool['name']}."})
    return _error(m, model, f"nenhuma análise registrada após {MAX_TURNS} turnos")


def _match_falas(m: Materia, data: dict, lines: list[LogLine]) -> None:
    for f in data.get("falas", [])[:MAX_FALAS]:
        fala = Fala(f["quem"], f["horario"], f["trecho"])
        if (found := match_fala(fala, lines)) is None:
            log.warning("trecho não encontrado na transcrição, descartado: %s — %r", fala.quem, fala.trecho)
        else:
            m.falas.append(found)


def _finish(m: Materia, model: str, data: dict, lines: list[LogLine]) -> Materia:
    m.modelo, m.analisado_em = model, now()
    m.relacionada, m.motivo = bool(data["relacionada"]), data["motivo"]
    m.titulo, m.veiculo, m.data_publicacao = data["titulo"], data["veiculo"], data["data_publicacao"]
    m.chamada, m.resumo, m.texto = data["chamada"], data["resumo"], data["texto"]
    _match_falas(m, data, lines)
    if m.relacionada and not m.falas:
        m.relacionada = False
        m.motivo = f"nenhuma fala citada foi encontrada na transcrição. {m.motivo}".strip()
    return m


def _finish_check(m: Materia, model: str, data: dict, lines: list[LogLine]) -> Materia:
    """Checagem de agência: a fala da transmissão é um extra; sem ela, a checagem sai do mesmo jeito."""
    m.modelo, m.analisado_em = model, now()
    m.relacionada, m.motivo = bool(data["e_checagem"]), data["motivo"]
    m.titulo, m.veiculo, m.autoria = data["titulo"], data["agencia"], data["autoria"]
    m.data_publicacao, m.afirmacao, m.quem_afirmou = data["data_publicacao"], data["afirmacao"], data["quem_afirmou"]
    m.selo, m.categoria = data["selo"], data["categoria"]
    m.resumo, m.texto = data["resumo"], data["justificativa"]
    _match_falas(m, data, lines)
    if m.relacionada and not m.selo.strip():
        m.relacionada = False
        m.motivo = f"a checagem não tem selo. {m.motivo}".strip()
    return m
