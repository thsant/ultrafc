import asyncio
import json
import time
from types import SimpleNamespace as NS

import pytest

from ultrafc.factcheck import check as checker
from ultrafc.factcheck import review as reviewer
from ultrafc.factcheck import usage
from ultrafc.factcheck.__main__ import FactChecker, parse_args
from ultrafc.factcheck.detect import Afirmacao, Deteccao
from ultrafc.factcheck.files import (Claim, Review, Source, Verdict, parse_claim, parse_review, parse_verdict, pending,
                                    render_claim, render_verdict)
from ultrafc.factcheck.tail import LogTail
from ultrafc.logger import format_line

T0 = time.mktime((2026, 10, 22, 22, 11, 35, 0, 0, -1))


def line(i, who, text):
    return format_line(T0 + i, text, who) + "\n"


def claim(**kw):
    base = dict(id="20261022-221135-x", stream="band", quem="Lula", horario="2026-10-22 22:11:35",
                tipo="votação", afirmacao="Flávio foi o único de 49 deputados a votar contra.",
                trecho="você foi o único que votou contra", rationale="Voto registrado e verificável.")
    base.update(kw)
    return Claim(**base)


# --- arquivos ---------------------------------------------------------------------------------

def test_claim_roundtrip_and_pending(tmp_path):
    c = claim()
    (tmp_path / "to_check").mkdir()
    (tmp_path / "to_check" / f"{c.id}.md").write_text(render_claim(c))
    assert parse_claim(render_claim(c)) == c
    assert pending(tmp_path) == [c]
    (tmp_path / "checked").mkdir()
    (tmp_path / "checked" / f"{c.id}.md").write_text("x")
    assert pending(tmp_path) == []
    (tmp_path / "checked" / f"{c.id}.md").unlink()
    (tmp_path / "rejected").mkdir()
    (tmp_path / "rejected" / f"{c.id}.md").write_text("x")
    assert pending(tmp_path) == []  # reprovada não é checada de novo


def test_verdict_file_has_required_sections():
    v = Verdict("falso", "Registros mostram outro resultado.",
                [Source("Alerj", "https://alerj.rj.gov.br/x", "lista de votação"),
                 Source("Blog", "https://blog/y", "repete a fala", consultada=False)],
                "claude-opus-5-5", "2026-10-22 22:13:02")
    md = render_verdict(claim(), v)
    assert "resultado: falso" in md and "## Resultado: FALSO" in md
    assert "quem: Lula" in md and "# Afirmação\nFlávio foi o único" in md
    assert "1. [Alerj](https://alerj.rj.gov.br/x) — lista de votação" in md
    assert "(não consultada pelo agente)" in md.splitlines()[-1]
    assert parse_review(md) is None


def test_verdict_summary_roundtrip_and_in_tool_and_review_prompt():
    v = Verdict("falso", "Justificativa longa.", [], "m", "t", "Resumo curto para o card.")
    md = render_verdict(claim(), v)
    assert "## Resumo\nResumo curto para o card." in md and parse_verdict(md)[1] == v
    old = render_verdict(claim(), Verdict("falso", "J", [], "m", "t"))
    assert "## Resumo" not in old and parse_verdict(old)[1].resumo == ""
    assert "resumo" in checker.VERDICT_TOOL["input_schema"]["required"]
    assert "Resumo para o card: Resumo curto para o card." in reviewer.build_prompt(claim(), v)


def test_verdict_with_review_roundtrip():
    v = Verdict("falso", "Registros mostram outro resultado.",
                [Source("Alerj", "https://alerj.rj.gov.br/x", "lista de votação")], "claude-haiku-5-5", "t1")
    r = Review(False, "A lista mostra 3 votos contra, não 1.", "claude-opus-5-5", "t2")
    md = render_verdict(claim(), v, r)
    assert "revisao: reprovada" in md and "## Revisão\nA lista mostra" in md
    assert parse_review(md) == r
    assert parse_verdict(md)[1] == v


# --- leitura do log ---------------------------------------------------------------------------

def test_tail_reads_only_complete_new_lines_and_resumes(tmp_path):
    log = tmp_path / "logs" / "2026-10-22.log"
    log.parent.mkdir()
    log.write_text(line(0, "Lula", "um") + "[2026-10-22 22:11:3")  # última linha incompleta
    state = tmp_path / "state.json"
    tail = LogTail(log.parent, state, from_start=True)
    assert [l.text for l in tail.read()] == ["um"]
    with open(log, "a") as f:
        f.write("6] Lula: dois\n")
    assert [l.text for l in tail.read()] == ["dois"]
    tail.commit()
    with open(log, "a") as f:
        f.write(line(2, "P2", "três"))
    again = LogTail(log.parent, state)  # reinício: continua do estado salvo
    assert [(l.speaker, l.text) for l in again.read()] == [("P2", "três")]


def test_tail_without_state_starts_at_end_and_follows_new_day(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    (d / "2026-10-22.log").write_text(line(0, "Lula", "antigo"))
    tail = LogTail(d, tmp_path / "s.json")
    assert tail.read() == []
    (d / "2026-10-23.log").write_text(line(90000, "Lula", "novo dia"))
    assert [l.text for l in tail.read()] == ["novo dia"]


# --- agente de checagem -----------------------------------------------------------------------

def response(stop, *blocks):
    return NS(stop_reason=stop, stop_details=None, content=list(blocks), usage=NS(input_tokens=1))


def search_result(*urls):
    return NS(type="web_search_tool_result", content=[NS(type="web_search_result", url=u, title="t") for u in urls])


def review_call(aprovado, motivo="conferido"):
    return NS(type="tool_use", name="registrar_revisao", id="tu2", input={"aprovado": aprovado, "motivo": motivo})


def verdict_call(resultado, *urls):
    return NS(type="tool_use", name="registrar_veredito", id="tu1",
              input={"resultado": resultado, "justificativa": "ok",
                     "fontes": [{"titulo": "f", "url": u, "o_que_sustenta": "x"} for u in urls]})


class FakeMessages:
    def __init__(self, creates=(), parses=(), reviews=()):
        self.creates, self.parses, self.reviews, self.calls = list(creates), list(parses), list(reviews), []
        self.active = self.peak = 0

    async def create(self, **kw):
        self.calls.append(kw)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        reviewing = any(t.get("name") == "registrar_revisao" for t in kw["tools"])
        if reviewing:
            item = self.reviews.pop(0) if self.reviews else response("tool_use", review_call(True))
        else:
            item = self.creates.pop(0) if self.creates else response("tool_use", verdict_call("verdadeiro"))
        if isinstance(item, Exception):
            raise item
        return item

    async def parse(self, **kw):
        self.calls.append(kw)
        found = self.parses.pop(0) if self.parses else []
        return NS(stop_reason="end_turn", stop_details=None, parsed_output=Deteccao(afirmacoes=found),
                  usage=NS(input_tokens=1))


def client(**kw):
    return NS(beta=NS(messages=FakeMessages(**kw)))


def test_check_handles_pause_turn_and_flags_unconsulted_sources():
    c = client(creates=[
        response("pause_turn", search_result("https://alerj.rj.gov.br/votacao/")),
        response("tool_use", verdict_call("falso", "https://alerj.rj.gov.br/votacao", "https://inventada.com")),
    ])
    v = asyncio.run(checker.check(c, "claude-opus-5-5", claim()))
    assert v.resultado == "falso"
    assert [s.consultada for s in v.fontes] == [True, False]
    second = c.beta.messages.calls[1]["messages"]
    assert second[-1]["role"] == "assistant"  # retomada do pause_turn sem mensagem extra
    tools = {t.get("type", t["name"]) for t in c.beta.messages.calls[0]["tools"]}
    assert {"web_search_20260209", "web_fetch_20260209", "registrar_veredito"} <= tools


def test_haiku_check_uses_basic_web_tools_without_fallback():
    c = client()
    asyncio.run(checker.check(c, "claude-haiku-5-5", claim()))
    call = c.beta.messages.calls[0]
    assert "fallbacks" not in call and "betas" not in call
    assert {t.get("type") for t in call["tools"]} >= {"web_search_20250305", "web_fetch_20250910"}


def test_check_without_verdict_becomes_error():
    c = client(creates=[response("end_turn", NS(type="text", text="acho que é falso"))] * checker.MAX_TURNS)
    v = asyncio.run(checker.check(c, "m", claim()))
    assert v.resultado == "erro" and "nenhum veredito" in v.justificativa


# --- revisão -----------------------------------------------------------------------------------

VERDICT = Verdict("falso", "A ata mostra outro placar.",
                  [Source("Ata", "https://alerj.rj.gov.br/ata", "placar da votação"),
                   Source("Blog", "https://blog/y", "repete a fala", consultada=False)], "claude-haiku-5-5", "t")


def test_review_only_fetches_cited_sources():
    c = client(reviews=[response("tool_use", review_call(False, "a ata mostra 3 votos contra"))])
    r = asyncio.run(reviewer.review(c, "claude-opus-5-5", claim(), VERDICT))
    assert not r.aprovado and r.motivo == "a ata mostra 3 votos contra" and r.modelo == "claude-opus-5-5"
    call = c.beta.messages.calls[0]
    web = [t for t in call["tools"] if t["name"] != "registrar_revisao"]
    assert web == [{"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2,
                    "max_content_tokens": reviewer.FETCH_TOKENS}]  # sem busca: só as URLs citadas
    prompt = call["messages"][0]["content"]
    assert "https://alerj.rj.gov.br/ata" in prompt and "(não consultada pelo checador)" in prompt
    assert call["fallbacks"] == "default"


@pytest.mark.parametrize("answer, reason", [
    (response("refusal"), "recusou"),
    (response("end_turn", NS(type="text", text="parece certo")), "nenhuma revisão"),
])
def test_review_without_conclusion_rejects(answer, reason):
    c = client(reviews=[answer] * reviewer.MAX_TURNS)
    r = asyncio.run(reviewer.review(c, "claude-opus-5-5", claim(), VERDICT))
    assert not r.aprovado and reason in r.motivo


# --- orquestração -----------------------------------------------------------------------------

def found(i, texto):
    return Afirmacao(afirmacao=texto, quem="Lula", horario=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(T0 + i)),
                     trecho_original=texto, rationale="r", tipo="número/estatística")


def run_once(tmp_path, c, extra=()):
    logs = tmp_path / "logs" / "band"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "2026-10-22.log").write_text("".join(line(i, "Lula", f"fala {i}") for i in range(12)))
    args = parse_args(["--log-dir", str(logs), "--out", str(tmp_path / "checking"), "--from-start", "--once",
                       *extra])
    from rich.console import Console
    asyncio.run(FactChecker(args, c, Console(file=open("/dev/null", "w"))).run())
    return tmp_path / "checking"


def test_pipeline_writes_to_check_and_checked_with_parallel_limit(tmp_path):
    claims = [found(i, f"Afirmação número {i} com dado verificável") for i in range(5)]
    c = client(parses=[claims[:3], claims[3:]], creates=[])
    out = run_once(tmp_path, c)
    todo = sorted(p.name for p in (out / "to_check").glob("*.md"))
    done = sorted(p.name for p in (out / "checked").glob("*.md"))
    assert len(todo) == 5 and todo == done
    assert c.beta.messages.peak <= 3  # no máximo 3 checagens simultâneas
    # a 2ª janela recebeu as afirmações já registradas, para não repetir
    parse_calls = [k for k in c.beta.messages.calls if "output_format" in k]
    assert "<ja_registradas>" in parse_calls[1]["messages"][0]["content"]
    assert "<contexto_anterior>" in parse_calls[1]["messages"][0]["content"]


def test_rejected_checks_go_to_rejected_and_usage_is_saved(tmp_path):
    claims = [found(i, f"Afirmação {i}") for i in range(3)]
    c = client(parses=[claims],
               reviews=[response("tool_use", review_call(True)), response("tool_use", review_call(False, "não")),
                        RuntimeError("API fora do ar")])
    out = run_once(tmp_path, c, ["--max-parallel", "1"])
    assert len(list((out / "checked").glob("*.md"))) == 1
    rejected = sorted((out / "rejected").glob("*.md"))
    assert len(rejected) == 2
    assert "falha na revisão: API fora do ar" in rejected[1].read_text()
    [snapshot] = (out / "usage").glob("band-*.json")
    data = json.loads(snapshot.read_text())
    assert (data["checagens"], data["aprovadas"], data["rejeitadas"]) == (3, 1, 2)
    assert set(data["modelos"]) >= {"claude-haiku-5-5", "claude-opus-5-5"}


def test_usage_cost_includes_searches():
    t = usage.collections.Counter(input_tokens=1_000_000, output_tokens=1_000_000, buscas=10)
    assert usage.cost("claude-haiku-5-5", t) == pytest.approx(0.10 + 0.50 + 0.10)


def test_stop_after_limits_claims_and_failure_is_recorded(tmp_path):
    claims = [found(i, f"Afirmação {i}") for i in range(4)]
    c = client(parses=[claims], creates=[RuntimeError("API fora do ar")])
    out = run_once(tmp_path, c, ["--stop-after", "2"])
    done = sorted((out / "checked").glob("*.md"))
    assert len(done) == 2
    results = sorted(d.read_text().split("resultado: ")[1].split("\n")[0] for d in done)
    assert results == ["erro", "verdadeiro"]


def test_id_keeps_accented_letters_and_is_unique(tmp_path):
    from ultrafc.factcheck.files import new_id
    first = new_id(T0, "Mais de 50 milhões votaram em Flávio", tmp_path)
    assert first == "20261022-221135-mais-de-50-milhoes-votaram-em-flavio"
    (tmp_path / f"{first}.md").write_text("x")
    assert new_id(T0, "Mais de 50 milhões votaram em Flávio", tmp_path) == first + "-2"


def test_verdict_tool_offers_new_results():
    enum = checker.VERDICT_TOOL["input_schema"]["properties"]["resultado"]["enum"]
    assert enum == ["verdadeiro", "falso", "exagerado", "omissão", "inconclusivo"]
    for r in enum:
        assert f"`{r}`" in checker.SYSTEM
    md = render_verdict(claim(), Verdict("omissão", "x", [], "m", "t"))
    assert "resultado: omissão" in md and "## Resultado: OMISSÃO" in md
