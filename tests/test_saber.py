import asyncio
import io
import time
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from ultrafc.logger import format_line, parse_line
from ultrafc.publish.card import COLORS, SABER_COLOR, SIZE, render_agency_card, render_saber_card
from ultrafc.publish.format import MAX_MESSAGE, format_agency_check, format_saber
from ultrafc.publish.review import PUBLISH, Reviewer
from ultrafc.saber import analyze as saber
from ultrafc.saber.__main__ import Saber, item_id, parse_args, receive
from ultrafc.saber.files import Fala, Materia, parse_materia, render_materia, speaker_name

T0 = time.mktime((2026, 10, 22, 22, 11, 35, 0, 0, -1))
URL = "https://g1.globo.com/politica/noticia/2026/10/22/divida-publica.ghtml"


def lines():
    raw = [(0, "Eduardo Oinegue", "Vamos ao próximo bloco."),
           (5, "Flávio Bolsonaro", "A dívida pública chegou a 11 trilhões de reais,"),
           (9, "Flávio Bolsonaro", "um recorde negativo deixado por este governo."),
           (20, "P2", "Isso não é verdade, a dívida caiu em proporção do PIB.")]
    return [parse_line(format_line(T0 + dt, text, who)) for dt, who, text in raw]


def materia(**kw):
    base = dict(id="saber-abc", url=URL, stream="band", relacionada=True, motivo="trata da dívida citada",
                titulo="Dívida pública bate R$ 11 trilhões & <recorde> nominal", veiculo="g1",
                data_publicacao="22/10/2026", chamada="Dívida pública passa de R$ 11 trilhões",
                resumo="O Tesouro registrou R$ 11 tri em setembro; em proporção do PIB, o nível é o mesmo de 2024.",
                texto="Segundo o g1, a dívida chegou a R$ 11 trilhões.\n\nEm proporção do PIB, ficou estável.",
                falas=[Fala("Flávio Bolsonaro", "2026-10-22 22:11:40", "A dívida pública chegou a 11 trilhões")],
                modelo="claude-sonnet-5-5", analisado_em="2026-10-22 22:20:00")
    return Materia(**(base | kw))


# --- falas -------------------------------------------------------------------------------------

def test_quote_is_found_across_lines_and_takes_speaker_and_time_from_log():
    f = saber.match_fala(Fala("Flavio", "2026-10-22 22:11:00", "“11 trilhões de reais, um recorde negativo”"),
                         lines())
    assert f == Fala("Flávio Bolsonaro", "2026-10-22 22:11:40", "11 trilhões de reais, um recorde negativo")


def test_invented_quote_is_not_found():
    assert saber.match_fala(Fala("Flávio Bolsonaro", "2026-10-22 22:11:40", "a dívida dobrou"), lines()) is None


def test_transcript_keeps_last_lines_when_too_long():
    text = saber.transcript_text(lines(), limit=120)
    assert "proporção do PIB" in text and "próximo bloco" not in text


def test_transcript_from_stream_dir_uses_latest_log(tmp_path):
    (tmp_path / "2026-10-21.log").write_text(str(lines()[0]) + "\n")
    (tmp_path / "2026-10-22.log").write_text("\n".join(map(str, lines())) + "\nlixo\n")
    assert len(saber.load_transcript(tmp_path)) == 4


# --- leitura e análise -------------------------------------------------------------------------

def response(stop, *blocks):
    return NS(stop_reason=stop, stop_details=None, content=list(blocks), usage=NS(input_tokens=1))


ARTICLE = "Segundo o Tesouro, a dívida pública chegou a R$ 11 trilhões em setembro. " * 6


def fetch_result(url=URL, error=None):
    if error:
        return NS(type="web_fetch_tool_result", content=NS(type="web_fetch_tool_result_error", error_code=error))
    doc = NS(type="document", title="Dívida pública", source={"type": "text", "media_type": "text/plain", "data": ARTICLE})
    return NS(type="web_fetch_tool_result", content=NS(type="web_fetch_result", url=url, content=doc))


def saber_call(relacionada=True, falas=None):
    falas = falas if falas is not None else [
        {"quem": "Flávio Bolsonaro", "horario": "2026-10-22 22:11:40", "trecho": "A dívida pública chegou a 11 trilhões"}]
    return NS(type="tool_use", name="registrar_materia", id="tu1",
              input={"relacionada": relacionada, "motivo": "m", "titulo": "Dívida", "veiculo": "g1",
                     "data_publicacao": "22/10/2026", "chamada": "c", "resumo": "r", "texto": "t", "falas": falas})


class FakeMessages:
    def __init__(self, creates):
        self.creates, self.calls = list(creates), []

    async def create(self, **kw):
        self.calls.append(kw)
        return self.creates.pop(0)


def client(*creates):
    return NS(beta=NS(messages=FakeMessages(creates)))


def test_fetch_reads_only_the_url_with_cheap_model_and_no_transcript():
    c = client(response("pause_turn"), response("end_turn", fetch_result()))
    doc, error = asyncio.run(saber.fetch_article(c, "claude-haiku-5-5", URL))
    assert error == "" and doc["source"]["data"] == ARTICLE and doc["title"] == "Dívida pública"
    call = c.beta.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5" and "system" not in call
    fetch = next(t for t in call["tools"] if t["name"] == "web_fetch")
    assert fetch["allowed_domains"] == ["g1.globo.com"]
    assert not any(t["name"] == "web_search" for t in call["tools"])


def test_fetch_reports_blocked_site():
    c = client(response("end_turn", fetch_result(error="url_not_allowed")))
    assert asyncio.run(saber.fetch_article(c, "claude-haiku-5-5", URL)) == (None, saber.BLOCKED)


def test_analyze_gets_transcript_cached_then_article_and_no_web_tools():
    c = client(response("tool_use", saber_call()))
    article = saber.pasted_document(ARTICLE)
    m = asyncio.run(saber.analyze(c, "claude-sonnet-5-5", "saber-x", URL, lines(), article, "band"))
    assert m.relacionada and not m.erro and m.falas[0].quem == "Flávio Bolsonaro"
    call = c.beta.messages.calls[0]
    assert [t["name"] for t in call["tools"]] == ["registrar_materia"]
    first, doc, last = call["messages"][0]["content"]
    assert first["cache_control"] == {"type": "ephemeral"} and "11 trilhões" in first["text"]
    assert doc == article and URL in last["text"]


def test_analyze_without_quotes_found_is_not_related():
    bad = [{"quem": "P2", "horario": "2026-10-22 22:11:55", "trecho": "frase que ninguém disse"}]
    c = client(response("tool_use", saber_call(falas=bad)))
    m = asyncio.run(saber.analyze(c, "claude-sonnet-5-5", "saber-x", URL, lines(), saber.pasted_document(ARTICLE)))
    assert not m.relacionada and m.falas == [] and "nenhuma fala" in m.motivo


def test_analyze_without_tool_call_becomes_error():
    c = client(*[response("end_turn", NS(type="text", text="..."))] * saber.MAX_TURNS)
    m = asyncio.run(saber.analyze(c, "m", "saber-x", URL, lines(), saber.pasted_document(ARTICLE)))
    assert "nenhuma análise" in m.erro


# --- arquivo, mensagem e card ------------------------------------------------------------------

def test_materia_file_round_trip():
    m = materia()
    assert parse_materia(render_materia(m)) == m


def test_unnamed_speakers_are_not_published_as_labels():
    assert speaker_name("P3") == speaker_name("?") == speaker_name(None) == "Falante não identificado"
    assert speaker_name("Lula") == "Lula"


def test_saber_message_escapes_html_and_links_the_article():
    html = format_saber(materia(falas=[Fala("P2", "2026-10-22 22:11:55", "a dívida <caiu>")]), footer="Equipe")
    assert html.startswith("🔎 <b>Procure saber!</b>\n<b>Dívida pública bate R$ 11 trilhões &amp; &lt;recorde&gt;")
    assert "🗣 <i>Falante não identificado, 22:11:</i> «a dívida &lt;caiu&gt;»" in html
    assert f'<a href="{URL}">Leia a matéria em g1</a>' in html
    assert html.endswith("#procuresaber\n<i>Equipe</i>")


def test_long_saber_text_fits_telegram():
    html = format_saber(materia(texto="\n\n".join(["palavra " * 300] * 10)))
    assert len(html) <= MAX_MESSAGE and "Leia a matéria" in html


def test_saber_card_is_square_with_band_color():
    for m in (materia(), materia(chamada="x " * 200, resumo="y " * 200, falas=[Fala("P1", "t", "z " * 300)]),
              materia(veiculo="", data_publicacao="", resumo="", falas=[])):
        img = Image.open(io.BytesIO(render_saber_card(m, "@canal", "rodapé")))
        assert img.size == (SIZE, SIZE)
        assert img.getpixel((10, 10)) == Image.new("RGB", (1, 1), SABER_COLOR).getpixel((0, 0))


# --- fluxo -------------------------------------------------------------------------------------

class FakeBot:
    def __init__(self):
        self.sent, self._n = [], 0

    async def send_message(self, chat_id, text, **kw):
        self._n += 1
        self.sent.append((chat_id, text))
        return NS(message_id=self._n)

    async def send_photo(self, chat_id, photo, **kw):
        self._n += 1
        self.sent.append((chat_id, "<foto>"))
        return NS(message_id=self._n)

    async def edit_message_text(self, **kw):
        pass


def setup(tmp_path, *creates):
    log = tmp_path / "logs" / "band" / "2026-10-22.log"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(map(str, lines())) + "\n")
    args = parse_args([str(log.parent), "--channel", "@c", "--review-chat", "42", "--operators", "1",
                       "--checking", str(tmp_path / "checking")])
    bot = FakeBot()
    reviewer = Reviewer(bot, tmp_path / "state.json", 42, "@c", {1})
    c = client(*creates)
    return Saber(reviewer, c, args, "@c", tmp_path / "usage.json"), bot, c


def test_url_is_analyzed_once_and_published_with_card_after_approval(tmp_path):
    s, bot, c = setup(tmp_path, response("end_turn", fetch_result()), response("tool_use", saber_call()))

    async def run():
        await s.handle(URL)
        await s.handle(URL + "#comentarios")  # mesma matéria: não reanalisa
        return await s.reviewer.decide(PUBLISH, s.reviewer.state[item_id(URL)]["key"], 1, "Op")

    assert asyncio.run(run()) == "✅ Publicado"
    assert [call["model"] for call in c.beta.messages.calls] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert [chat for chat, _ in bot.sent] == [42, 42, "@c", "@c"]
    assert bot.sent[2][1] == "<foto>" and "Procure saber!" in bot.sent[3][1]
    saved = parse_materia((tmp_path / "checking" / "saber" / f"{item_id(URL)}.md").read_text())
    assert saved.stream == "band" and saved.relacionada and saved.origem == "web"


async def until(condition):
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condição não atingida")


def test_blocked_site_asks_for_pasted_text(tmp_path):
    other = "https://g1.globo.com/politica/outra-materia.ghtml"
    s, bot, c = setup(tmp_path, response("end_turn", fetch_result(error="url_not_allowed")),
                      response("tool_use", saber_call()))

    async def run():
        queue = asyncio.Queue()
        task = asyncio.create_task(receive(s, queue))
        queue.put_nowait(URL)
        queue.put_nowait("")  # Enter: matéria
        await until(lambda: s.awaiting)
        queue.put_nowait(other)  # mesmo site: pede o texto sem tentar ler
        queue.put_nowait("m")
        await until(lambda: len(s.awaiting) == 2)
        for line in ["curto", "."]:  # texto curto demais: continua esperando
            queue.put_nowait(line)
        queue.put_nowait("pular")
        await until(lambda: len(s.awaiting) == 1)
        for line in ["Dívida pública", "", ARTICLE, "."]:
            queue.put_nowait(line)
        queue.put_nowait(None)
        await task

    asyncio.run(run())
    assert list(s.awaiting) == [] and s.blocked_hosts == {"g1.globo.com"}
    assert len(c.beta.messages.calls) == 2  # a 2ª URL do site bloqueado não gastou uma leitura
    pasted = c.beta.messages.calls[1]["messages"][0]["content"][1]
    assert pasted["source"]["data"].startswith("Dívida pública\n\nSegundo o Tesouro")
    key = item_id(other)
    assert s.reviewer.known(key) and not s.reviewer.known(item_id(URL))
    assert parse_materia((tmp_path / "checking" / "saber" / f"{key}.md").read_text()).origem == "colado"
    assert (tmp_path / "checking" / "saber" / f"{key}.txt").read_text().startswith("Dívida pública")


def test_cli_requires_transcript_channel_review_chat_and_operators():
    for argv in ([], ["log"], ["log", "--channel", "@c", "--review-chat", "1"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
    assert parse_args(["--whoami"]).whoami


# --- checagem de agência -----------------------------------------------------------------------

AOS_FATOS = "https://www.aosfatos.org/noticias/flavio-bolsonaro-mente-recorde-falencias-empresas/"


def agency_check(**kw):
    base = dict(id="saber-chk", url=AOS_FATOS, stream="band", relacionada=True, motivo="Flávio citou falências",
                titulo="Flávio Bolsonaro mente ao falar em recorde de falências", veiculo="Aos Fatos",
                data_publicacao="20/10/2026", resumo="Segundo o Aos Fatos, os pedidos de falência caíram em 2026.",
                texto="Segundo o Aos Fatos, dados da Serasa mostram queda.\n\nO recorde foi em 2016.",
                falas=[Fala("Flávio Bolsonaro", "2026-10-22 22:11:40", "A dívida pública chegou a 11 trilhões")],
                modelo="claude-sonnet-5-5", analisado_em="2026-10-22 22:20:00", origem="web", tipo="checagem",
                autoria="Bianca Bortolon e Luiz Fernando Menezes", afirmacao="O Brasil bate recorde de falências <&>",
                quem_afirmou="Flávio Bolsonaro", selo="Falso", categoria="falso")
    return Materia(**(base | kw))


def check_call(e_checagem=True, falas=None):
    return NS(type="tool_use", name="registrar_checagem", id="tu1",
              input={"e_checagem": e_checagem, "motivo": "m", "agencia": "Lupa",
                     "autoria": "Gabriela Soares, Evelyn Fagundes, Leonardo Nogueira", "titulo": "Lista viral",
                     "data_publicacao": "22/10/2026", "afirmacao": "Bolsonaro criou o Pix",
                     "quem_afirmou": "post viral", "selo": "Falso", "categoria": "falso", "resumo": "r",
                     "justificativa": "j", "falas": falas or []})


def test_agency_check_without_transcript_link_still_goes():
    c = client(response("tool_use", check_call()))
    m = asyncio.run(saber.analyze(c, "claude-sonnet-5-5", "saber-x", URL, lines(), saber.pasted_document(ARTICLE),
                                  "band", "checagem"))
    assert m.relacionada and m.falas == [] and m.tipo == "checagem"
    assert (m.veiculo, m.selo, m.categoria, m.quem_afirmou) == ("Lupa", "Falso", "falso", "post viral")
    assert m.autoria == "Gabriela Soares, Evelyn Fagundes, Leonardo Nogueira"
    call = c.beta.messages.calls[0]
    assert [t["name"] for t in call["tools"]] == ["registrar_checagem"] and call["system"] == saber.CHECK_SYSTEM


def test_agency_check_drops_invented_quote_and_rejects_non_checks():
    bad = [{"quem": "P2", "horario": "2026-10-22 22:11:55", "trecho": "frase que ninguém disse"}]
    c = client(response("tool_use", check_call(falas=bad)), response("tool_use", check_call(e_checagem=False)))
    doc = saber.pasted_document(ARTICLE)
    m = asyncio.run(saber.analyze(c, "m", "saber-x", URL, lines(), doc, kind="checagem"))
    assert m.relacionada and m.falas == []
    m = asyncio.run(saber.analyze(c, "m", "saber-x", URL, lines(), doc, kind="checagem"))
    assert not m.relacionada


def test_agency_check_file_round_trip():
    m = agency_check()
    assert parse_materia(render_materia(m)) == m


def test_agency_check_message_credits_agency_and_authors():
    html = format_agency_check(agency_check(), footer="Equipe")
    assert html.startswith("❌ <b>FALSO</b> · checagem de Aos Fatos\nFlávio Bolsonaro\n"
                           "«O Brasil bate recorde de falências &lt;&amp;&gt;»")
    assert "<b>Na transmissão:</b>\n🗣 <i>Flávio Bolsonaro, 22:11:</i> «A dívida pública" in html
    viral = format_agency_check(agency_check(quem_afirmou="post viral"))  # quem repetiu não vira o autor
    assert viral.split("\n")[1] == "post viral" and "Na transmissão" in viral
    assert (f'<b>Apuração:</b> <a href="{AOS_FATOS}">Aos Fatos</a>, por Bianca Bortolon e Luiz Fernando Menezes'
            in html)
    assert html.endswith("#checagem\n<i>Equipe</i>")
    plain = format_agency_check(agency_check(falas=[], autoria="", quem_afirmou="post viral"))
    assert "\npost viral\n" in plain and "🗣" not in plain and "Aos Fatos</a>\n" in plain
    long = format_agency_check(agency_check(texto="\n\n".join(["palavra " * 300] * 10)))
    assert len(long) <= MAX_MESSAGE


def test_agency_card_uses_agency_seal_with_our_color():
    for m in (agency_check(), agency_check(falas=[], autoria="", selo="Insustentável", categoria="inconclusivo"),
              agency_check(afirmacao="x " * 300, resumo="y " * 200, autoria="Fulano de Tal, " * 10)):
        img = Image.open(io.BytesIO(render_agency_card(m, "@canal", "rodapé")))
        assert img.size == (SIZE, SIZE)
        assert img.getpixel((10, 10)) == Image.new("RGB", (1, 1), COLORS[m.categoria][0]).getpixel((0, 0))


def test_cli_asks_type_for_each_url_in_order(tmp_path):
    lupa = "https://www.agencialupa.org/noticias/2026/10/22/lista-viral/"
    s, bot, c = setup(tmp_path, response("end_turn", fetch_result(url=AOS_FATOS)),
                      response("tool_use", check_call()),
                      response("end_turn", fetch_result(url=lupa)),
                      response("tool_use", saber_call()))
    kinds = []
    real = s.handle

    async def spy(url, kind="materia", text=None):
        kinds.append((url, kind))
        await real(url, kind, text)

    s.handle = spy

    async def run():
        queue = asyncio.Queue()
        for line in [AOS_FATOS, lupa, "talvez", "c", "m", None]:
            queue.put_nowait(line)
        await receive(s, queue)

    asyncio.run(run())
    assert kinds == [(AOS_FATOS, "checagem"), (lupa, "materia")]
    assert s.reviewer.state[item_id(AOS_FATOS)]["tipo"] == "checagem-externa"
    assert s.reviewer.state[item_id(lupa)]["tipo"] == "procure-saber"
    assert "checagem de Lupa" in s.reviewer.state[item_id(AOS_FATOS)]["texto"]
