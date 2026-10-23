import asyncio
import time
from types import SimpleNamespace as NS

from ultrafc.factcheck.files import Claim, Source, Verdict, render_verdict
from ultrafc.logger import LogLine
from ultrafc.publish import summary as summarizer
from ultrafc.publish.card import COLORS, SIZE, render_card, summary_text
from ultrafc.publish.__main__ import parse_args, scan_checks
from ultrafc.publish.format import MAX_MESSAGE, format_check, format_summary, trim_paragraphs
from ultrafc.publish.review import DISCARD, PUBLISH, Reviewer, short_key

T0 = time.mktime((2026, 10, 22, 22, 11, 35, 0, 0, -1))
OPERATOR, STRANGER = 111, 999


def claim(i="20261022-221135-divida-de-r-11-trilhoes-recorde-negativo-deixado-pelo-governo-atual"):
    return Claim(i, "band", "Flávio Bolsonaro", "2026-10-22 22:11:35", "número/estatística",
                 "A dívida é de R$ 11 trilhões & um <recorde>.", "a dívida", "r")


def verdict(resultado="omissão", justificativa="O valor está certo.\n\nMas só é recorde em reais.", fontes=None):
    fontes = fontes if fontes is not None else [
        Source("Banco Central", "https://bcb.gov.br/a?x=1&y=2", "dado oficial"),
        Source("Inventada", "https://x.com", "?", consultada=False)]
    return Verdict(resultado, justificativa, fontes, "claude-opus-5-5", "2026-10-22 22:13:02")


class FakeBot:
    def __init__(self, fail_channel=False):
        self.sent, self.edited, self.fail_channel, self._n = [], [], fail_channel, 0

    async def send_message(self, chat_id, text, **kw):
        if self.fail_channel and chat_id == "@canal":
            raise RuntimeError("bot não é admin do canal")
        await asyncio.sleep(0.01)
        self._n += 1
        self.sent.append((chat_id, text, kw))
        return NS(message_id=self._n)

    async def send_photo(self, chat_id, photo, **kw):
        self._n += 1
        self.sent.append((chat_id, "<foto>", {"photo": photo}))
        return NS(message_id=self._n)

    async def edit_message_text(self, chat_id, message_id, text, **kw):
        self.edited.append((chat_id, message_id, text, kw))


def reviewer(tmp_path, bot):
    return Reviewer(bot, tmp_path / "state.json", 42, "@canal", {OPERATOR})


# --- formatação -------------------------------------------------------------------------------

def test_check_message_escapes_html_and_lists_only_consulted_sources():
    html = format_check(claim(), verdict(), footer="Revisado pela equipe")
    assert html.startswith("🟡 <b>OMISSÃO</b>\nFlávio Bolsonaro · 22/10 22:11\n")
    assert "R$ 11 trilhões &amp; um &lt;recorde&gt;." in html
    assert '<a href="https://bcb.gov.br/a?x=1&amp;y=2">Banco Central</a>' in html
    assert "Inventada" not in html
    assert html.endswith("#checagem\n<i>Revisado pela equipe</i>")


def test_base_source_without_url_is_plain_text():
    html = format_check(claim(), verdict(fontes=[Source("Nota da equipe", "base:nota.txt", "x")]))
    assert "<b>Fontes:</b> Nota da equipe" in html and "base:" not in html


def test_long_justification_is_cut_at_paragraph_and_fits_telegram():
    long = "\n\n".join(f"Parágrafo {i} " + "palavra " * 60 for i in range(30))
    html = format_check(claim(), verdict(justificativa=long))
    assert len(html) <= MAX_MESSAGE
    assert "Parágrafo 0" in html and "Parágrafo 29" not in html and "…" in html
    assert trim_paragraphs("curto", 900) == "curto"
    assert trim_paragraphs("x " * 1000, 50).endswith("…")


def test_summary_message_drops_whole_topics_to_fit():
    html = format_summary(T0, T0 + 600, ["Lula disse <isso>"] + ["tópico " * 200] * 10)
    assert html.startswith("🗒 <b>Resumo — 22:11–22:21</b>")
    assert "Lula disse &lt;isso&gt;" in html and len(html) <= MAX_MESSAGE and html.endswith("#resumo")


def test_callback_key_fits_telegram_limit():
    assert len(f"{PUBLISH}:{short_key(claim().id)}".encode()) <= 64


# --- revisão ----------------------------------------------------------------------------------

def test_submit_publish_and_state_survives_restart(tmp_path):
    bot = FakeBot()
    r = reviewer(tmp_path, bot)

    async def go():
        await r.submit("a", "checagem", "<b>texto</b>")
        await r.submit("a", "checagem", "<b>texto</b>")  # já conhecido: não reenvia
        return await r.decide(PUBLISH, short_key("a"), OPERATOR, "Ana <op>")

    assert asyncio.run(go()) == "✅ Publicado"
    assert [c for c, *_ in bot.sent] == [42, "@canal"]
    assert bot.sent[1][1] == "<b>texto</b>"  # publica exatamente o texto revisado
    assert "✅ Publicado por Ana &lt;op&gt;" in bot.edited[0][2] and "reply_markup" not in bot.edited[0][3]
    again = reviewer(tmp_path, FakeBot())
    assert again.known("a") and again.state["a"]["status"] == "publicado"


def test_stranger_cannot_approve_and_double_click_publishes_once(tmp_path):
    bot = FakeBot()
    r = reviewer(tmp_path, bot)

    async def go():
        await r.submit("a", "checagem", "x")
        denied = await r.decide(PUBLISH, short_key("a"), STRANGER, "Intruso")
        both = await asyncio.gather(r.decide(PUBLISH, short_key("a"), OPERATOR, "Ana"),
                                    r.decide(PUBLISH, short_key("a"), OPERATOR, "Ana"))
        late = await r.decide(DISCARD, short_key("a"), OPERATOR, "Bia")
        return denied, both, late

    denied, both, late = asyncio.run(go())
    assert "não está autorizado" in denied
    assert sorted(both) == ["Já publicado por Ana.", "✅ Publicado"]
    assert late == "Já publicado por Ana."
    assert [c for c, *_ in bot.sent].count("@canal") == 1


def test_channel_failure_keeps_item_in_review(tmp_path):
    r = reviewer(tmp_path, FakeBot(fail_channel=True))

    async def go():
        await r.submit("a", "checagem", "x")
        return await r.decide(PUBLISH, short_key("a"), OPERATOR, "Ana")

    assert "Falha ao publicar" in asyncio.run(go())
    assert r.state["a"]["status"] == "revisao"


def test_scan_sends_checks_but_not_errors(tmp_path):
    checked = tmp_path / "checked"
    checked.mkdir()
    ok, bad = claim("ok-1"), claim("erro-1")
    (checked / "ok-1.md").write_text(render_verdict(ok, verdict()))
    (checked / "erro-1.md").write_text(render_verdict(bad, verdict("erro", "API fora do ar", [])))
    bot = FakeBot()
    r = reviewer(tmp_path, bot)
    asyncio.run(scan_checks(r, checked, None))
    asyncio.run(scan_checks(r, checked, None))  # segunda varredura não reenvia
    assert len(bot.sent) == 1 and "OMISSÃO" in bot.sent[0][1]
    assert r.state["erro-1"]["status"] == "ignorado"


# --- cards -----------------------------------------------------------------------------------

def test_card_is_square_png_with_result_color():
    import io

    from PIL import Image

    for resultado, (band, _) in COLORS.items():
        png = render_card(claim(), verdict(resultado), "@canal", "Revisado pela equipe")
        img = Image.open(io.BytesIO(png))
        assert img.format == "PNG" and img.size == (SIZE, SIZE)
        assert img.getpixel((SIZE - 5, 5)) == Image.new("RGB", (1, 1), band).getpixel((0, 0))
    from ultrafc.publish.card import fit

    f, lines, _ = fit("INCONCLUSIVO", "B", range(112, 60, -4), 700, 130)  # uma palavra: cabe na largura
    assert len(lines) == 1 and f.getlength(lines[0]) <= 700
    long = claim()
    long.afirmacao = "Ação, ônibus e coração " * 80  # cabe cortada, sem erro
    render_card(long, verdict(justificativa="ç " * 2000))


def test_card_summary_prefers_checker_summary():
    v = verdict(justificativa="Primeira frase. Segunda frase.")
    assert summary_text(v) == "Primeira frase."  # checagens antigas, sem resumo
    v.resumo = "O valor está certo, mas não é recorde em proporção do PIB."
    assert summary_text(v) == v.resumo


def test_card_goes_before_text_and_is_not_repeated_on_retry(tmp_path):
    checked = tmp_path / "checked"
    checked.mkdir()
    (checked / "ok-1.md").write_text(render_verdict(claim("ok-1"), verdict()))

    class FlakyBot(FakeBot):
        fails = 1

        async def send_message(self, chat_id, text, **kw):
            if chat_id == "@canal" and self.fails:
                self.fails -= 1
                raise RuntimeError("timeout")
            return await super().send_message(chat_id, text, **kw)

    bot = FlakyBot()
    r = reviewer(tmp_path, bot)

    async def go():
        await scan_checks(r, checked, None, tmp_path / "cards", "@canal")
        first = await r.decide(PUBLISH, short_key("ok-1"), OPERATOR, "Ana")
        second = await r.decide(PUBLISH, short_key("ok-1"), OPERATOR, "Ana")
        return first, second

    first, second = asyncio.run(go())
    assert "Falha ao publicar" in first and second == "✅ Publicado"
    assert (tmp_path / "cards" / "ok-1.png").read_bytes().startswith(b"\x89PNG")
    kinds = [(c, "foto" if t == "<foto>" else "texto") for c, t, _ in bot.sent]
    assert kinds == [(42, "foto"), (42, "texto"), ("@canal", "foto"), ("@canal", "texto")]


# --- resumo e CLI -----------------------------------------------------------------------------

def test_summarize_and_save(tmp_path):
    from ultrafc.publish.summary import Resumo

    calls = []

    async def parse(**kw):
        calls.append(kw)
        return NS(stop_reason="end_turn", parsed_output=Resumo(topicos=[" Lula disse X ", ""]),
                  usage=NS(input_tokens=1, output_tokens=1))

    client = NS(beta=NS(messages=NS(parse=parse)))
    lines = [LogLine(T0 + i, "Lula", f"fala {i}") for i in range(8)]
    topics = asyncio.run(summarizer.summarize(client, "claude-sonnet-5-5", lines))
    assert topics == ["Lula disse X"]
    assert "[2026-10-22 22:11:35] Lula: fala 0" in calls[0]["messages"][0]["content"]
    item = summarizer.summary_id(T0, T0 + 7)
    summarizer.save(tmp_path, item, "band", T0, T0 + 7, topics)
    assert "- Lula disse X" in (tmp_path / f"{item}.md").read_text()


def test_cli_requires_channel_review_chat_and_operators():
    import pytest

    for argv in ([], ["--channel", "@c"], ["--channel", "@c", "--review-chat", "1"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
    assert parse_args(["--whoami"]).whoami
    a = parse_args(["--channel", "@c", "--review-chat", "-100123", "--operators", "1,2"])
    assert a.operators == "1,2"


class ChannelBot:
    id = 7

    def __init__(self, status=None, can_post=True, username="ultrafc"):
        self.status, self.can_post, self.username = status, can_post, username

    async def get_chat(self, channel):
        return NS(id=-1002900827044, title="Yt", username=self.username)

    async def get_chat_member(self, chat_id, user_id):
        from telegram.error import BadRequest

        if self.status is None:
            raise BadRequest("Member list is inaccessible")
        return NS(status=self.status, can_post_messages=self.can_post)


def test_channel_check_refuses_channel_where_bot_cannot_post():
    import pytest

    from ultrafc.publish.__main__ import check_channel

    for bot in (ChannelBot(None), ChannelBot("member"), ChannelBot("administrator", can_post=False)):
        with pytest.raises(SystemExit, match="'Yt' \\(@ultrafc, id -1002900827044\\)"):
            asyncio.run(check_channel(bot, "@UltraFC"))
    assert asyncio.run(check_channel(ChannelBot("administrator", username=None), -1002900827044)) == -1002900827044
