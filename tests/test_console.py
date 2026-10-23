import logging
import re

from rich.console import Console

from ultrafc.console import PALETTE, Display, MessageHandler, make_consoles, speaker_style
from ultrafc.logger import format_line

TS = 1_761_177_600.0
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def terminal(width=80):
    return Console(record=True, force_terminal=True, width=width, color_system="truecolor", highlight=False)


def test_speaker_color_comes_from_label_number():
    assert speaker_style("P1") == speaker_style("P1")
    assert speaker_style("P1") != speaker_style("P2")
    assert speaker_style("P1").color.name == PALETTE[0]
    assert speaker_style(f"P{len(PALETTE) + 1}") == speaker_style("P1")  # paleta repete
    assert speaker_style(None).color is None


def test_renamed_speaker_keeps_label_color():
    con = terminal()
    d = Display(con)
    d.utterance(TS, "fala", "P1", "P1")
    d.utterance(TS, "fala", "P1", "William Waack")
    out = con.export_text(styles=True)
    rgb = "38;2;78;121;167"  # PALETTE[0] = #4e79a7
    assert out.count(rgb) >= 4  # nome (2x) + texto (2x)
    assert "William Waack" in out


def test_long_text_wraps_with_hanging_indent():
    con = terminal(width=60)
    Display(con).utterance(TS, "palavra " * 20, "P2", "P2")
    lines = con.export_text().rstrip("\n").splitlines()
    assert len(lines) > 1
    indent = lines[0].index("palavra")
    assert all(line[:indent].strip() == "" for line in lines[1:])


def test_piped_output_is_plain_log_line():
    con = Console(record=True, force_terminal=False, width=40)
    Display(con).utterance(TS, "texto bem longo " * 5, "P1", "Marcos")
    assert con.export_text() == format_line(TS, "texto bem longo " * 5, "Marcos") + "\n"


def test_messages_are_italic_with_level_icon():
    con = terminal()
    log = logging.getLogger("ultrafc.test")
    log.handlers[:] = [MessageHandler(con)]
    log.propagate = False
    log.setLevel(logging.INFO)
    log.info("nomes atualizados: %s", "P1→Marcos")
    log.warning("transcrição atrasada")
    out = con.export_text(styles=True)
    plain = ANSI.sub("", out)
    assert "ⓘ nomes atualizados: P1→Marcos" in plain and "⚠ transcrição atrasada" in plain
    assert all("\x1b[3" in line for line in out.splitlines() if line.strip())  # itálico (SGR 3)


def test_color_never_has_no_ansi():
    stdout, stderr = make_consoles("never")
    stdout.begin_capture()
    stdout.print("[bold]x[/bold]")
    Display(stdout).utterance(TS, "fala", "P1", "P1")
    assert "\x1b[" not in stdout.end_capture()


def test_long_message_wraps_aligned_to_text():
    con = terminal(width=50)
    log = logging.getLogger("ultrafc.test.wrap")
    log.handlers[:] = [MessageHandler(con)]
    log.propagate = False
    log.warning("capturando https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1792741577")
    lines = con.export_text().rstrip("\n").splitlines()
    assert len(lines) > 1
    indent = lines[0].index("capturando")
    assert all(line[:indent].strip() == "" for line in lines[1:])


def test_secrets_are_masked_and_http_loggers_stay_quiet():
    from ultrafc.console import setup_logging

    con = terminal(width=200)
    setup_logging(con, verbose=True)
    logging.getLogger("httpx").info("POST https://api.telegram.org/bot123:AAHsecretsecretsecret_123/getMe")
    logging.getLogger("ultrafc.x").info("url https://api.telegram.org/bot123:AAHsecretsecretsecret_123/getMe %s",
                                       "e chave sk-ant-api03-abcdefghijkl")
    out = con.export_text()
    assert "AAHsecret" not in out and "abcdefghijkl" not in out
    assert out.count("bot123:<oculto>") == 1 and "sk-ant-<oculto>" in out  # httpx nem aparece
    logging.getLogger().handlers.clear()
