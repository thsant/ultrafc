import subprocess

import numpy as np

from ultrafc.logger import TranscriptLog, format_line
from ultrafc.source import SAMPLE_RATE, AudioSource, Media, ResolverOptions, ffmpeg_command, resolve


def test_direct_urls_bypass_ytdlp():
    for url in ["udp://127.0.0.1:1234", "rtmp://host/live/x", "https://cdn.example/a/index.m3u8"]:
        m = resolve(url, ResolverOptions())
        assert m.url == url and m.is_live


def test_ffmpeg_command_adds_http_options_and_headers():
    cmd = ffmpeg_command(Media("https://x/y.m3u8", headers={"User-Agent": "UA"}))
    assert "-reconnect" in cmd and "User-Agent: UA\r\n" in cmd
    assert "-reconnect" not in ffmpeg_command(Media("udp://127.0.0.1:1"))


def test_pulse_input_uses_given_source_or_default_monitor(monkeypatch):
    m = resolve("pulse:x.monitor", ResolverOptions())
    assert (m.url, m.input_format, m.is_live) == ("x.monitor", "pulse", True)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="sink1\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve("pulse:", ResolverOptions()).url == "sink1.monitor"
    assert calls == [["pactl", "get-default-sink"]]  # nunca o yt-dlp
    cmd = ffmpeg_command(Media("x.monitor", input_format="pulse"))
    assert cmd[cmd.index("-i") - 2:cmd.index("-i") + 2] == ["-f", "pulse", "-i", "x.monitor"]
    assert "-reconnect" not in cmd


def test_audio_source_reads_local_file(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=3",
                    "-ar", "44100", str(wav)], check=True)
    src = AudioSource(str(wav), ResolverOptions())
    chunks = list(src.chunks())
    audio = np.concatenate([c for _, c in chunks])
    assert abs(len(audio) / SAMPLE_RATE - 3) < 0.05
    assert audio.dtype == np.float32 and 0.05 < np.abs(audio).max() <= 1.0


def test_transcript_log_daily_file(tmp_path):
    out = TranscriptLog(tmp_path)
    out.write(1_761_177_600.0, "olá mundo")
    out.close()
    (f,) = tmp_path.glob("*.log")
    assert f.read_text(encoding="utf-8") == format_line(1_761_177_600.0, "olá mundo") + "\n"
