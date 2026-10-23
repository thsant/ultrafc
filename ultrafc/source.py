"""Resolução da URL da live e captura do áudio via ffmpeg."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urlparse

import numpy as np

SAMPLE_RATE = 16000

log = logging.getLogger(__name__)

# Protocolos que o ffmpeg abre diretamente, sem passar por yt-dlp/streamlink.
_DIRECT_SCHEMES = {"udp", "rtp", "rtmp", "rtmps", "rtsp", "srt", "tcp", "file"}
_DIRECT_SUFFIXES = (".m3u8", ".mpd", ".ts", ".flv", ".mp3", ".aac", ".wav", ".mp4", ".mkv")


@dataclass
class Media:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    slug: str = "stream"
    is_live: bool = True


@dataclass
class ResolverOptions:
    resolver: str = "yt-dlp"  # "yt-dlp" | "streamlink" | "direct"
    cookies: str | None = None
    cookies_from_browser: str | None = None


def slugify(text: str, maxlen: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:maxlen] or "stream"


def _is_direct(url: str) -> bool:
    if os.path.exists(url):
        return True
    parsed = urlparse(url)
    return parsed.scheme in _DIRECT_SCHEMES or parsed.path.lower().endswith(_DIRECT_SUFFIXES)


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    return slugify(f"{parsed.netloc}{parsed.path}" if parsed.netloc else os.path.basename(url))


def _resolve_ytdlp(url: str, opts: ResolverOptions) -> Media:
    cmd = ["yt-dlp", "-j", "--no-warnings", "--no-playlist", "-f", "bestaudio/best"]
    if opts.cookies:
        cmd += ["--cookies", opts.cookies]
    if opts.cookies_from_browser:
        cmd += ["--cookies-from-browser", opts.cookies_from_browser]
    out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp falhou: {out.stderr.strip()}")
    info = json.loads(out.stdout.splitlines()[0])

    fmt = info
    # Formato combinado (vídeo + áudio separados): usa a faixa de áudio.
    for f in info.get("requested_formats") or []:
        if f.get("acodec") not in (None, "none"):
            fmt = f
            break

    name = info.get("channel") or info.get("uploader") or info.get("id") or ""
    return Media(
        url=fmt["url"],
        headers=fmt.get("http_headers") or info.get("http_headers") or {},
        slug=slugify(f"{info.get('extractor_key', '')}-{name}-{info.get('id', '')}"),
        is_live=bool(info.get("is_live", True)),
    )


def _resolve_streamlink(url: str, opts: ResolverOptions) -> Media:
    cmd = ["streamlink", "--stream-url", url, "audio_only,worst,best"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"streamlink falhou: {(out.stdout + out.stderr).strip()}")
    return Media(url=out.stdout.strip(), slug=_slug_from_url(url))


def resolve(url: str, opts: ResolverOptions) -> Media:
    """Converte a URL da página (YouTube, Instagram...) na URL de mídia que o ffmpeg abre."""
    if opts.resolver == "direct" or _is_direct(url):
        return Media(url=url, slug=_slug_from_url(url), is_live=not os.path.exists(url))
    if opts.resolver == "streamlink":
        return _resolve_streamlink(url, opts)
    return _resolve_ytdlp(url, opts)


def ffmpeg_command(media: Media) -> list[str]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if media.url.startswith(("http://", "https://")):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
        if media.headers:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in media.headers.items())]
    cmd += ["-i", media.url, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    return cmd


class AudioSource:
    """Lê o áudio da live em blocos float32 de 16 kHz, reconectando quando a stream cai.

    Cada bloco vem com o horário de parede (epoch) em que o áudio começou a ser recebido.
    """

    def __init__(self, url: str, opts: ResolverOptions, chunk_seconds: float = 0.5,
                 reconnect: bool = True, max_backoff: float = 60.0):
        self.url = url
        self.opts = opts
        self.chunk_bytes = int(SAMPLE_RATE * chunk_seconds) * 2
        self.reconnect = reconnect
        self.max_backoff = max_backoff
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self.media: Media | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def chunks(self) -> Iterator[tuple[float, np.ndarray]]:
        backoff = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                # Re-resolve a cada conexão: URLs de mídia do YouTube/Instagram expiram.
                self.media = resolve(self.url, self.opts)
                if not self.media.is_live:
                    self.reconnect = False
                yield from self._read(self.media)
            except Exception as exc:  # noqa: BLE001 - qualquer falha leva à reconexão
                log.warning("falha na captura: %s", exc)

            if self._stop.is_set() or not self.reconnect:
                break
            if time.monotonic() - started > 60:
                backoff = 1.0
            log.warning("stream encerrada; reconectando em %.0f s", backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.max_backoff)

    def _read(self, media: Media) -> Iterator[tuple[float, np.ndarray]]:
        log.info("capturando %s", media.url[:120])
        self._proc = subprocess.Popen(ffmpeg_command(media), stdout=subprocess.PIPE)
        assert self._proc.stdout is not None
        pending = b""
        clock: float | None = None
        try:
            while True:
                data = self._proc.stdout.read(self.chunk_bytes)
                if not data:
                    break
                data = pending + data
                usable = len(data) - len(data) % 2
                pending = data[usable:]
                samples = np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0
                # O ffmpeg entrega o buffer inicial do HLS de uma vez: a hora do bloco segue a contagem
                # de amostras, mas nunca passa da hora em que ele de fato chegou.
                arrived = time.time() - len(samples) / SAMPLE_RATE
                ts = arrived if clock is None else min(clock, arrived)
                clock = ts + len(samples) / SAMPLE_RATE
                yield ts, samples
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
            self._proc.wait()
