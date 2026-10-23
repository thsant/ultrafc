"""CLI: `python -m livetx <url>` transcreve uma live e grava em logs/<stream>/<data>.log."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from pathlib import Path

from rich_argparse import RichHelpFormatter

from .console import Display, make_consoles, setup_logging
from .diarizer import DEFAULT_MODEL, DEFAULT_THRESHOLD, Diarizer, Embedder, SpeakerRegistry
from .logger import TranscriptLog
from .people import PeopleMap
from .segmenter import Segment, Segmenter, silero_vad
from .source import AudioSource, ResolverOptions, resolve
from .transcriber import Transcriber, join_words

log = logging.getLogger("livetx")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="livetx", description=__doc__, formatter_class=RichHelpFormatter)
    p.add_argument("url", help="URL da live (YouTube, Instagram, G1...), .m3u8, udp://, rtmp:// ou arquivo")
    p.add_argument("--out-dir", default="logs", help="diretório base dos logs (padrão: logs)")
    p.add_argument("--name", help="nome da subpasta do log (padrão: derivado da stream)")
    p.add_argument("--model", default="small", help="modelo Whisper: tiny, base, small, medium, large-v3, turbo...")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-type", help="padrão: int8_float16 na GPU, int8 na CPU")
    p.add_argument("--language", default="pt")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--resolver", default="yt-dlp", choices=["yt-dlp", "streamlink", "direct"])
    p.add_argument("--cookies", help="arquivo cookies.txt repassado ao yt-dlp")
    p.add_argument("--cookies-from-browser", help="ex.: firefox, chrome (repassado ao yt-dlp)")
    p.add_argument("--max-segment", type=float, default=15.0, help="duração máxima de um segmento (s)")
    p.add_argument("--max-backlog", type=int, default=4,
                   help="segmentos aguardando transcrição antes de descartar os mais antigos")
    p.add_argument("--no-diarize", action="store_true", help="não identifica os falantes (P1, P2...)")
    p.add_argument("--speaker-threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="similaridade mínima (cosseno) para considerar a mesma voz")
    p.add_argument("--speaker-model", default=DEFAULT_MODEL, help="modelo de embedding de voz (sherpa-onnx)")
    p.add_argument("--reset-speakers", action="store_true", help="esquece os falantes salvos desta stream")
    p.add_argument("--people", default="people.txt",
                   help="arquivo 'P1:Nome' relido a cada 10 s (padrão: people.txt)")
    p.add_argument("--no-reconnect", action="store_true", help="encerra quando a stream cair")
    p.add_argument("--quiet", action="store_true", help="não repete a transcrição no terminal")
    p.add_argument("--color", default="auto", choices=["auto", "always", "never"],
                   help="cores no terminal (auto: só em terminal interativo; respeita NO_COLOR)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def capture(source: AudioSource, segmenter: Segmenter, segments: queue.Queue, max_backlog: int) -> None:
    """Thread produtora: áudio → VAD → fila de segmentos. Coloca None no fim."""

    def put(seg: Segment) -> None:
        while segments.qsize() >= max_backlog:
            try:
                old = segments.get_nowait()
            except queue.Empty:
                break
            if old is not None:
                log.warning("transcrição atrasada: descartando %.1f s de áudio", old.duration)
        segments.put(seg)

    try:
        for ts, chunk in source.chunks():
            for seg in segmenter.push(ts, chunk):
                put(seg)
        if (seg := segmenter.flush()) is not None:
            put(seg)
    finally:
        segments.put(None)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stdout, stderr = make_consoles(args.color)
    setup_logging(stderr, args.verbose)
    status = Display(stderr).status
    display = None if args.quiet else Display(stdout)

    opts = ResolverOptions(args.resolver, args.cookies, args.cookies_from_browser)
    with status("resolvendo a stream…"):
        name = args.name or resolve(args.url, opts).slug
    out = TranscriptLog(Path(args.out_dir) / name)
    Display(stderr).header("livetx", name, f"modelo {args.model}", str(out.directory))

    diarize = None
    people = None
    with status("carregando modelos…"):
        transcribe = Transcriber(args.model, args.device, args.compute_type, args.language, args.beam_size)
        if not args.no_diarize:
            speakers = out.directory / "speakers.npz"
            if args.reset_speakers:
                speakers.unlink(missing_ok=True)
            diarize = Diarizer(Embedder(args.speaker_model),
                               SpeakerRegistry(args.speaker_threshold, speakers, model=args.speaker_model))
            people = PeopleMap(args.people)

    def emit(seg: Segment) -> None:
        t0 = time.monotonic()
        words = transcribe(seg.audio)
        if not words:
            return  # sem fala reconhecida: não alimenta o registro de vozes
        if diarize:
            for u in diarize(seg.audio, words):
                ts, who = seg.start_time + u.start, people.name(u.speaker)
                out.write(ts, u.text, who)
                if display:
                    display.utterance(ts, u.text, u.speaker, who)
        elif text := join_words(words):
            out.write(seg.start_time, text)
            if display:
                display.utterance(seg.start_time, text)
        log.debug("%.1f s de áudio em %.2f s (atraso %.1f s)", seg.duration,
                  time.monotonic() - t0, time.time() - seg.start_time - seg.duration)

    source = AudioSource(args.url, opts, reconnect=not args.no_reconnect)
    segmenter = Segmenter(silero_vad(), max_seconds=args.max_segment)
    segments: queue.Queue = queue.Queue()
    producer = threading.Thread(target=capture, args=(source, segmenter, segments, args.max_backlog),
                                daemon=True)
    producer.start()

    try:
        while (seg := segments.get()) is not None:
            emit(seg)
    except KeyboardInterrupt:
        log.info("encerrando...")
        source.stop()
        producer.join(timeout=5)
        # Transcreve o que já estava na fila (inclui o resto do buffer).
        while not segments.empty() and (seg := segments.get_nowait()) is not None:
            emit(seg)
    finally:
        source.stop()
        if diarize:
            diarize.registry.save()
            people.stop()
        out.close()


if __name__ == "__main__":
    main()
