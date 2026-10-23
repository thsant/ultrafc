"""Agrupa o áudio contínuo em segmentos de fala usando VAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .source import SAMPLE_RATE

# Recebe áudio float32 e devolve intervalos de fala [(início, fim)] em amostras.
VadFn = Callable[[np.ndarray], list[tuple[int, int]]]


def silero_vad(threshold: float = 0.5) -> VadFn:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=threshold, min_silence_duration_ms=300, speech_pad_ms=100)

    def run(audio: np.ndarray) -> list[tuple[int, int]]:
        return [(s["start"], s["end"]) for s in get_speech_timestamps(audio, opts)]

    return run


@dataclass
class Segment:
    start_time: float  # epoch do início do segmento
    audio: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class Segmenter:
    """Acumula blocos e fecha um segmento após uma pausa na fala ou ao atingir a duração máxima."""

    def __init__(self, vad: VadFn, min_seconds: float = 2.0, max_seconds: float = 15.0,
                 silence_seconds: float = 0.5, long_silence_seconds: float = 1.5,
                 pad_seconds: float = 0.2):
        self.vad = vad
        self.min_len = int(min_seconds * SAMPLE_RATE)
        self.max_len = int(max_seconds * SAMPLE_RATE)
        self.silence = int(silence_seconds * SAMPLE_RATE)
        self.long_silence = int(long_silence_seconds * SAMPLE_RATE)
        self.pad = int(pad_seconds * SAMPLE_RATE)
        self._buf = np.zeros(0, dtype=np.float32)
        # (posição no buffer, horário) do início de cada bloco recebido.
        self._marks: list[tuple[int, float]] = []

    def _time_at(self, i: int) -> float:
        off, ts = next((m for m in reversed(self._marks) if m[0] <= i), self._marks[0])
        return ts + (i - off) / SAMPLE_RATE

    def _take(self, n: int) -> Segment:
        seg = Segment(self._time_at(0), self._buf[:n].copy())
        self._drop(n)
        return seg

    def _drop(self, n: int) -> None:
        if n >= len(self._buf):
            self._buf, self._marks = self._buf[:0], []
            return
        start = self._time_at(n)
        self._buf = self._buf[n:]
        self._marks = [(0, start)] + [(off - n, ts) for off, ts in self._marks if off > n]

    def push(self, ts: float, chunk: np.ndarray) -> list[Segment]:
        self._marks.append((len(self._buf), ts))
        self._buf = np.concatenate([self._buf, chunk])
        out: list[Segment] = []

        speech = self.vad(self._buf)
        if not speech:
            # Só silêncio/ruído: descarta, mantendo um pouco de margem para o início da próxima fala.
            if len(self._buf) > self.pad:
                self._drop(len(self._buf) - self.pad)
            return out

        if speech[0][0] > self.pad:
            # Silêncio antes da primeira fala não interessa.
            self._drop(speech[0][0] - self.pad)
            speech = [(s - (speech[0][0] - self.pad), e - (speech[0][0] - self.pad)) for s, e in speech]

        last_end = speech[-1][1]
        trailing = len(self._buf) - last_end
        if (trailing >= self.silence and len(self._buf) >= self.min_len) or trailing >= self.long_silence:
            out.append(self._take(min(last_end + self.pad, len(self._buf))))
        elif len(self._buf) >= self.max_len:
            out.append(self._take(self._cut_point(speech)))
        return out

    def _cut_point(self, speech: list[tuple[int, int]]) -> int:
        """Ponto de corte quando a fala não para: a maior pausa entre trechos, senão o trecho mais baixo."""
        gaps = [(speech[i + 1][0] - speech[i][1], speech[i][1]) for i in range(len(speech) - 1)]
        gaps = [(g, end) for g, end in gaps if end >= self.min_len]
        if gaps:
            gap, end = max(gaps)
            return end + gap // 2
        # Menor energia nos últimos 3 s, em janelas de 50 ms.
        win = SAMPLE_RATE // 20
        lo = max(self.min_len, len(self._buf) - 3 * SAMPLE_RATE)
        tail = self._buf[lo:len(self._buf) - len(self._buf[lo:]) % win].reshape(-1, win)
        if len(tail) == 0:
            return len(self._buf)
        return lo + int(np.argmin((tail ** 2).mean(axis=1))) * win + win // 2

    def flush(self) -> Segment | None:
        if len(self._buf) < SAMPLE_RATE // 2 or not self.vad(self._buf):
            return None
        return self._take(len(self._buf))
