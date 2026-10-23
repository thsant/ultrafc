"""Transcrição dos segmentos com faster-whisper."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Frases que o Whisper costuma "alucinar" em silêncio/música (vindas das legendas de treino).
_HALLUCINATIONS = re.compile(
    r"amara\.org|legendas? (pela|por)|inscreva-se no canal|obrigad[oa] por assistir",
    re.IGNORECASE,
)


@dataclass
class Word:
    start: float  # segundos desde o início do segmento
    end: float
    text: str  # inclui o espaço inicial, como o Whisper devolve


def join_words(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


def default_compute_type(device: str) -> str:
    return "int8_float16" if device == "cuda" else "int8"


class Transcriber:
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str | None = None,
                 language: str = "pt", beam_size: int = 5):
        from faster_whisper import WhisperModel

        if device == "auto":
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = compute_type or default_compute_type(device)
        log.info("carregando modelo %s (%s, %s)", model, device, compute_type)
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size
        self._previous = ""

    def __call__(self, audio: np.ndarray) -> list[Word]:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # o áudio já chega segmentado pelo VAD
            condition_on_previous_text=False,
            initial_prompt=self._previous[-200:] or None,
            word_timestamps=True,
        )
        words: list[Word] = []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            if _HALLUCINATIONS.search(s.text):
                continue
            words += [Word(w.start, w.end, w.word) for w in s.words or []]
        if text := join_words(words):
            self._previous = text
        return words
