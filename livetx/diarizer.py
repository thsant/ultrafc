"""Diarização online: identifica quem fala (P1, P2, … PN) por embeddings de voz."""

from __future__ import annotations

import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .source import SAMPLE_RATE
from .transcriber import Word

log = logging.getLogger(__name__)

# 3D-Speaker CAM++: no debate da Band (4 min, 3 vozes) errou 8,9% dos pares de janelas de 1,5 s,
# contra 32% do WeSpeaker ResNet34, e é ~2x mais rápido (~19 ms por janela).
DEFAULT_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
DEFAULT_THRESHOLD = 0.45
UNKNOWN = "?"  # fala curta demais para criar um falante e sem nenhum conhecido parecido
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{}"
MODEL_DIR = Path.home() / ".cache" / "livetx" / "models"

# Recebe áudio float32 de 16 kHz e devolve o embedding normalizado (L2).
EmbedFn = Callable[[np.ndarray], np.ndarray]


def download_model(name_or_path: str = DEFAULT_MODEL) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    path = MODEL_DIR / path.name
    if not path.exists():
        log.info("baixando modelo de voz %s", path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL.format(path.name), tmp)
        tmp.rename(path)
    return path


class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, num_threads: int = 2):
        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(download_model(model)),
                                                             num_threads=num_threads)
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        emb = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)


def windows(n_samples: int, size: float = 1.5, hop: float = 0.75) -> list[tuple[int, int]]:
    """Janelas (início, fim) em amostras cobrindo todo o áudio; a última termina no fim."""
    size_n, hop_n = int(size * SAMPLE_RATE), int(hop * SAMPLE_RATE)
    if n_samples <= size_n:
        return [(0, n_samples)]
    spans = [(s, s + size_n) for s in range(0, n_samples - size_n + 1, hop_n)]
    if spans[-1][1] < n_samples:
        spans.append((n_samples - size_n, n_samples))
    return spans


def _label_number(label: str) -> int:
    m = re.fullmatch(r"P(\d+)", label)
    return int(m.group(1)) if m else 0


class SpeakerRegistry:
    """Centróides dos falantes conhecidos, persistidos em .npz entre execuções."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, path: Path | None = None, max_count: int = 200,
                 save_every: float = 60.0, margin: float = 0.1, model: str = DEFAULT_MODEL):
        self.threshold = threshold
        self.model = Path(model).name  # embeddings de modelos diferentes não são comparáveis
        # Vozes no mesmo estúdio/microfone ficam próximas: só troca de falante, e só atualiza um
        # centróide, quando a melhor voz vence a concorrente por pelo menos `margin`.
        self.margin = margin
        self.path = Path(path) if path else None
        self.max_count = max_count  # limita o peso do histórico para o centróide continuar adaptando
        self.save_every = save_every
        self.labels: list[str] = []
        self.centroids = np.zeros((0, 0), dtype=np.float32)
        self.counts: list[int] = []
        self._last: int | None = None  # falante da última janela (continua entre segmentos)
        self._saved_at = time.monotonic()
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        data = np.load(self.path)
        saved_model = str(data["model"]) if "model" in data else "desconhecido"
        if saved_model != self.model:
            old = self.path.with_name(f"{self.path.stem}.{Path(saved_model).stem}.npz")
            self.path.replace(old)
            log.warning("falantes salvos com outro modelo de voz (%s): começando do zero; antigos em %s",
                        saved_model, old.name)
            return
        self.labels = [str(x) for x in data["labels"]]
        self.centroids = data["centroids"].astype(np.float32)
        self.counts = [int(x) for x in data["counts"]]
        log.info("falantes carregados: %s", ", ".join(self.labels) or "nenhum")

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, labels=np.array(self.labels), centroids=self.centroids, counts=np.array(self.counts),
                 model=np.array(self.model))
        tmp.replace(self.path)
        self._saved_at = time.monotonic()

    def maybe_save(self) -> None:
        if time.monotonic() - self._saved_at >= self.save_every:
            self.save()

    def _match(self, emb: np.ndarray) -> tuple[int, float]:
        if not self.labels:
            return -1, -1.0
        sims = self.centroids @ emb
        best = int(np.argmax(sims))
        return best, float(sims[best])

    def _choose(self, emb: np.ndarray) -> tuple[int, float, bool]:
        """(falante, similaridade, confiante). Prefere continuar com o falante anterior."""
        sims = self.centroids @ emb
        best = int(np.argmax(sims))
        last = self._last
        if last is not None and sims[last] >= self.threshold and sims[best] - sims[last] < self.margin:
            best = last
        others = np.delete(sims, best)
        confident = others.size == 0 or sims[best] - others.max() >= self.margin
        return best, float(sims[best]), confident

    def _update(self, i: int, emb: np.ndarray) -> None:
        n = min(self.counts[i], self.max_count)
        c = self.centroids[i] * n + emb
        self.centroids[i] = c / (np.linalg.norm(c) + 1e-9)
        self.counts[i] += 1

    def _create(self, emb: np.ndarray) -> str:
        label = f"P{max(map(_label_number, self.labels), default=0) + 1}"
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        self.centroids = emb[None, :] if not self.labels else np.vstack([self.centroids, emb])
        self.labels.append(label)
        self.counts.append(1)
        log.info("novo falante: %s", label)
        self.save()
        return label

    def assign(self, embs: list[np.ndarray], min_new: int = 3) -> list[str]:
        """Rotula as janelas de um segmento. Um falante novo só nasce de `min_new` janelas
        consecutivas (3 janelas de 1,5 s com passo 0,75 s = 3 s de áudio) que não batem com
        ninguém, mas batem entre si. Trechos curtos (vinheta, música, "Hum!") não criam falante."""
        out: list[str | None] = []
        run: list[int] = []  # índices consecutivos sem correspondência

        def close_run() -> None:
            if len(run) >= min_new:
                label = self._create(np.mean([embs[j] for j in run], axis=0))
                self._last = self.labels.index(label)
                for j in run:
                    out[j] = label
            run.clear()

        for i, emb in enumerate(embs):
            best, sim, confident = self._choose(emb) if self.labels else (-1, -1.0, False)
            log.debug("janela %d: %s sim=%.2f%s", i, self.labels[best] if best >= 0 else "-", sim,
                      "" if confident else " ?")
            if sim >= self.threshold:
                close_run()
                out.append(self.labels[best])
                self._last = best
                if confident:
                    self._update(best, emb)
            else:
                if run and float(embs[run[-1]] @ emb) < self.threshold:
                    close_run()
                out.append(None)
                run.append(i)
        close_run()

        if all(label is None for label in out):
            # Ninguém bateu: aproxima de um conhecido parecido; senão cria, se o trecho for longo o
            # bastante, ou deixa como desconhecido.
            mean = np.mean(embs, axis=0)
            best, sim = self._match(mean / (np.linalg.norm(mean) + 1e-9))
            if sim >= self.threshold * 0.8:
                label = self.labels[best]
            elif len(embs) >= min_new:
                label = self._create(mean)
            else:
                return [UNKNOWN] * len(embs)
            self._last = self.labels.index(label)
            return [label] * len(embs)
        return _fill_gaps(out)


def _fill_gaps(labels: list[str | None]) -> list[str]:
    """Janelas sem rótulo herdam o da janela rotulada mais próxima."""
    known = [i for i, label in enumerate(labels) if label is not None]
    return [label if label is not None else labels[min(known, key=lambda k: abs(k - i))]
            for i, label in enumerate(labels)]


def smooth(labels: list[str], k: int = 3) -> list[str]:
    """Filtro de maioria: remove trocas isoladas de uma janela (P1 P2 P1 → P1 P1 P1)."""
    if len(labels) < k:
        return list(labels)
    half = k // 2
    out = []
    for i, label in enumerate(labels):
        win = labels[max(0, i - half):i + half + 1]
        top = max(set(win), key=win.count)
        out.append(top if win.count(top) > win.count(label) else label)
    return out


@dataclass
class Utterance:
    start: float  # segundos desde o início do segmento
    speaker: str
    text: str


def label_words(words: list[Word], spans: list[tuple[int, int]], labels: list[str]) -> list[Utterance]:
    """Atribui a cada palavra o falante da janela mais próxima e agrupa palavras consecutivas."""
    centers = np.array([(s + e) / 2 / SAMPLE_RATE for s, e in spans])
    out: list[Utterance] = []
    for w in words:
        speaker = labels[int(np.argmin(np.abs(centers - (w.start + w.end) / 2)))]
        if out and out[-1].speaker == speaker:
            out[-1].text += w.text
        else:
            out.append(Utterance(w.start, speaker, w.text))
    for u in out:
        u.text = u.text.strip()
    return [u for u in out if u.text]


class Diarizer:
    def __init__(self, embed: EmbedFn, registry: SpeakerRegistry):
        self.embed = embed
        self.registry = registry

    def __call__(self, audio: np.ndarray, words: list[Word]) -> list[Utterance]:
        spans = windows(len(audio))
        labels = smooth(self.registry.assign([self.embed(audio[s:e]) for s, e in spans]))
        self.registry.maybe_save()
        return label_words(words, spans, labels)
