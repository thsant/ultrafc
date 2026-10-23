import numpy as np

from ultrafc.segmenter import Segmenter
from ultrafc.source import SAMPLE_RATE

CHUNK = SAMPLE_RATE // 2


def energy_vad(audio, win=SAMPLE_RATE // 20, thr=1e-3):
    """VAD falso para testes: janelas com energia acima do limiar são "fala"."""
    n = len(audio) // win
    active = (audio[: n * win].reshape(n, win) ** 2).mean(axis=1) > thr
    spans, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            spans.append((start * win, i * win))
            start = None
    if start is not None:
        spans.append((start * win, n * win))
    return spans


def tone(seconds):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def feed(seg, audio, t0=1000.0):
    out = []
    for i in range(0, len(audio), CHUNK):
        out += seg.push(t0 + i / SAMPLE_RATE, audio[i:i + CHUNK])
    return out


def test_silence_produces_nothing():
    seg = Segmenter(energy_vad)
    assert feed(seg, silence(10)) == []
    assert seg.flush() is None


def test_speech_then_pause_emits_one_segment_with_wall_clock_start():
    seg = Segmenter(energy_vad)
    out = feed(seg, np.concatenate([silence(3), tone(4), silence(2)]))
    assert len(out) == 1
    assert 3.8 <= out[0].duration <= 4.6
    assert abs(out[0].start_time - (1000.0 + 3 - 0.2)) < 0.1


def test_two_utterances_become_two_segments():
    seg = Segmenter(energy_vad)
    out = feed(seg, np.concatenate([tone(3), silence(1), tone(3), silence(2)]))
    assert len(out) == 2
    assert out[1].start_time > out[0].start_time + 3


def test_continuous_speech_is_cut_at_max_length():
    seg = Segmenter(energy_vad, max_seconds=10)
    out = feed(seg, tone(35))
    assert len(out) >= 3
    assert all(s.duration <= 10.5 for s in out)
    rest = seg.flush()
    total = sum(s.duration for s in out) + (rest.duration if rest else 0)
    assert abs(total - 35) < 0.1


def test_short_utterance_emitted_after_long_silence():
    seg = Segmenter(energy_vad, min_seconds=2)
    out = feed(seg, np.concatenate([tone(0.8), silence(2)]))
    assert len(out) == 1


def test_start_time_follows_chunk_timestamps_not_sample_count():
    # Blocos que chegaram juntos (rajada inicial do HLS) têm o mesmo horário.
    seg = Segmenter(energy_vad)
    audio = np.concatenate([silence(2), tone(3), silence(2)])
    out = []
    for n, i in enumerate(range(0, len(audio), CHUNK)):
        ts = 1000.0 if n < 10 else 1000.0 + (n - 10) * 0.5
        out += seg.push(ts, audio[i:i + CHUNK])
    assert len(out) == 1
    # Início em 1,8 s = bloco 3 (ts=1000, posição 1,5 s) + 0,3 s; não 1000 + 1,8.
    assert abs(out[0].start_time - 1000.3) < 0.05
