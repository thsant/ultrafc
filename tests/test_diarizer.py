import numpy as np

from livetx.diarizer import UNKNOWN, Diarizer, SpeakerRegistry, label_words, smooth, windows
from livetx.source import SAMPLE_RATE
from livetx.transcriber import Word

RNG = np.random.default_rng(0)
DIM = 64


def voice():
    v = RNG.normal(size=DIM)
    return v / np.linalg.norm(v)


def sample(v, noise=0.15):
    x = v + RNG.normal(size=DIM) * noise / np.sqrt(DIM)
    return (x / np.linalg.norm(x)).astype(np.float32)


A, B, C = voice(), voice(), voice()


def test_windows_cover_audio():
    assert windows(SAMPLE_RATE) == [(0, SAMPLE_RATE)]
    spans = windows(int(4.2 * SAMPLE_RATE))
    assert spans[0][0] == 0 and spans[-1][1] == int(4.2 * SAMPLE_RATE)
    assert all(e - s == int(1.5 * SAMPLE_RATE) for s, e in spans)


def test_two_speakers_get_stable_labels():
    reg = SpeakerRegistry(threshold=0.55)
    assert reg.assign([sample(A) for _ in range(4)]) == ["P1"] * 4
    assert reg.assign([sample(B) for _ in range(4)]) == ["P2"] * 4
    assert reg.assign([sample(A), sample(A), sample(B), sample(B)]) == ["P1", "P1", "P2", "P2"]


def test_single_unmatched_window_does_not_create_speaker():
    reg = SpeakerRegistry(threshold=0.55)
    reg.assign([sample(A) for _ in range(4)])
    labels = reg.assign([sample(A), sample(B), sample(A)])
    assert labels == ["P1"] * 3 and reg.labels == ["P1"]


def test_short_segment_from_new_voice_does_not_create_speaker():
    # Ex.: vinheta/música de 2 s ("Hum!") no início do debate da Band virava P1.
    reg = SpeakerRegistry(threshold=0.55)
    assert reg.assign([sample(A), sample(A)]) == [UNKNOWN, UNKNOWN]
    assert reg.labels == []
    assert reg.assign([sample(A) for _ in range(3)]) == ["P1"] * 3  # 3 s de fala: cria
    assert reg.assign([sample(B)]) == [UNKNOWN]
    assert reg.labels == ["P1"]


def test_registry_persists(tmp_path):
    path = tmp_path / "speakers.npz"
    reg = SpeakerRegistry(threshold=0.55, path=path)
    reg.assign([sample(A)] * 3)
    reg.assign([sample(B)] * 3)
    reg.save()
    again = SpeakerRegistry(threshold=0.55, path=path)
    assert again.labels == ["P1", "P2"]
    assert again.assign([sample(B), sample(B)]) == ["P2", "P2"]
    assert again.assign([sample(C)] * 3)[0] == "P3"  # numeração continua


def test_speakers_from_another_model_are_set_aside(tmp_path):
    path = tmp_path / "speakers.npz"
    reg = SpeakerRegistry(threshold=0.55, path=path, model="modelo_antigo.onnx")
    reg.assign([sample(A)] * 3)
    reg.save()
    novo = SpeakerRegistry(threshold=0.55, path=path, model="modelo_novo.onnx")
    assert novo.labels == []
    assert (tmp_path / "speakers.modelo_antigo.npz").exists() and not path.exists()
    novo.assign([sample(B)] * 3)
    novo.save()
    assert SpeakerRegistry(threshold=0.55, path=path, model="modelo_novo.onnx").labels == ["P1"]


def test_smooth_removes_isolated_switch():
    assert smooth(["P1", "P2", "P1", "P1"]) == ["P1", "P1", "P1", "P1"]
    assert smooth(["P1", "P1", "P2", "P2"]) == ["P1", "P1", "P2", "P2"]


def test_label_words_splits_at_speaker_change():
    spans = windows(6 * SAMPLE_RATE)  # centros em 0.75, 1.5, ..., 5.25 s
    labels = ["P1" if (s + e) / 2 < 3 * SAMPLE_RATE else "P2" for s, e in spans]
    words = [Word(0.2, 0.6, " Fábio,"), Word(0.7, 1.2, " explica?"),
             Word(3.4, 3.8, " Eu"), Word(3.9, 4.5, " acho.")]
    out = label_words(words, spans, labels)
    assert [(u.speaker, u.text) for u in out] == [("P1", "Fábio, explica?"), ("P2", "Eu acho.")]
    assert out[1].start == 3.4


def test_diarizer_end_to_end_with_fake_embedder():
    # Embedder falso: a voz é decidida pela amplitude do trecho.
    embed = lambda a: sample(A) if np.abs(a).mean() < 0.5 else sample(B)  # noqa: E731
    audio = np.concatenate([np.full(3 * SAMPLE_RATE, 0.1), np.full(3 * SAMPLE_RATE, 0.9)]).astype(np.float32)
    words = [Word(0.5, 1.0, " pergunta"), Word(4.5, 5.0, " resposta")]
    out = Diarizer(embed, SpeakerRegistry())(audio, words)
    assert [(u.speaker, u.text) for u in out] == [("P1", "pergunta"), ("P2", "resposta")]


def test_close_voices_do_not_flip_or_merge():
    # Duas vozes parecidas (mesmo estúdio): cosseno ~0.5 entre si.
    c, u, w = np.linalg.qr(RNG.normal(size=(DIM, 3)))[0].T  # ortonormais
    a, b = (c + u) / np.sqrt(2), (c + w) / np.sqrt(2)  # a·b = 0.5
    reg = SpeakerRegistry(threshold=0.65)
    reg.assign([sample(a, 0.3) for _ in range(6)])
    reg.assign([sample(b, 0.3) for _ in range(6)])
    assert reg.labels == ["P1", "P2"]
    before = float(reg.centroids[0] @ reg.centroids[1])

    # Janelas ambíguas (meio-termo entre as vozes) mantêm o falante atual e não puxam os centróides.
    mid = (a + b) / np.linalg.norm(a + b)
    labels = reg.assign([sample(a, 0.3)] * 3 + [sample(mid, 0.1) for _ in range(6)])
    assert set(labels) == {"P1"}
    after = float(reg.centroids[0] @ reg.centroids[1])
    assert after <= before + 0.02
