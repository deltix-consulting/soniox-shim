import io
import wave

import pytest
from fastapi.testclient import TestClient

from shim import app, to_segments, to_words
from tests.fake_soniox import FakeSoniox


def wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def tok(text, start, end, final=True, conf=0.9, lang="nl"):
    return {
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "is_final": final,
        "confidence": conf,
        "language": lang,
    }


SCRIPT = [
    {
        "tokens": [
            tok("Hal", 0, 100),
            tok("lo", 100, 200),
            tok(" wereld", 200, 500),
            tok(".", 500, 520),
        ]
    },
    {"tokens": [tok(" dit", 600, 700, final=False)]},  # non-final: must be ignored
    {"tokens": [tok(" Dit", 600, 700), tok(" werkt", 700, 900, conf=0.4), tok("<end>", 900, 900)]},
    {"tokens": [], "finished": True},
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "test-key")
    monkeypatch.setenv("LANGUAGE_HINTS", "nl,en")
    monkeypatch.delenv("SHIM_API_TOKEN", raising=False)
    return TestClient(app)


def post(client, data, **form):
    form.setdefault("model", "whisper-1")
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", data, "audio/wav")},
        data=form,
    )


def test_verbose_json_matches_whisper_shape(client, monkeypatch):
    with FakeSoniox(SCRIPT) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        r = post(client, wav_bytes(), response_format="verbose_json", language="nl")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Hallo wereld. Dit werkt"
    assert body["language"] == "nl"
    assert body["duration"] == pytest.approx(0.5)
    assert [s["text"] for s in body["segments"]] == ["Hallo wereld.", "Dit werkt"]
    seg = body["segments"][0]
    assert seg["start"] == 0.0 and seg["end"] == 0.52
    assert seg["words"] == [
        {"word": "Hallo", "start": 0.0, "end": 0.2, "probability": 0.9},
        {"word": "wereld.", "start": 0.2, "end": 0.52, "probability": 0.9},
    ]
    assert body["segments"][1]["words"][1]["probability"] == 0.4
    # what Soniox saw
    assert fake.config["model"] == "stt-rt-v5"
    assert fake.config["api_key"] == "test-key"
    assert fake.config["audio_format"] == "pcm_s16le"
    assert fake.config["sample_rate"] == 16000 and fake.config["num_channels"] == 1
    assert fake.config["language_hints"] == ["nl", "en"]
    assert fake.audio == b"\x00\x00" * 8000  # raw frames, header stripped


def test_json_and_text_formats(client, monkeypatch):
    with FakeSoniox(SCRIPT) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        assert post(client, wav_bytes()).json() == {"text": "Hallo wereld. Dit werkt"}
        assert post(client, wav_bytes(), response_format="text").text == "Hallo wereld. Dit werkt"


def test_non_wav_is_forwarded_as_auto(client, monkeypatch):
    with FakeSoniox([{"tokens": [], "finished": True}]) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        r = post(client, b"OggS not really audio", response_format="verbose_json")
    assert r.status_code == 200
    assert r.json() == {
        "task": "transcribe",
        "language": "unknown",
        "duration": 0.0,
        "text": "",
        "segments": [],
    }
    assert fake.config["audio_format"] == "auto"
    assert fake.audio == b"OggS not really audio"


def test_soniox_error_codes_map_to_http(client, monkeypatch):
    with FakeSoniox([{"error_code": 402, "error_message": "balance exhausted"}]) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        r = post(client, wav_bytes())
    assert r.status_code == 402
    assert "balance exhausted" in r.json()["detail"]
    with FakeSoniox([{"error_code": 500, "error_message": "boom"}]) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        assert post(client, wav_bytes()).status_code == 502


def test_unreachable_soniox_is_503(client, monkeypatch):
    monkeypatch.setenv("SONIOX_URL", "ws://127.0.0.1:1")
    assert post(client, wav_bytes()).status_code == 503


def test_timeout_is_504(client, monkeypatch):
    monkeypatch.setenv("SONIOX_TIMEOUT_S", "0.3")
    with FakeSoniox([], hang=True) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        assert post(client, wav_bytes()).status_code == 504


def test_empty_file_is_400(client):
    assert post(client, b"").status_code == 400


def test_bearer_token(client, monkeypatch):
    monkeypatch.setenv("SHIM_API_TOKEN", "s3cret")
    assert post(client, wav_bytes()).status_code == 401
    with FakeSoniox([{"tokens": [], "finished": True}]) as fake:
        monkeypatch.setenv("SONIOX_URL", fake.url)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
            data={"model": "whisper-1"},
            headers={"Authorization": "Bearer s3cret"},
        )
    assert r.status_code == 200


def test_health(client, monkeypatch):
    assert client.get("/health").status_code == 200
    monkeypatch.delenv("SONIOX_API_KEY")
    assert client.get("/health").status_code == 503


def test_segments_split_on_pause():
    words = to_words([tok(" een", 0, 100), tok(" twee", 1500, 1600), tok(" drie", 1700, 1800)])
    assert [s["text"] for s in to_segments(words)] == ["een", "twee drie"]
    assert to_segments([]) == []
