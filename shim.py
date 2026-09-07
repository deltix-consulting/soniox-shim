"""soniox-shim — an OpenAI-compatible `/v1/audio/transcriptions` endpoint in front of Soniox.

Each request opens one Soniox real-time WebSocket session, streams the audio, waits for the
final tokens and returns them as Whisper-style `verbose_json` (segments with word timestamps).
Configuration is environment only; see README.md.
"""

import asyncio
import io
import json
import logging
import os
import secrets
import time
import wave
from collections import Counter
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

SEGMENT_GAP_S = 1.0  # a pause longer than this starts a new segment
WS_CHUNK = 64 * 1024

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("soniox-shim")
app = FastAPI(title="soniox-shim", docs_url=None, redoc_url=None)
audio_seconds_total = 0.0  # audio sent to Soniox since start — the number your bill follows


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class SonioxError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"soniox {code}: {message}")
        self.code, self.message = code, message


def unpack_audio(data: bytes) -> tuple[dict, bytes, float]:
    """16-bit PCM WAV → (pcm_s16le config, raw frames, duration). Anything else → Soniox 'auto'."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() == 2 and w.getcomptype() == "NONE":
                fmt = {
                    "audio_format": "pcm_s16le",
                    "sample_rate": w.getframerate(),
                    "num_channels": w.getnchannels(),
                }
                return fmt, w.readframes(w.getnframes()), w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001 — not a WAV we understand; let Soniox detect it
        pass
    return {"audio_format": "auto"}, data, 0.0


async def soniox_transcribe(fmt: dict, audio: bytes, hints: list[str]) -> list[dict]:
    """One real-time session: config → audio → end-of-stream → final tokens until `finished`."""
    cfg = {"api_key": env("SONIOX_API_KEY"), "model": env("SONIOX_MODEL", "stt-rt-v5"), **fmt}
    if hints:
        cfg["language_hints"] = hints
    url = env("SONIOX_URL", "wss://stt-rt.soniox.com/transcribe-websocket")
    tokens: list[dict] = []
    try:
        async with asyncio.timeout(float(env("SONIOX_TIMEOUT_S", "25"))):
            async with connect(url, max_size=None) as ws:
                await ws.send(json.dumps(cfg))
                for i in range(0, len(audio), WS_CHUNK):
                    await ws.send(audio[i : i + WS_CHUNK])
                await ws.send("")  # empty frame = end of audio
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("error_code"):
                        raise SonioxError(int(msg["error_code"]), msg.get("error_message", ""))
                    tokens += [t for t in msg.get("tokens", []) if t.get("is_final")]
                    if msg.get("finished"):
                        break
    except TimeoutError as e:
        raise HTTPException(504, "soniox: no result within SONIOX_TIMEOUT_S") from e
    except (OSError, WebSocketException) as e:
        raise HTTPException(503, f"soniox unavailable: {e}") from e
    except SonioxError as e:
        status = e.code if e.code in (400, 401, 402, 429) else 502
        raise HTTPException(status, str(e)) from e
    return tokens


def to_words(tokens: list[dict]) -> list[dict]:
    """Soniox sub-word tokens → words. A leading space starts a word; anything else continues it."""
    words: list[dict] = []
    for t in tokens:
        text = t.get("text", "")
        if not text.strip() or (text.startswith("<") and text.endswith(">")):  # <end>, <fin>
            continue
        conf = float(t.get("confidence", 1.0))
        if words and not text[0].isspace():
            w = words[-1]
            w["word"] += text.strip()
            w["end"] = t["end_ms"] / 1000
            w["probability"] = min(w["probability"], conf)
        else:
            words.append(
                {
                    "word": text.strip(),
                    "start": t["start_ms"] / 1000,
                    "end": t["end_ms"] / 1000,
                    "probability": conf,
                    "language": t.get("language"),
                }
            )
    return words


def to_segments(words: list[dict]) -> list[dict]:
    """Sentence-shaped segments: split after .?! or on a pause > SEGMENT_GAP_S."""
    segments: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            segments.append(
                {
                    "id": len(segments),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": " ".join(w["word"] for w in current),
                    "words": [
                        {k: w[k] for k in ("word", "start", "end", "probability")} for w in current
                    ],
                }
            )
            current.clear()

    for w in words:
        if current and w["start"] - current[-1]["end"] > SEGMENT_GAP_S:
            flush()
        current.append(w)
        if w["word"][-1:] in ".?!":
            flush()
    flush()
    return segments


def require_token(authorization: str | None) -> None:
    expected = env("SHIM_API_TOKEN")
    if expected and not secrets.compare_digest(authorization or "", f"Bearer {expected}"):
        raise HTTPException(401, "invalid bearer token")


@app.get("/health")
def health() -> dict:
    if not env("SONIOX_API_KEY"):
        raise HTTPException(503, "SONIOX_API_KEY not set")
    return {"status": "ok", "model": env("SONIOX_MODEL", "stt-rt-v5")}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "whisper-1",  # accepted for compatibility, ignored
    response_format: Annotated[str, Form()] = "json",
    language: Annotated[str | None, Form()] = None,
    authorization: Annotated[str | None, Header()] = None,
):
    global audio_seconds_total
    require_token(authorization)
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    fmt, audio, duration = unpack_audio(data)
    # DECISION: the request's language comes first, then LANGUAGE_HINTS (e.g. "nl,en" so a Dutch
    # meeting with English terms is not forced into one language). Duplicates dropped.
    hints = list(dict.fromkeys([h for h in [language, *env("LANGUAGE_HINTS").split(",")] if h]))

    t0 = time.monotonic()
    tokens = await soniox_transcribe(fmt, audio, hints)
    words = to_words(tokens)
    segments = to_segments(words)
    text = " ".join(s["text"] for s in segments)
    langs = Counter(w["language"] for w in words if w.get("language"))
    detected = langs.most_common(1)[0][0] if langs else (language or "unknown")
    audio_seconds_total += duration
    log.info(
        "ok ms=%d audio_s=%.1f words=%d lang=%s fmt=%s audio_s_total=%.0f",
        (time.monotonic() - t0) * 1000,
        duration,
        len(words),
        detected,
        fmt["audio_format"],
        audio_seconds_total,
    )

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": detected,
            "duration": duration,
            "text": text,
            "segments": segments,
        }
    return {"text": text}
