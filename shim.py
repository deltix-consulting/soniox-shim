"""soniox-shim — an OpenAI-compatible `/v1/audio/transcriptions` endpoint in front of Soniox.

Each request is one Soniox async job: upload the file, create a transcription, poll until it is
done, fetch the tokens, delete both again. The tokens come back as Whisper-style `verbose_json`
(sentence-shaped segments with word timestamps). Configuration is environment only; see README.md.
"""

import asyncio
import io
import logging
import os
import secrets
import time
import wave
from collections import Counter
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

SEGMENT_GAP_S = 1.0  # a pause longer than this starts a new segment
POLL_S = 0.4

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


def wav_duration(data: bytes) -> float:
    """Seconds of audio if this is a WAV we can read, else 0 (duration is informational only)."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        return 0.0


def _checked(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise SonioxError(r.status_code, r.text[:200])
    return r.json() if r.content else {}


async def _cleanup(client_args: dict, transcription_id: str | None, file_id: str | None) -> None:
    """Best effort, own budget: leave no audio or transcript behind at Soniox."""
    try:
        async with asyncio.timeout(10), httpx.AsyncClient(**client_args, timeout=10) as c:
            paths = [f"/transcriptions/{transcription_id}"] if transcription_id else []
            paths += [f"/files/{file_id}"] if file_id else []
            await asyncio.gather(*(c.delete(p) for p in paths))
    except (httpx.HTTPError, TimeoutError) as e:
        log.warning("cleanup failed transcription=%s file=%s: %s", transcription_id, file_id, e)


async def soniox_transcribe(audio: bytes, filename: str, hints: list[str]) -> list[dict]:
    """Upload → create transcription → poll → fetch tokens. Raises HTTPException on any failure."""
    client_args = {
        "base_url": env("SONIOX_URL", "https://api.soniox.com/v1"),
        "headers": {"Authorization": f"Bearer {env('SONIOX_API_KEY')}"},
    }
    timeout = float(env("SONIOX_TIMEOUT_S", "25"))
    file_id = transcription_id = None
    try:
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(**client_args, timeout=timeout) as c,
        ):
            r = await c.post(
                "/files", files={"file": (filename, audio, "application/octet-stream")}
            )
            file_id = _checked(r)["id"]
            body: dict = {"file_id": file_id, "model": env("SONIOX_MODEL", "stt-async-v5")}
            if hints:
                body["language_hints"] = hints
            job = _checked(await c.post("/transcriptions", json=body))
            transcription_id = job["id"]
            while job.get("status") not in ("completed", "error"):
                await asyncio.sleep(POLL_S)
                job = _checked(await c.get(f"/transcriptions/{transcription_id}"))
            if job["status"] == "error":
                raise SonioxError(502, job.get("error_message", "transcription failed"))
            return _checked(await c.get(f"/transcriptions/{transcription_id}/transcript")).get(
                "tokens", []
            )
    except TimeoutError as e:
        raise HTTPException(504, "soniox: no result within SONIOX_TIMEOUT_S") from e
    except httpx.HTTPError as e:
        raise HTTPException(503, f"soniox unavailable: {e}") from e
    except SonioxError as e:
        status = e.code if e.code in (400, 401, 402, 429) else 502
        raise HTTPException(status, str(e)) from e
    finally:
        if file_id:
            await _cleanup(client_args, transcription_id, file_id)


def to_words(tokens: list[dict]) -> list[dict]:
    """Soniox sub-word tokens → words. A leading space starts a word; anything else continues it."""
    words: list[dict] = []
    for t in tokens:
        text = t.get("text", "")
        if not text.strip() or t.get("is_audio_event") or text.strip().startswith("<"):
            continue  # silence, "(laughter)"-style events, <end>/<fin> markers
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
    return {"status": "ok", "model": env("SONIOX_MODEL", "stt-async-v5")}


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
    duration = wav_duration(data)
    # DECISION: the request's language comes first, then LANGUAGE_HINTS (e.g. "nl,en" so a Dutch
    # meeting with English terms is not forced into one language). Duplicates dropped.
    hints = list(dict.fromkeys([h for h in [language, *env("LANGUAGE_HINTS").split(",")] if h]))

    t0 = time.monotonic()
    tokens = await soniox_transcribe(data, file.filename or "audio.wav", hints)
    words = to_words(tokens)
    segments = to_segments(words)
    text = " ".join(s["text"] for s in segments)
    langs = Counter(w["language"] for w in words if w.get("language"))
    detected = langs.most_common(1)[0][0] if langs else (language or "unknown")
    audio_seconds_total += duration
    log.info(
        "ok ms=%d audio_s=%.1f words=%d lang=%s audio_s_total=%.0f",
        (time.monotonic() - t0) * 1000,
        duration,
        len(words),
        detected,
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
