"""soniox-shim — an OpenAI-compatible `/v1/audio/transcriptions` endpoint in front of Soniox.

Each request is one Soniox async job: upload the file, create a transcription, poll until it is
done, fetch the tokens, delete both again. The tokens come back as Whisper-style `verbose_json`
(sentence-shaped segments with word timestamps). Configuration is environment only; see README.md.
"""

import asyncio
import io
import logging
import math
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
POLL_RETRIES = 3  # transient 429/5xx on a status poll must not kill a running (billed) job
CLOSERS = "\"'”’)]»"  # closing punctuation that may follow the sentence-ending mark
CONTEXT_MAX_CHARS = 10_000  # Soniox context limit (~8k tokens)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("soniox-shim")
app = FastAPI(title="soniox-shim", docs_url=None, redoc_url=None)
audio_seconds_total = 0.0  # audio sent to Soniox since start — the number your bill follows
_background: set[asyncio.Task] = set()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@app.on_event("startup")
async def warn_open_auth() -> None:
    if not env("SHIM_API_TOKEN"):
        log.warning("SHIM_API_TOKEN is empty: anyone on the network can use this shim's Soniox key")


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
    try:
        return r.json() if r.content else {}
    except ValueError as e:  # a proxy page instead of JSON
        raise SonioxError(502, f"non-JSON response from Soniox: {r.text[:100]!r}") from e


async def _cleanup(client_args: dict, transcription_id: str | None, file_id: str | None) -> None:
    """Best effort, own budget, off the request path: leave nothing behind at Soniox."""
    try:
        async with asyncio.timeout(10), httpx.AsyncClient(**client_args, timeout=10) as c:
            paths = [f"/transcriptions/{transcription_id}"] if transcription_id else []
            paths += [f"/files/{file_id}"] if file_id else []
            await asyncio.gather(*(c.delete(p) for p in paths))
    except (httpx.HTTPError, TimeoutError) as e:
        log.warning("cleanup failed transcription=%s file=%s: %s", transcription_id, file_id, e)


def _schedule_cleanup(client_args: dict, transcription_id: str | None, file_id: str | None) -> None:
    task = asyncio.create_task(_cleanup(client_args, transcription_id, file_id))
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _poll(c: httpx.AsyncClient, transcription_id: str) -> dict:
    """One status read, tolerant of a transient 429/5xx (the job itself keeps running)."""
    for attempt in range(POLL_RETRIES + 1):
        try:
            return _checked(await c.get(f"/transcriptions/{transcription_id}"))
        except (SonioxError, httpx.HTTPError) as e:
            transient = isinstance(e, httpx.HTTPError) or e.code == 429 or e.code >= 500
            if not transient or attempt == POLL_RETRIES:
                raise
            await asyncio.sleep(1.0)
    raise AssertionError("unreachable")


async def soniox_transcribe(
    audio: bytes, filename: str, hints: list[str], prompt: str
) -> list[dict]:
    """Upload → create transcription → poll → fetch tokens. Raises HTTPException on any failure."""
    client_args = {
        "base_url": env("SONIOX_URL", "https://api.soniox.com/v1"),
        "headers": {"Authorization": f"Bearer {env('SONIOX_API_KEY')}"},
    }
    timeout = float(env("SONIOX_TIMEOUT_S", "20"))
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
            if prompt:
                # Whisper's `prompt` (the caller's already-confirmed text) → Soniox context text.
                body["context"] = {"text": prompt[-CONTEXT_MAX_CHARS:]}
            job = _checked(await c.post("/transcriptions", json=body))
            transcription_id = job["id"]
            while job.get("status") not in ("completed", "error"):
                await asyncio.sleep(POLL_S)
                job = await _poll(c, transcription_id)
            if job["status"] == "error":
                # The job failed on this input; it will fail again on a retry → final (400).
                raise SonioxError(400, job.get("error_message", "transcription failed"))
            return _checked(await c.get(f"/transcriptions/{transcription_id}/transcript")).get(
                "tokens", []
            )
    except TimeoutError as e:
        raise HTTPException(504, "soniox: no result within SONIOX_TIMEOUT_S") from e
    except httpx.HTTPError as e:
        raise HTTPException(503, f"soniox unavailable: {e}") from e
    except SonioxError as e:
        # 4xx are final for the caller (bad key, model, payload); 5xx come back as 502 = retry.
        status = e.code if e.code < 500 else 502
        raise HTTPException(status, str(e)) from e
    finally:
        if file_id:
            _schedule_cleanup(client_args, transcription_id, file_id)


def to_words(tokens: list[dict]) -> list[dict]:
    """Soniox sub-word tokens → words in Whisper's shape: every word carries its leading space
    (consumers concatenate `word` verbatim). A token with a leading space, a whitespace-only token,
    or a trailing space on the previous token starts a new word; anything else continues it."""
    words: list[dict] = []
    boundary = True
    for t in tokens:
        text = t.get("text", "")
        if not text:
            continue
        if not text.strip():  # whitespace-only token: a boundary, nothing to add
            boundary = True
            continue
        if t.get("is_audio_event") or text.strip().startswith("<"):  # "(laughter)", <end>, <fin>
            boundary = True
            continue
        if t.get("start_ms") is None or t.get("end_ms") is None:
            continue
        conf = float(t.get("confidence", 1.0))
        if words and not boundary and not text[0].isspace():
            w = words[-1]
            w["word"] += text.strip()
            w["end"] = t["end_ms"] / 1000
            w["probability"] = min(w["probability"], conf)
        else:
            words.append(
                {
                    "word": " " + text.strip(),
                    "start": t["start_ms"] / 1000,
                    "end": t["end_ms"] / 1000,
                    "probability": conf,
                    "language": t.get("language"),
                }
            )
        boundary = text[-1].isspace()
    return words


def _ends_sentence(word: str) -> bool:
    return word.rstrip(CLOSERS)[-1:] in ".?!"


def to_segments(words: list[dict]) -> list[dict]:
    """Sentence-shaped segments: split after .?! (closing quotes allowed) or on a pause >
    SEGMENT_GAP_S. `avg_logprob` is derived from Soniox confidence so Whisper-style consumers can
    still drop low-confidence segments; no_speech_prob / compression_ratio are not available."""
    segments: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            probs = [max(w["probability"], 1e-6) for w in current]
            segments.append(
                {
                    "id": len(segments),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": "".join(w["word"] for w in current).strip(),
                    "avg_logprob": round(math.log(sum(probs) / len(probs)), 4),
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
        if _ends_sentence(w["word"]):
            flush()
    flush()
    return segments


def require_token(authorization: str | None) -> None:
    expected = env("SHIM_API_TOKEN")
    if expected and not secrets.compare_digest(
        (authorization or "").encode("utf-8", "replace"), f"Bearer {expected}".encode()
    ):
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
    prompt: Annotated[str | None, Form()] = None,
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
    raw = [language, *env("LANGUAGE_HINTS").split(",")]
    hints = list(dict.fromkeys(h.strip() for h in raw if h and h.strip()))

    t0 = time.monotonic()
    tokens = await soniox_transcribe(data, file.filename or "audio.wav", hints, prompt or "")
    words = to_words(tokens)
    segments = to_segments(words)
    text = " ".join(s["text"] for s in segments)
    langs = Counter(w["language"] for w in words if w.get("language"))
    detected = langs.most_common(1)[0][0] if langs else language  # None when nobody knows
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
