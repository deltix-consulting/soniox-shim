# soniox-shim

A tiny OpenAI-compatible speech-to-text endpoint (`POST /v1/audio/transcriptions`) that forwards to [Soniox](https://soniox.com) speech-to-text. Point any client that speaks the Whisper API at it: [Vexa](https://github.com/Vexa-ai/vexa), Open WebUI, LibreChat, your own code.

One file, no database, no state. Apache 2.0.

## How it works

Every request is one Soniox async job: upload the file, create a transcription, poll until it completes, fetch the tokens, then delete the file and the transcription again so nothing stays behind at Soniox. The tokens are returned in Whisper's `verbose_json` shape: sentence-shaped segments with word-level timestamps and a per-word probability (Soniox's confidence).

Why the async API and not the real-time WebSocket: Soniox real-time processes at roughly real-time pace, so a 14 s clip takes ~12 s to come back. The async model runs faster than real time with a fixed overhead of a few seconds. Measured with a 16 kHz Dutch clip:

| Clip | Real-time API | Async API |
|---|---|---|
| 2 s | 2.6 s | 4.8 s |
| 14 s | 12.3 s | 7.7 s |

Callers like Vexa send windows of up to ~30 s with a 30 s timeout, so async is the one that fits. It is also cheaper per hour and was more accurate on the same clip.

## Run

```bash
cp .env.example .env    # set SONIOX_API_KEY
docker compose up -d
```

Or locally:

```bash
uv sync && uv run uvicorn shim:app --port 8083
```

Smoke test with any 16 kHz mono WAV:

```bash
curl -s http://localhost:8083/v1/audio/transcriptions \
  -F file=@sample.wav -F model=whisper-1 -F response_format=verbose_json -F language=nl | jq .
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SONIOX_API_KEY` | | Required. `GET /health` returns 503 without it. |
| `SONIOX_MODEL` | `stt-async-v5` | Soniox async model id. |
| `LANGUAGE_HINTS` | | Comma-separated ISO codes passed as `language_hints`, e.g. `nl,en`. The request's `language` is put in front. |
| `SHIM_API_TOKEN` | | Optional. When set, requests must carry `Authorization: Bearer <token>`. |
| `SONIOX_TIMEOUT_S` | `25` | Whole-request budget for upload, polling and fetch. Returns 504 when exceeded. |
| `SONIOX_URL` | `https://api.soniox.com/v1` | Override for tests. |
| `LOG_LEVEL` | `INFO` | |

## Request and response

Accepted multipart fields: `file` (any format Soniox accepts; passed through unchanged), `model` (ignored), `response_format` (`json` default, `verbose_json`, `text`), `language`. Other Whisper fields (`prompt`, `timestamp_granularities`, …) are accepted and ignored.

`verbose_json`:

```json
{
  "task": "transcribe",
  "language": "nl",
  "duration": 4.2,
  "text": "Hallo wereld. Dit werkt",
  "segments": [
    {"id": 0, "start": 0.0, "end": 0.52, "text": "Hallo wereld.",
     "words": [{"word": "Hallo", "start": 0.0, "end": 0.2, "probability": 0.9}, …]}
  ]
}
```

Segments split after `.`, `?`, `!` or on a pause longer than one second. Whisper-only fields (`avg_logprob`, `no_speech_prob`, `compression_ratio`) are not returned; clients that filter on them keep the segment.

Errors: Soniox `400/401/402/429` are returned as-is, other Soniox errors as `502`, unreachable as `503`, timeout as `504`. Retrying clients treat 5xx as transient and 401/402 as final, which is what you want.

## Wiring it into Vexa

In Vexa's `.env`:

```
TRANSCRIPTION_SERVICE_URL=http://soniox-shim:8000
TRANSCRIPTION_SERVICE_TOKEN=<same value as SHIM_API_TOKEN>
TRANSCRIPTION_MODEL=
```

Put both stacks on the same Docker network (`STT_NETWORK` in `docker-compose.yml`, default `vexa-v012_vexa`, the network of Vexa v0.12's compose project).

**Cost note.** Vexa resubmits the unconfirmed tail of each turn every ~2 s, so each second of meeting audio is transcribed more than once. Every log line carries `audio_s_total`, the seconds of audio sent to Soniox since start; compare it with real meeting minutes to know your multiplier before you rely on a price estimate.

## Development

```bash
uv run ruff check . && uv run pytest -q
```

Tests run against a fake Soniox REST server in `tests/fake_soniox.py`; no API key needed.
