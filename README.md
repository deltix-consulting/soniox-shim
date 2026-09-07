# soniox-shim

A tiny OpenAI-compatible speech-to-text endpoint (`POST /v1/audio/transcriptions`) that forwards to [Soniox](https://soniox.com) real-time STT. Point any client that speaks the Whisper API at it: [Vexa](https://github.com/Vexa-ai/vexa), Open WebUI, LibreChat, your own code.

One file, no database, no state. Apache 2.0.

## How it works

Every request opens one Soniox real-time WebSocket session, streams the audio, sends end-of-stream, collects the **final** tokens and returns them in Whisper's `verbose_json` shape: sentence-shaped segments with word-level timestamps and a per-word probability (Soniox's confidence).

The real-time API is used rather than Soniox's async file API because callers like Vexa send short, growing audio windows (1–30 s) every couple of seconds and expect an answer within their request timeout. Upload → poll → fetch would not fit.

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
| `SONIOX_MODEL` | `stt-rt-v5` | Soniox real-time model id. |
| `LANGUAGE_HINTS` | | Comma-separated ISO codes passed as `language_hints`, e.g. `nl,en`. The request's `language` is put in front. |
| `SHIM_API_TOKEN` | | Optional. When set, requests must carry `Authorization: Bearer <token>`. |
| `SONIOX_TIMEOUT_S` | `25` | Whole-request budget for the Soniox session. Returns 504 when exceeded. |
| `SONIOX_URL` | `wss://stt-rt.soniox.com/transcribe-websocket` | Override for tests. |
| `LOG_LEVEL` | `INFO` | |

## Request and response

Accepted multipart fields: `file` (16-bit PCM WAV is sent as raw `pcm_s16le`; other formats are passed through with Soniox's `auto` detection), `model` (ignored), `response_format` (`json` default, `verbose_json`, `text`), `language`. Other Whisper fields (`prompt`, `timestamp_granularities`, …) are accepted and ignored.

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

Put both stacks on the same Docker network (`STT_NETWORK` in `docker-compose.yml`, default `vexa_default`).

**Cost note.** Vexa resubmits the unconfirmed tail of each turn every ~2 s, so each second of meeting audio is transcribed more than once. Every log line carries `audio_s_total`, the seconds of audio sent to Soniox since start; compare it with real meeting minutes to know your multiplier before you rely on a price estimate.

## Development

```bash
uv run ruff check . && uv run pytest -q
```

Tests run against a fake Soniox WebSocket server in `tests/fake_soniox.py`; no API key needed.
