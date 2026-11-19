# Voice Travel Agent

Data layer and services for the voice travel agent (architecture: `references/`).

## Layout

```
src/voice_travel/
  config.py          single Settings class (the only module that reads env/.env)
  container.py       composition root: wires concrete implementations
  domain/            pure business rules, no I/O (pricing)
  infra/postgres/    Database (injected), ORM models = the ONE schema definition
  infra/redis/       client + keys.py (the ONLY place key names live)
  services/          use-cases: identity, trips, bookings, sessions, ops
  seeding/           seed-postgres / seed-redis commands
  voice/             voice layer: events.py (voice<->agent contract), bus.py (Redis Streams),
                     transport.py + livekit_transport.py, vad.py + segmenter.py (Silero, utterances),
                     stt.py (Faster-Whisper), pipeline.py (one call), worker.py, tokens.py, api.py, static/demo.html
migrations/          Alembic (schema history)
data/                synthetic dataset generator, validator and output (data/out)
tests/               unit/ (no I/O) and integration/ (real dataset, SQLite or Postgres)
```

## Setup

```powershell
pip install -e ".[dev,voice]"
docker compose up -d                 # postgres, redis, qdrant, minio, livekit (local dev)
# put secrets in .env (git-ignored): DATABASE_URL=..., REDIS_URL=...
alembic upgrade head                 # create the schema
seed-postgres                        # load data/out into Postgres
seed-redis                           # load mock live state into Redis (--clear to remove)
```

## Quality gates

```powershell
ruff check src tests ; mypy ; pytest
```

Integration tests run on in-memory SQLite by default. To exercise Postgres-only behaviour
(row locks, jsonb), point `TEST_DATABASE_URL` at a **scratch** Postgres database; the suite
drops and recreates the app tables.

## Dataset

```powershell
python data/generate.py              # deterministic (seed 42), writes data/out
python data/validate.py              # integrity + balance report
```

## Voice layer

The voice layer is an always-listening audio pipe. It reports what it heard and played as events and only
speaks what the agent sends. The contract is `voice/events.py`; authority flows agent -> voice layer ->
transport. Conversation state belongs to the agent, not here.

```
INGRESS  caller mic -> LiveKit -> Silero VAD -> segmenter -> Faster-Whisper -> utterance.partial / .final
EGRESS   agent speak -> Piper TTS (per sentence) -> resample -> LiveKit track -> caller      (stops on barge-in)
CONTROL  agent cancel_speech / tick_control / hangup;  events on one Redis Stream per call and direction
```

What it guarantees (each one is a test in `tests/unit/test_voice_contract.py`): listening never pauses (also
while the agent speaks or STT is busy); barge-in is always on and cuts playback and drops queued speech;
every cut or finished message is reported with `completed` and `played_ms`; it never speaks on its own
(errors go out silently as `transport.error`); ticks flow only while the agent asked for them.

```powershell
fetch-voice-models                   # one-off: Silero VAD + Piper voice into models/ (SHA-256 verified)
docker compose up -d livekit redis   # LiveKit uses the dev key in livekit.yaml
voice-worker                         # joins announced calls (loads Whisper + Piper; first run downloads Whisper)
voice-echo-agent                     # DEV stand-in agent: replies "You said: ..." (replace with the real agent)
voice-api                            # http://localhost:8000/demo  (needs REDIS_URL)
```

Open `/demo`, press Start call, speak, hear the reply, talk over it to interrupt. `POST /api/v1/voice/session
{"user_id": "U0001"}` returns `{session_id, room, url, token}`; `/api/v1/voice/events/{id}?token=...` streams the
call's events (the join token is required: events contain transcripts).

The real agent integrates through the bus only: read `utterance.final` etc. from
`vta:call:<id>:events:up`, write `speak` / `cancel_speech` / `tick_control` / `hangup` to `...:events:down`
(see `voice/dev_echo_agent.py` for the minimal pattern).

Settings (`.env`): `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (override outside local dev),
`STT_MODEL` (default `small.en`), `STT_DEVICE`, `STT_COMPUTE_TYPE`, `STT_PARTIALS`, `TTS_VOICE_PATH`.

Not built: recording, metrics/dashboards, load testing, an Asterisk/SIP transport adapter.
