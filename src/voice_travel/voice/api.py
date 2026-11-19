"""Voice API: creates a call (session row + LiveKit room + join token) and ends it.

Plain `def` endpoints on purpose: the data layer is synchronous, so FastAPI runs them in its threadpool.
The API only mints access; all audio and events flow through LiveKit and the event bus, never through here."""
import asyncio
import hmac
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import redis
import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from livekit import api as lk_api
from pydantic import BaseModel

from voice_travel.config import get_settings
from voice_travel.container import Container, build_container
from voice_travel.infra.postgres import User
from voice_travel.logging import configure_logging
from voice_travel.services import ServiceError
from voice_travel.voice.bus import START, EventBus, RedisStreamBus, announce_call
from voice_travel.voice.events import encode
from voice_travel.voice.tokens import create_join_token, room_name

logger = logging.getLogger(__name__)
DEMO_PAGE = Path(__file__).parent / "static" / "demo.html"
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011


class CreateSessionRequest(BaseModel):
    user_id: str
    trip_id: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str          # doubles as the call_id on the event bus
    room: str
    url: str                 # LiveKit signalling URL the client connects to
    token: str               # short-lived JWT for this room only


class EndSessionResponse(BaseModel):
    session_id: str
    status: str


def create_app(container: Container, *, bus: EventBus | None = None,
               announce: Callable[[str], None] | None = None) -> FastAPI:
    """bus: where the events WebSocket reads from. announce: tells a voice worker a call needs it."""
    app = FastAPI(title="Voice Travel Agent: voice API")
    api_key = container.settings.voice_api_key

    def require_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
        """Session endpoints mint access, so they need the bearer key whenever one is configured."""
        if api_key is None:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode(),
                                                                  api_key.get_secret_value().encode()):
            raise HTTPException(status_code=401, detail="missing or invalid API key",
                                headers={"WWW-Authenticate": "Bearer"})

    def caller_may_watch(session_id: str, token: str) -> bool:
        """Events carry transcripts: only the holder of this call's join token may read them."""
        try:
            claims = lk_api.TokenVerifier(container.settings.livekit_api_key,
                                          container.settings.livekit_api_secret.get_secret_value()).verify(token)
        except Exception:
            return False
        return claims.video is not None and claims.video.room == room_name(session_id)

    @app.get("/demo", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(DEMO_PAGE)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/voice/session", response_model=CreateSessionResponse,
              dependencies=[Depends(require_api_key)])
    def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
        try:
            with container.db.session() as s:
                if s.get(User, req.user_id) is None:
                    raise HTTPException(status_code=404, detail="unknown user")
            voice_session = container.sessions.start(req.user_id, req.trip_id)
        except ServiceError as exc:
            logger.error("create_session failed: %s", exc)
            raise HTTPException(status_code=503, detail="session store unavailable") from exc
        token = create_join_token(container.settings, call_id=voice_session.id, identity=req.user_id)
        if announce is not None:
            try:
                announce(voice_session.id)
            except redis.RedisError as exc:
                logger.error("could not announce call %s: %s", voice_session.id, type(exc).__name__)
                container.sessions.end(voice_session.id)
                raise HTTPException(status_code=503, detail="voice workers unreachable") from exc
        return CreateSessionResponse(session_id=voice_session.id, room=room_name(voice_session.id),
                                     url=container.settings.livekit_url, token=token)

    @app.post("/api/v1/voice/session/{session_id}/end", response_model=EndSessionResponse,
              dependencies=[Depends(require_api_key)])
    def end_session(session_id: str) -> EndSessionResponse:
        try:
            ended = container.sessions.end(session_id)
        except ServiceError as exc:
            logger.error("end_session failed: %s", exc)
            raise HTTPException(status_code=503, detail="session store unavailable") from exc
        if ended is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return EndSessionResponse(session_id=ended.id, status=ended.status)

    @app.websocket("/api/v1/voice/events/{session_id}")
    async def events(ws: WebSocket, session_id: str, token: str = "") -> None:
        if bus is None:
            await ws.close(code=WS_INTERNAL_ERROR)
            return
        if not caller_may_watch(session_id, token):
            await ws.close(code=WS_POLICY_VIOLATION)
            return
        await ws.accept()
        closed = asyncio.create_task(ws.receive_text())        # completes when the client goes away
        last = START
        try:
            while not closed.done():
                for entry_id, event in await bus.read_up(session_id, after=last, block_ms=500):
                    last = entry_id
                    if event is not None:
                        await ws.send_text(encode(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            closed.cancel()
            if closed.done() and not closed.cancelled():
                closed.exception()          # mark the disconnect as handled; avoids an "unretrieved" warning

    return app


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_container(settings)
    if container.redis is None or settings.redis_url is None:
        raise RuntimeError("REDIS_URL is not set: the voice API needs Redis for the event bus")
    bus = RedisStreamBus(aioredis.from_url(settings.redis_url.get_secret_value(), decode_responses=True))
    redis_client = container.redis
    app = create_app(container, bus=bus, announce=lambda call_id: announce_call(redis_client, call_id))
    uvicorn.run(app, host="0.0.0.0", port=8000)   # noqa: S104
