import asyncio

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient
from livekit import api
from starlette.websockets import WebSocketDisconnect

from voice_travel.config import Settings
from voice_travel.container import Container
from voice_travel.services import BookingService, IdentityService, OpsLogService, SessionService, TripService
from voice_travel.voice import events as ev
from voice_travel.voice.api import create_app
from voice_travel.voice.bus import RedisStreamBus
from voice_travel.voice.tokens import create_join_token


def make_client(db):
    c = Container(settings=Settings(_env_file=None), db=db, redis=None, identity=IdentityService(db),
                  trips=TripService(db), bookings=BookingService(db), sessions=SessionService(db),
                  ops=OpsLogService(db))
    return TestClient(create_app(c)), c


def test_create_session_returns_room_url_and_valid_token(db):
    client, c = make_client(db)
    r = client.post("/api/v1/voice/session", json={"user_id": "U0001"})
    assert r.status_code == 200
    body = r.json()
    assert body["room"] == f"call_{body['session_id']}"
    assert body["url"] == c.settings.livekit_url
    claims = api.TokenVerifier(c.settings.livekit_api_key,
                               c.settings.livekit_api_secret.get_secret_value()).verify(body["token"])
    assert claims.identity == "U0001" and claims.video.room == body["room"]


def test_unknown_user_is_404_and_creates_nothing(db):
    client, _ = make_client(db)
    assert client.post("/api/v1/voice/session", json={"user_id": "NOPE"}).status_code == 404


def test_end_session(db):
    client, _ = make_client(db)
    sid = client.post("/api/v1/voice/session", json={"user_id": "U0001"}).json()["session_id"]
    r = client.post(f"/api/v1/voice/session/{sid}/end")
    assert r.status_code == 200 and r.json()["status"] == "ended"
    assert client.post("/api/v1/voice/session/UNKNOWN/end").status_code == 404


def test_healthz(db):
    client, _ = make_client(db)
    assert client.get("/healthz").json() == {"status": "ok"}


# ---- announce + events websocket -------------------------------------------------
def make_app_client(db, **kw):
    c = Container(settings=Settings(_env_file=None), db=db, redis=None, identity=IdentityService(db),
                  trips=TripService(db), bookings=BookingService(db), sessions=SessionService(db),
                  ops=OpsLogService(db))
    return TestClient(create_app(c, **kw)), c


def test_create_session_announces_the_call_to_workers(db):
    announced = []
    client, _ = make_app_client(db, announce=announced.append)
    sid = client.post("/api/v1/voice/session", json={"user_id": "U0001"}).json()["session_id"]
    assert announced == [sid]


def test_unreachable_workers_fail_the_request_and_end_the_session(db):
    def boom(_call_id):
        raise redis.ConnectionError("down")
    client, _ = make_app_client(db, announce=boom)
    assert client.post("/api/v1/voice/session", json={"user_id": "U0001"}).status_code == 503


def test_events_websocket_streams_utterances_to_the_token_holder(db):
    bus = RedisStreamBus(fakeredis.FakeAsyncRedis(decode_responses=True))
    client, c = make_app_client(db, bus=bus)
    body = client.post("/api/v1/voice/session", json={"user_id": "U0001"}).json()
    sid = body["session_id"]
    asyncio.run(bus.publish_up(ev.UtteranceFinal(call_id=sid, utterance_id="T1", text="plan a trip")))
    with client.websocket_connect(f"/api/v1/voice/events/{sid}?token={body['token']}") as ws:
        got = ev.decode_up(ws.receive_text())
    assert got.type == "utterance.final" and got.text == "plan a trip"


@pytest.mark.parametrize("token_for", ["other-call", "garbage", "missing"])
def test_events_websocket_rejects_anyone_without_this_calls_token(db, token_for):
    bus = RedisStreamBus(fakeredis.FakeAsyncRedis(decode_responses=True))
    client, c = make_app_client(db, bus=bus)
    sid = client.post("/api/v1/voice/session", json={"user_id": "U0001"}).json()["session_id"]
    token = {"other-call": create_join_token(c.settings, call_id="DIFFERENT", identity="U0001"),
             "garbage": "not-a-jwt", "missing": ""}[token_for]
    with (pytest.raises(WebSocketDisconnect) as exc,
          client.websocket_connect(f"/api/v1/voice/events/{sid}?token={token}")):
        pass
    assert exc.value.code == 1008


def test_demo_page_is_served(db):
    client, _ = make_app_client(db)
    r = client.get("/demo")
    assert r.status_code == 200 and "Voice layer demo" in r.text


# ---- API key ---------------------------------------------------------------------
def make_secured_client(db):
    c = Container(settings=Settings(_env_file=None, voice_api_key="s3cret-key"), db=db, redis=None,
                  identity=IdentityService(db), trips=TripService(db), bookings=BookingService(db),
                  sessions=SessionService(db), ops=OpsLogService(db))
    return TestClient(create_app(c))


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret-key"},
                                     {"Authorization": "Basic s3cret-key"}])
def test_session_endpoints_reject_missing_or_wrong_key(db, headers):
    client = make_secured_client(db)
    assert client.post("/api/v1/voice/session", json={"user_id": "U0001"}, headers=headers).status_code == 401
    assert client.post("/api/v1/voice/session/X/end", headers=headers).status_code == 401


def test_session_endpoints_accept_the_right_key(db):
    client = make_secured_client(db)
    h = {"Authorization": "Bearer s3cret-key"}
    sid = client.post("/api/v1/voice/session", json={"user_id": "U0001"}, headers=h).json()["session_id"]
    assert client.post(f"/api/v1/voice/session/{sid}/end", headers=h).status_code == 200
    assert client.get("/healthz").status_code == 200      # health stays open
