"""Seed Redis with mock LIVE state derived from the generated dataset, so it matches Postgres.

    seed-redis            # seed (idempotent: overwrites its own keys)
    seed-redis --clear    # delete every vta:* key, then exit

Only keys under the vta: prefix are ever written or deleted. Sends are pipelined (one round trip).
Mock data expires after 24h; production idle expiry would be ~30 min.
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import redis as redis_lib

from voice_travel.config import get_settings
from voice_travel.infra.redis import create_redis
from voice_travel.infra.redis import keys as K
from voice_travel.logging import configure_logging
from voice_travel.seeding.csv_io import read_csv

SESSION_TTL = 24 * 3600
TASK_TTL = 3600
TURN_STATES = ("listening", "transcribing", "thinking", "speaking")
OUTCOMES_SEEDED = ("booked", "changed_mind", "tool_error", "cancelled")      # one session per intent x outcome
CACHE_LIMIT = {"flights.search": 12, "hotels.search": 12, "weather.forecast": 10}
CACHE_TTL = {"flights.search": 15 * 60, "hotels.search": 30 * 60, "weather.forecast": 3600}
HISTORY_LEN = 10
QUEUE_JOBS = 10


def clear(r: redis_lib.Redis) -> int:
    deleted, batch = 0, []
    for key in r.scan_iter(match=K.pattern(), count=500):
        batch.append(key)
        if len(batch) == 500:
            deleted += r.unlink(*batch)
            batch = []
    return deleted + (r.unlink(*batch) if batch else 0)


def _pick_scenarios(scenarios: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(scenarios, key=lambda s: s["session_id"])
    return [next(s for s in ordered if s["intent"] == intent and s["outcome"] == outcome)
            for intent in sorted({s["intent"] for s in ordered}) for outcome in OUTCOMES_SEEDED]


def _seed_sessions(pipe: redis_lib.client.Pipeline, data_dir: Path, now: dt.datetime) -> int:
    scenarios = read_csv(data_dir, "scenarios.csv")
    users = {u["id"]: u for u in read_csv(data_dir, "users.csv")}
    session_user = {s["id"]: s["user_id"] for s in read_csv(data_dir, "sessions.csv")}
    convs: dict[str, list[dict[str, str]]] = {}
    for c in read_csv(data_dir, "conversations.csv"):
        convs.setdefault(c["session_id"], []).append(c)
    tasks: dict[str, list[dict[str, str]]] = {}
    for t in read_csv(data_dir, "agent_tasks.csv"):
        tasks.setdefault(t["session_id"], []).append(t)

    picked = _pick_scenarios(scenarios)
    for i, sc in enumerate(picked):
        sid, uid = sc["session_id"], session_user[sc["session_id"]]
        turns = sorted(convs[sid], key=lambda c: int(c["turn_no"]))
        so_far = turns[:max(1, len(turns) - (i % 3))]          # mid-conversation at different points
        cur = so_far[-1]
        state = TURN_STATES[i % len(TURN_STATES)]

        pipe.hset(K.session(sid), mapping={
            "user_id": uid, "user_name": users[uid]["name"], "status": "active", "channel": "voice",
            "intent": cur["intent"], "turn_no": cur["turn_no"], "last_agent": cur["agent"],
            "accent": sc["accent"], "noise_profile": sc["noise_profile"], "destination": sc["destination"],
            "started_at": (now - dt.timedelta(minutes=2 + i)).isoformat(timespec="seconds"),
            "is_speaking": int(state == "speaking"), "interrupt": 0})
        pipe.expire(K.session(sid), SESSION_TTL)

        text = cur["user_text"]
        pipe.hset(K.turn(sid), mapping={
            "turn_no": cur["turn_no"], "state": state,
            "partial_transcript": text[:max(8, len(text) // 2)] if state == "transcribing" else "",
            "final_transcript": text if state in ("thinking", "speaking") else "",
            "updated_at": now.isoformat(timespec="seconds")})
        pipe.expire(K.turn(sid), SESSION_TTL)

        pipe.delete(K.history(sid))
        for c in so_far[-HISTORY_LEN:]:
            pipe.rpush(K.history(sid), json.dumps({"turn_no": int(c["turn_no"]), "user": c["user_text"],
                                                   "agent": c["agent_text"], "intent": c["intent"]},
                                                  ensure_ascii=False))
        pipe.expire(K.history(sid), SESSION_TTL)

        pipe.delete(K.session_tasks(sid))
        live = tasks.get(sid, [])[-2:] if state in ("thinking", "speaking") else []
        for t in live:
            pipe.hset(K.task(t["id"]), mapping={
                "session_id": sid, "agent": t["agent"], "tool": t["tool"],
                "status": "running" if state == "thinking" else t["status"], "request": t["request"]})
            pipe.expire(K.task(t["id"]), TASK_TTL)
            pipe.sadd(K.session_tasks(sid), t["id"])
        if live:
            pipe.expire(K.session_tasks(sid), TASK_TTL)

        pipe.set(K.user_session(uid), sid, ex=SESSION_TTL)
        pipe.sadd(K.ACTIVE_SESSIONS, sid)
    pipe.expire(K.ACTIVE_SESSIONS, SESSION_TTL)
    return len(picked)


def _seed_cache_and_queue(pipe: redis_lib.client.Pipeline, data_dir: Path) -> tuple[int, int]:
    all_tasks = read_csv(data_dir, "agent_tasks.csv")
    cached = {tool: 0 for tool in CACHE_LIMIT}
    for t in all_tasks:
        tool = t["tool"]
        if tool in CACHE_LIMIT and t["status"] == "success" and cached[tool] < CACHE_LIMIT[tool]:
            digest = hashlib.sha1(t["request"].encode()).hexdigest()[:12]
            pipe.set(K.cache(tool, digest),
                     json.dumps({"request": json.loads(t["request"]), "result": json.loads(t["result"])}),
                     ex=CACHE_TTL[tool])
            cached[tool] += 1

    pipe.delete(K.ASYNC_QUEUE)
    jobs = [t for t in all_tasks if t["tool"] == "notify.email"][:QUEUE_JOBS]
    for t in jobs:
        pipe.rpush(K.ASYNC_QUEUE, json.dumps({
            "job": "send_email", "task_id": t["id"], "session_id": t["session_id"],
            "template": json.loads(t["request"]).get("template"), "attempts": 0}))
    pipe.expire(K.ASYNC_QUEUE, SESSION_TTL)
    return sum(cached.values()), len(jobs)


def seed(r: redis_lib.Redis, data_dir: Path) -> dict[str, int]:
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    pipe = r.pipeline(transaction=False)
    sessions = _seed_sessions(pipe, data_dir, now)
    cache_entries, jobs = _seed_cache_and_queue(pipe, data_dir)
    commands = len(pipe.execute())
    return {"commands": commands, "sessions": sessions, "cache_requests": cache_entries, "queued_jobs": jobs,
            "keys_now": sum(1 for _ in r.scan_iter(match=K.pattern(), count=1000))}


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.redis_url is None:
        sys.exit("REDIS_URL is not set (environment or .env)")
    r = create_redis(settings.redis_url)
    if "--clear" in sys.argv:
        print(f"deleted {clear(r)} keys under {K.pattern()}")
        return
    print(seed(r, settings.dataset_dir))


if __name__ == "__main__":
    main()
