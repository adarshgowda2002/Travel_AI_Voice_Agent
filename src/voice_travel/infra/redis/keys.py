"""The ONLY place Redis key names are defined. Everything lives under PREFIX so tooling can scan/clear safely.

  session(sid)            HASH    live session state (status, turn_no, is_speaking, interrupt flag...)
  turn(sid)               HASH    turn state machine (listening/transcribing/thinking/speaking)
  history(sid)            LIST    last N exchanges (JSON), newest at the tail
  session_tasks(sid)      SET     ids of in-flight agent tasks
  task(task_id)           HASH    task detail
  user_session(uid)       STRING  active session id for a caller
  ACTIVE_SESSIONS         SET     all active session ids
  cache(tool, digest)     STRING  JSON tool-result cache (short TTL)
  ASYNC_QUEUE             LIST    pending background jobs
  NEW_CALLS               STREAM  call ids announced by the API for a voice worker to join
  events_up(call_id)      STREAM  voice layer -> agent events (utterances, playback, ticks)
  events_down(call_id)    STREAM  agent -> voice layer commands (speak, cancel, hangup)
"""
PREFIX = "vta"
ACTIVE_SESSIONS = f"{PREFIX}:sessions:active"
ASYNC_QUEUE = f"{PREFIX}:queue:async_tasks"
NEW_CALLS = f"{PREFIX}:calls:new"


def session(sid: str) -> str:
    return f"{PREFIX}:session:{sid}"


def turn(sid: str) -> str:
    return f"{PREFIX}:turn:{sid}"


def history(sid: str) -> str:
    return f"{PREFIX}:session:{sid}:history"


def session_tasks(sid: str) -> str:
    return f"{PREFIX}:session:{sid}:tasks"


def task(task_id: str) -> str:
    return f"{PREFIX}:task:{task_id}"


def user_session(uid: str) -> str:
    return f"{PREFIX}:user:{uid}:session"


def cache(tool: str, digest: str) -> str:
    return f"{PREFIX}:cache:{tool}:{digest}"


def events_up(call_id: str) -> str:
    return f"{PREFIX}:call:{call_id}:events:up"


def events_down(call_id: str) -> str:
    return f"{PREFIX}:call:{call_id}:events:down"


def pattern() -> str:
    return f"{PREFIX}:*"
