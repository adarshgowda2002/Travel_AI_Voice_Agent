"""Event bus between the voice layer and the agent side: one Redis Stream per call and direction.

Streams (not pub/sub) so a restarting consumer can resume from its last id and nothing is lost mid-call.
The client MUST be a redis.asyncio client created with decode_responses=True."""
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar, cast

import redis
import redis.asyncio as aioredis
from pydantic import ValidationError

from voice_travel.infra.redis import keys
from voice_travel.voice.events import DownCommand, UpEvent, decode_down, decode_up, encode

logger = logging.getLogger(__name__)
T = TypeVar("T")
FIELD = "data"
START = "0-0"          # read from the beginning of the stream


def announce_call(client: redis.Redis, call_id: str) -> None:
    """Tell voice workers a call needs one. Sync (the API's data layer is sync); workers read it via
    RedisStreamBus.read_new_calls. Trimmed so announcements nobody consumed can't pile up."""
    client.xadd(keys.NEW_CALLS, {FIELD: call_id}, maxlen=1000, approximate=True)


class EventBus(Protocol):
    async def publish_up(self, event: UpEvent) -> str: ...
    async def publish_down(self, command: DownCommand) -> str: ...
    async def read_up(self, call_id: str, after: str = START, count: int = 100,
                      block_ms: int | None = None) -> list[tuple[str, UpEvent | None]]: ...
    async def read_down(self, call_id: str, after: str = START, count: int = 100,
                        block_ms: int | None = None) -> list[tuple[str, DownCommand | None]]: ...


class RedisStreamBus:
    def __init__(self, client: aioredis.Redis, *, maxlen: int = 1000) -> None:
        self._r = client
        self._maxlen = maxlen      # approximate trim: a call's backlog never grows without bound

    async def _add(self, key: str, payload: str) -> str:
        return str(await cast(Awaitable[object], self._r.xadd(
            key, {FIELD: payload}, maxlen=self._maxlen, approximate=True)))

    async def _read(self, key: str, after: str, count: int, block_ms: int | None) -> list[tuple[str, str]]:
        """Returns (entry_id, raw_json). block_ms=None returns immediately; otherwise waits up to block_ms."""
        reply = await cast(Awaitable[list], self._r.xread({key: after}, count=count, block=block_ms))
        return [(entry_id, fields[FIELD]) for _stream, entries in reply or [] for entry_id, fields in entries]

    async def read_new_calls(self, after: str, block_ms: int | None = None) -> list[tuple[str, str]]:
        """(entry_id, call_id) for calls announced after `after` ("$" = only calls from now on)."""
        return await self._read(keys.NEW_CALLS, after, 100, block_ms)

    async def publish_up(self, event: UpEvent) -> str:
        return await self._add(keys.events_up(event.call_id), encode(event))

    async def publish_down(self, command: DownCommand) -> str:
        return await self._add(keys.events_down(command.call_id), encode(command))

    async def read_up(self, call_id: str, after: str = START, count: int = 100,
                      block_ms: int | None = None) -> list[tuple[str, UpEvent | None]]:
        """An entry that fails validation comes back as (id, None): consumers skip it but still advance past
        it, so one bad message can never wedge a call."""
        return _decode_all(await self._read(keys.events_up(call_id), after, count, block_ms), decode_up)

    async def read_down(self, call_id: str, after: str = START, count: int = 100,
                        block_ms: int | None = None) -> list[tuple[str, DownCommand | None]]:
        return _decode_all(await self._read(keys.events_down(call_id), after, count, block_ms), decode_down)


def _decode_all(entries: list[tuple[str, str]], decode: Callable[[str], T]) -> list[tuple[str, T | None]]:
    out: list[tuple[str, T | None]] = []
    for entry_id, raw in entries:
        try:
            out.append((entry_id, decode(raw)))
        except ValidationError:
            logger.warning("skipping undecodable bus entry %s", entry_id)
            out.append((entry_id, None))
    return out
