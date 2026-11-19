"""DEV STAND-IN for the real agent, so the voice loop can be heard end to end. `voice-echo-agent`.

It lives on the AGENT side of the event contract: the voice layer itself never produces speech. It listens
for `utterance.final` and replies with a `speak` command, exactly as the real agent will. Replace it with
the real conversation agent; the voice layer needs no change."""
import asyncio
import logging

import redis.asyncio as aioredis

from voice_travel.config import Settings, get_settings
from voice_travel.logging import configure_logging
from voice_travel.utils import new_id
from voice_travel.voice import events as ev
from voice_travel.voice.bus import START, EventBus, RedisStreamBus

logger = logging.getLogger(__name__)


async def echo_call(bus: EventBus, call_id: str) -> None:
    last = START
    while True:
        for entry_id, event in await bus.read_up(call_id, after=last, block_ms=1000):
            last = entry_id
            if isinstance(event, ev.CallEnded):
                return
            if isinstance(event, ev.UtteranceFinal):
                await bus.publish_down(ev.Speak(call_id=call_id, speech_id=new_id("P"),
                                                text=f"You said: {event.text}"))


async def serve(settings: Settings) -> None:
    if settings.redis_url is None:
        raise RuntimeError("REDIS_URL is not set")
    bus = RedisStreamBus(aioredis.from_url(settings.redis_url.get_secret_value(), decode_responses=True))
    logger.info("echo agent ready (dev stand-in)")
    tasks: set[asyncio.Task[None]] = set()
    last_id = "$"
    while True:
        for entry_id, call_id in await bus.read_new_calls(last_id, block_ms=5000):
            last_id = entry_id
            task = asyncio.create_task(echo_call(bus, call_id))
            tasks.add(task)
            task.add_done_callback(tasks.discard)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(serve(settings))
