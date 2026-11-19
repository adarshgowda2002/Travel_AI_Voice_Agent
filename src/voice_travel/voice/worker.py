"""Voice worker: waits for announced calls and runs one CallPipeline per call. `voice-worker`.

Shares one STT model and one TTS voice across calls. Calls announced while no worker is running
are not replayed (a call is only useful while its caller is connected)."""
import asyncio
import logging

import redis.asyncio as aioredis

from voice_travel.config import Settings, get_settings
from voice_travel.logging import configure_logging
from voice_travel.voice.bus import RedisStreamBus
from voice_travel.voice.livekit_transport import LiveKitTransport
from voice_travel.voice.pipeline import CallPipeline
from voice_travel.voice.stt import STT, FasterWhisperSTT
from voice_travel.voice.tts import TTS, PiperTTS
from voice_travel.voice.vad import SileroVAD

logger = logging.getLogger(__name__)


async def run_call(call_id: str, settings: Settings, stt: STT, tts: TTS, bus: RedisStreamBus) -> None:
    pipeline = CallPipeline(call_id, LiveKitTransport(settings), SileroVAD(settings.vad_model_path), stt, tts, bus,
                            partials=settings.stt_partials, partial_interval_ms=settings.stt_partial_interval_ms)
    try:
        await pipeline.run()
    except Exception:
        logger.exception("call %s ended abnormally", call_id)


async def serve(settings: Settings) -> None:
    if settings.redis_url is None:
        raise RuntimeError("REDIS_URL is not set")
    client = aioredis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
    bus = RedisStreamBus(client)
    stt = FasterWhisperSTT.from_settings(settings)
    await stt.warm_up()
    tts = await asyncio.to_thread(PiperTTS.from_settings, settings)
    logger.info("voice worker ready")
    tasks: set[asyncio.Task[None]] = set()
    last_id = "$"
    while True:
        for entry_id, call_id in await bus.read_new_calls(last_id, block_ms=5000):
            last_id = entry_id
            task = asyncio.create_task(run_call(call_id, settings, stt, tts, bus))
            tasks.add(task)
            task.add_done_callback(tasks.discard)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(serve(settings))
