"""Shared fakes for voice-layer tests: scripted audio in, simulated (time-compressed) playback out."""
import asyncio

import fakeredis
import numpy as np

from voice_travel.voice import events as ev
from voice_travel.voice.bus import RedisStreamBus
from voice_travel.voice.pipeline import CallPipeline
from voice_travel.voice.stt import Transcript
from voice_travel.voice.transport import AudioFrame

CALL = "S1"
SPEEDUP = 10          # simulated playback runs 10x faster than real time so tests stay quick


class AmplitudeVAD:
    """Loud window = speech. Lets segmentation run in tests without the Silero model."""
    def reset(self):
        pass

    def probability(self, window):
        return 0.9 if float(np.abs(window).max()) > 0.1 else 0.0


def loud(ms):
    return np.full(ms * 16, 10_000, dtype=np.int16).tobytes()


def quiet(ms):
    return bytes(ms * 16 * 2)


class ScriptTransport:
    """Feeds scripted (delay_s, pcm) chunks, then stays 'connected' until closed, like a real call."""
    def __init__(self, script=(), fail_on_connect=False):
        self.script, self.fail_on_connect = list(script), fail_on_connect
        self.closed = asyncio.Event()
        self.play_calls = 0
        self.stop_calls = 0
        self.playing = False
        self.heard_ms = 0

    async def connect(self, call_id):
        if self.fail_on_connect:
            raise ConnectionError("no route")

    async def incoming_audio(self):
        for delay, pcm in self.script:
            if delay:
                await asyncio.sleep(delay)
            yield AudioFrame(pcm)
        await self.closed.wait()

    async def play(self, frames):
        self.play_calls += 1
        self.playing, self.heard_ms = True, 0
        try:
            async for f in frames:
                ms = len(f.pcm) * 1000 // (2 * f.sample_rate)
                await asyncio.sleep(ms / 1000 / SPEEDUP)
                self.heard_ms += ms
            return self.heard_ms
        finally:
            self.playing = False

    async def stop_playback(self):
        self.stop_calls += 1
        return self.heard_ms

    async def close(self):
        self.closed.set()


class FakeTTS:
    def __init__(self, frames=20, frame_ms=100, fail_on=()):
        self.frames, self.frame_ms, self.fail_on = frames, frame_ms, set(fail_on)

    async def synthesize(self, text):
        if text in self.fail_on:
            raise RuntimeError("tts down")
        for _ in range(self.frames):
            yield AudioFrame(bytes(self.frame_ms * 16 * 2), 16000)
            await asyncio.sleep(0)


class ScriptedSTT:
    def __init__(self, results=()):
        self.results = list(results)

    def is_busy(self):
        return False

    async def transcribe(self, pcm):
        r = self.results.pop(0) if self.results else Transcript("text")
        if isinstance(r, Exception):
            raise r
        return r


def make_bus():
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    return RedisStreamBus(client), client


async def events_of(bus):
    return [e for _i, e in await bus.read_up(CALL) if e is not None]


async def wait_for(bus, predicate, timeout=5.0):
    """Poll the bus until some event satisfies predicate; fail loudly with what WAS seen."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        seen = await events_of(bus)
        if any(predicate(e) for e in seen):
            return seen
        if loop.time() > deadline:
            raise AssertionError(f"timed out; saw {[e.type for e in seen]}")
        await asyncio.sleep(0.02)


def speak(text="Hello", speech_id="p1"):
    return ev.Speak(call_id=CALL, speech_id=speech_id, text=text)


def run_call(transport, *, stt=None, tts=None, driver=None, partials=False, partial_interval_ms=1000):
    """Run one CallPipeline; `driver(bus, client)` plays the agent. Hangs up at the end and returns all events."""
    async def go():
        bus, client = make_bus()
        pipeline = CallPipeline(CALL, transport, AmplitudeVAD(), stt or ScriptedSTT(), tts or FakeTTS(), bus,
                                partials=partials, partial_interval_ms=partial_interval_ms)
        run = asyncio.create_task(pipeline.run())
        try:
            if driver:
                await driver(bus, client)
        finally:
            await bus.publish_down(ev.Hangup(call_id=CALL))
            await asyncio.wait_for(run, 10)
        return await events_of(bus)
    return asyncio.run(go())


def types(events):
    return [e.type for e in events]
