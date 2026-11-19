"""Plays the agent's `speak` commands, in order, and stops instantly on barge-in.

The agent decides WHAT is said and WHEN; this decides only how audio is delivered. Playback is best-effort:
it can be cut short at any moment, and every ending is reported (completed or not, with how many
milliseconds were actually heard) so the agent never assumes a message was fully delivered."""
import asyncio
import logging
from collections import deque

from voice_travel.voice import events as ev
from voice_travel.voice.bus import EventBus
from voice_travel.voice.transport import Transport
from voice_travel.voice.tts import TTS

logger = logging.getLogger(__name__)


class PlaybackController:
    def __init__(self, call_id: str, transport: Transport, tts: TTS, bus: EventBus) -> None:
        self._call_id = call_id
        self._transport = transport
        self._tts = tts
        self._bus = bus
        self._pending: deque[ev.Speak] = deque()
        self._wake = asyncio.Event()
        self._current: ev.Speak | None = None
        self._task: asyncio.Task[int] | None = None
        self._interrupted = False
        self._played_ms = 0
        self._runner: asyncio.Task[None] | None = None

    @property
    def is_active(self) -> bool:
        """Something is playing or waiting to play."""
        return self._current is not None or bool(self._pending)

    def start(self) -> None:
        self._runner = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def enqueue(self, speak: ev.Speak) -> None:
        self._pending.append(speak)
        self._wake.set()

    async def cancel(self, speech_id: str) -> None:
        """The agent withdrew one message: stop it if playing, drop it if still queued."""
        if self._current is not None and self._current.speech_id == speech_id:
            await self._interrupt_current()
            return
        for queued in list(self._pending):
            if queued.speech_id == speech_id:
                self._pending.remove(queued)
                await self._publish(ev.PlaybackEnded(call_id=self._call_id, speech_id=speech_id,
                                                     completed=False, played_ms=0))

    async def barge_in(self) -> None:
        """The caller started speaking: silence the agent now and drop everything queued. The agent hears
        about each cut message and decides what, if anything, to say next."""
        dropped = list(self._pending)
        self._pending.clear()
        await self._interrupt_current()
        for speak in dropped:
            await self._publish(ev.PlaybackEnded(call_id=self._call_id, speech_id=speak.speech_id,
                                                 completed=False, played_ms=0))

    async def _interrupt_current(self) -> None:
        if self._task is None:
            return
        self._interrupted = True
        self._played_ms = await self._transport.stop_playback()
        self._task.cancel()

    async def _run(self) -> None:
        while True:
            while not self._pending:
                self._wake.clear()
                await self._wake.wait()
            await self._play(self._pending.popleft())

    async def _play(self, speak: ev.Speak) -> None:
        self._current, self._interrupted, self._played_ms = speak, False, 0
        task = asyncio.create_task(self._transport.play(self._tts.synthesize(speak.text)))
        self._task = task
        await self._publish(ev.PlaybackStarted(call_id=self._call_id, speech_id=speak.speech_id))
        completed, played_ms = False, 0
        try:
            played_ms, completed = await task, True
        except asyncio.CancelledError:
            if not self._interrupted:
                raise                      # we are being shut down, not interrupted
            played_ms = self._played_ms
        except Exception as exc:           # TTS or transport failed: report silently, never speak about it
            logger.exception("call %s playback failed", self._call_id)
            await self._publish(ev.TransportError(call_id=self._call_id, component="playback",
                                                  detail=type(exc).__name__))
        finally:
            self._current, self._task = None, None
        await self._publish(ev.PlaybackEnded(call_id=self._call_id, speech_id=speak.speech_id,
                                             completed=completed, played_ms=played_ms))

    async def _publish(self, event: ev.PlaybackStarted | ev.PlaybackEnded | ev.TransportError) -> None:
        try:
            await self._bus.publish_up(event)
        except Exception:
            logger.exception("call %s could not publish %s", self._call_id, event.type)
