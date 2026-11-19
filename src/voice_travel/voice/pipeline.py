"""One call, end to end.

  ingress:  transport audio -> VAD segments -> STT -> utterance events          (always listening)
  egress:   agent `speak` commands -> TTS -> playback, cut instantly by barge-in
  control:  `cancel_speech`, `tick_control`, `hangup` from the agent

Authority flows one way: the agent decides what is said and when; this layer only delivers audio and
reports what it heard and played. It never speaks on its own, never ends a conversational turn, and
reports failures as silent `transport.error` events.

Listening never pauses: audio keeps flowing into the segmenter while earlier utterances are transcribed
(STT has its own task) and while the agent is speaking (playback has its own task)."""
import asyncio
import logging
from dataclasses import dataclass

from voice_travel.utils import new_id
from voice_travel.voice import events as ev
from voice_travel.voice.bus import START, EventBus
from voice_travel.voice.playback import PlaybackController
from voice_travel.voice.segmenter import (
    SegmenterConfig,
    SegmenterEvent,
    SpeechDiscarded,
    SpeechEnded,
    SpeechStarted,
    UtteranceSegmenter,
)
from voice_travel.voice.stt import STT
from voice_travel.voice.transport import Transport
from voice_travel.voice.tts import TTS
from voice_travel.voice.vad import SAMPLE_RATE, VAD

logger = logging.getLogger(__name__)

MIN_PARTIAL_BYTES = SAMPLE_RATE          # 0.5 s of int16 audio: shorter snippets just produce noise
COMMAND_POLL_MS = 500
COMMAND_RETRY_S = 1.0


@dataclass(frozen=True)
class _Utterance:
    utterance_id: str
    ended: SpeechEnded
    over_playback: bool                   # the caller started speaking while the agent was talking


class CallPipeline:
    def __init__(self, call_id: str, transport: Transport, vad: VAD, stt: STT, tts: TTS, bus: EventBus, *,
                 segmenter_config: SegmenterConfig | None = None, partials: bool = False,
                 partial_interval_ms: int = 1000) -> None:
        self._call_id = call_id
        self._transport = transport
        self._stt = stt
        self._bus = bus
        self._segmenter = UtteranceSegmenter(vad, segmenter_config)
        self._playback = PlaybackController(call_id, transport, tts, bus)
        self._partials = partials
        self._partial_interval_s = partial_interval_ms / 1000
        self._queue: asyncio.Queue[_Utterance | None] = asyncio.Queue()
        self._utterance_id: str | None = None        # set while the caller is mid-utterance
        self._over_playback = False
        self._ticker: asyncio.Task[None] | None = None

    async def run(self) -> None:
        await self._bus.publish_up(ev.CallStarted(call_id=self._call_id))
        transcriber = asyncio.create_task(self._transcribe_loop())
        side_tasks: list[asyncio.Task[None]] = []
        self._playback.start()
        try:
            await self._transport.connect(self._call_id)
            side_tasks.append(asyncio.create_task(self._command_loop()))
            if self._partials:
                side_tasks.append(asyncio.create_task(self._partial_loop()))
            async for frame in self._transport.incoming_audio():
                await self._handle(self._segmenter.feed(frame.pcm))
            await self._handle(self._segmenter.flush())
        except Exception as exc:
            logger.exception("call %s transport failed", self._call_id)
            await self._report("transport", exc)
        finally:
            await self._stop(*side_tasks, *([self._ticker] if self._ticker else []))
            self._queue.put_nowait(None)
            await transcriber
            await self._playback.aclose()
            await self._transport.close()
            await self._bus.publish_up(ev.CallEnded(call_id=self._call_id))

    @staticmethod
    async def _stop(*tasks: asyncio.Task[None]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ---- ingress ------------------------------------------------------------
    async def _handle(self, events: list[SegmenterEvent]) -> None:
        for e in events:
            if isinstance(e, SpeechStarted):
                self._utterance_id = new_id("T")
                self._over_playback = self._playback.is_active
                if self._over_playback:
                    await self._playback.barge_in()      # always on: the caller can interrupt at any moment
            elif isinstance(e, SpeechDiscarded):
                self._utterance_id = None
            elif isinstance(e, SpeechEnded):
                self._queue.put_nowait(_Utterance(self._utterance_id or new_id("T"), e, self._over_playback))
                self._utterance_id = None

    async def _transcribe_loop(self) -> None:
        while (item := await self._queue.get()) is not None:
            try:
                transcript = await self._stt.transcribe(item.ended.pcm)
            except Exception as exc:
                logger.exception("call %s stt failed", self._call_id)
                await self._report("stt", exc)
                continue
            if not transcript.text:
                continue            # nothing intelligible: say nothing, send nothing
            await self._bus.publish_up(ev.UtteranceFinal(
                call_id=self._call_id, utterance_id=item.utterance_id, text=transcript.text,
                confidence=transcript.confidence, during_playback=item.over_playback))

    async def _partial_loop(self) -> None:
        """Best-effort live text while the caller is still talking. Skipped whenever STT is busy, so a
        partial can never delay a final, and never published after its utterance has finished."""
        while True:
            await asyncio.sleep(self._partial_interval_s)
            utterance_id = self._utterance_id
            if utterance_id is None or self._stt.is_busy():
                continue
            pcm = self._segmenter.snapshot()
            if len(pcm) < MIN_PARTIAL_BYTES:
                continue
            try:
                transcript = await self._stt.transcribe(pcm)
            except Exception:
                logger.exception("call %s partial transcription failed", self._call_id)
                continue
            if transcript.text and self._utterance_id == utterance_id:
                await self._bus.publish_up(ev.UtterancePartial(
                    call_id=self._call_id, utterance_id=utterance_id, text=transcript.text))

    # ---- control from the agent ----------------------------------------------
    async def _command_loop(self) -> None:
        last = START
        while True:
            try:
                for entry_id, command in await self._bus.read_down(self._call_id, after=last,
                                                                   block_ms=COMMAND_POLL_MS):
                    last = entry_id
                    if command is not None:
                        await self._dispatch(command)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("call %s command loop error", self._call_id)
                await asyncio.sleep(COMMAND_RETRY_S)

    async def _dispatch(self, command: ev.DownCommand) -> None:
        if isinstance(command, ev.Speak):
            self._playback.enqueue(command)
        elif isinstance(command, ev.CancelSpeech):
            await self._playback.cancel(command.speech_id)
        elif isinstance(command, ev.TickControl):
            await self._set_ticks(command)
        elif isinstance(command, ev.Hangup):
            await self._transport.close()            # ends incoming_audio, which ends the call

    async def _set_ticks(self, control: ev.TickControl) -> None:
        if self._ticker is not None:
            await self._stop(self._ticker)
            self._ticker = None
        if control.enabled:
            self._ticker = asyncio.create_task(self._tick_loop(control.interval_ms))

    async def _tick_loop(self, interval_ms: int) -> None:
        """Elapsed time since the agent asked for ticks. The agent decides whether any of it warrants speech."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        while True:
            await asyncio.sleep(interval_ms / 1000)
            await self._bus.publish_up(ev.Tick(call_id=self._call_id,
                                               elapsed_ms=int((loop.time() - started) * 1000)))

    async def _report(self, component: str, exc: Exception) -> None:
        try:
            await self._bus.publish_up(ev.TransportError(call_id=self._call_id, component=component,
                                                         detail=type(exc).__name__))
        except Exception:
            logger.exception("call %s could not report %s error", self._call_id, component)
