"""LiveKit adapter for the Transport protocol: joins the call's room as the voice worker and exposes the
caller's microphone as a stream of 16 kHz mono PCM frames. Moves audio only; never interprets it."""
import asyncio
import logging
from collections.abc import AsyncIterator

from livekit import rtc

from voice_travel.config import Settings
from voice_travel.voice.tokens import create_join_token, room_name
from voice_travel.voice.transport import SAMPLE_RATE, AudioFrame

logger = logging.getLogger(__name__)

WORKER_PREFIX = "voice-"          # identity prefix of voice workers; every other participant is the caller
OUTPUT_RATE = 48_000              # the published agent track; TTS audio at any other rate is resampled to this
OUTPUT_FRAME_SAMPLES = OUTPUT_RATE // 50          # 20 ms
OUTPUT_QUEUE_MS = 200             # audio buffered ahead of the listener: bounds how much a barge-in must flush


class LiveKitTransport:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._room: rtc.Room | None = None
        self._frames: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._pumps: set[asyncio.Task[None]] = set()
        self._heard_audio = False
        self._source: rtc.AudioSource | None = None
        self._captured_samples = 0                 # audio handed to the output track during the current play()

    async def connect(self, call_id: str) -> None:
        token = create_join_token(self._settings, call_id=call_id, identity=f"{WORKER_PREFIX}{call_id}")
        room = rtc.Room()
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("participant_disconnected", self._on_participant_disconnected)
        room.on("disconnected", lambda *_: self._frames.put_nowait(None))
        await room.connect(self._settings.livekit_url, token)
        self._room = room
        self._source = rtc.AudioSource(OUTPUT_RATE, 1, queue_size_ms=OUTPUT_QUEUE_MS)
        await room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("agent-voice", self._source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        logger.info("joined room %s", room_name(call_id))

    def _on_track_subscribed(self, track: rtc.Track, _pub: rtc.RemoteTrackPublication,
                             participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or participant.identity.startswith(WORKER_PREFIX):
            return
        task = asyncio.create_task(self._pump(track))
        self._pumps.add(task)
        task.add_done_callback(self._pumps.discard)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if not participant.identity.startswith(WORKER_PREFIX):
            self._frames.put_nowait(None)       # the caller left: the call is over

    async def _pump(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            async for event in stream:
                self._frames.put_nowait(AudioFrame(bytes(event.frame.data)))
        finally:
            await stream.aclose()

    async def incoming_audio(self) -> AsyncIterator[AudioFrame]:
        """Caller audio until they leave. Gives up only if no audio ever arrives within the join timeout
        (a transport-level safeguard, not a conversational decision)."""
        while True:
            try:
                timeout = None if self._heard_audio else self._settings.voice_join_timeout_sec
                frame = await asyncio.wait_for(self._frames.get(), timeout)
            except TimeoutError:
                logger.warning("caller never published audio; giving up")
                return
            if frame is None:
                return
            self._heard_audio = True
            yield frame

    async def play(self, frames: AsyncIterator[AudioFrame]) -> int:
        source = self._require_source()
        self._captured_samples = 0
        resamplers: dict[int, rtc.AudioResampler] = {}
        async for frame in frames:
            if frame.sample_rate == OUTPUT_RATE:
                await self._capture(frame.pcm)
                continue
            resampler = resamplers.setdefault(
                frame.sample_rate, rtc.AudioResampler(frame.sample_rate, OUTPUT_RATE, num_channels=1))
            for out in resampler.push(bytearray(frame.pcm)):
                await self._capture(bytes(out.data))
        for resampler in resamplers.values():
            for out in resampler.flush():
                await self._capture(bytes(out.data))
        await source.wait_for_playout()
        return self._captured_ms()

    async def _capture(self, pcm: bytes) -> None:
        """Hand audio to the output track in 20 ms frames; this awaits when the small buffer is full, which
        paces synthesis to real time so an interruption only ever has to flush ~OUTPUT_QUEUE_MS."""
        source = self._require_source()
        step = OUTPUT_FRAME_SAMPLES * 2
        for offset in range(0, len(pcm), step):
            chunk = pcm[offset:offset + step]
            samples = len(chunk) // 2
            await source.capture_frame(rtc.AudioFrame(chunk, OUTPUT_RATE, 1, samples))
            self._captured_samples += samples

    def _captured_ms(self) -> int:
        return self._captured_samples * 1000 // OUTPUT_RATE

    async def stop_playback(self) -> int:
        """Silence the agent now. Returns what the caller actually heard: audio handed over minus what was
        still waiting in the buffer."""
        if self._source is None:
            return 0
        still_queued_ms = int(self._source.queued_duration * 1000)
        heard_ms = max(0, self._captured_ms() - still_queued_ms)
        self._source.clear_queue()
        return heard_ms

    def _require_source(self) -> rtc.AudioSource:
        if self._source is None:
            raise RuntimeError("transport is not connected")
        return self._source

    async def close(self) -> None:
        self._frames.put_nowait(None)               # ends incoming_audio()
        for task in list(self._pumps):
            task.cancel()
        if self._source is not None:
            await self._source.aclose()
            self._source = None
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
