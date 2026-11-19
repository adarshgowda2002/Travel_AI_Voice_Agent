"""Transport interface: the only thing that knows how audio physically reaches the caller.

The voice engine (VAD, STT, TTS, playback, barge-in) is written against this protocol. LiveKit is the first
adapter; an Asterisk adapter can be added later without touching the engine. A transport moves audio and
reports call lifecycle. It never interprets speech and never invents content."""
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

SAMPLE_RATE = 16_000        # mono PCM16 into VAD/STT


@dataclass(frozen=True)
class AudioFrame:
    pcm: bytes               # little-endian int16, mono
    sample_rate: int = SAMPLE_RATE


class Transport(Protocol):
    async def connect(self, call_id: str) -> None:
        """Join the call. Listening starts here and never pauses until close()."""

    def incoming_audio(self) -> AsyncIterator[AudioFrame]:
        """Caller audio, continuous, including while the agent is speaking (the barge-in guarantee)."""

    async def play(self, frames: AsyncIterator[AudioFrame]) -> int:
        """Play agent audio (frames may use any sample rate). Returns the milliseconds that were played
        once everything has been heard. The caller cancels this task to interrupt it."""

    async def stop_playback(self) -> int:
        """Silence outgoing audio immediately and drop what was buffered (barge-in / cancel_speech).
        Returns the milliseconds of this playback that actually reached the caller."""

    async def close(self) -> None: ...
