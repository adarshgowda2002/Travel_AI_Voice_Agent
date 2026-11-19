"""Text-to-speech: a rendering service. Turns the text it is given into audio, one sentence at a time so
playback can start before the whole reply is synthesized. It never chooses or alters wording."""
import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from voice_travel.config import Settings
from voice_travel.voice.transport import AudioFrame


class TTS(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        """Yield audio as it is produced. Closing the iterator early stops the synthesis work."""
        ...


class PiperTTS:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Piper voice not found at {model_path}; run `fetch-voice-models`")
        from piper import PiperVoice  # heavy import, deferred
        self._voice = PiperVoice.load(model_path)

    @classmethod
    def from_settings(cls, s: Settings) -> "PiperTTS":
        return cls(s.tts_voice_path)

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        out: asyncio.Queue[AudioFrame | BaseException | None] = asyncio.Queue()
        stop = threading.Event()               # set when the consumer goes away (barge-in): stop burning CPU

        def put(item: AudioFrame | BaseException | None) -> None:
            with contextlib.suppress(RuntimeError):        # loop already closed during shutdown
                loop.call_soon_threadsafe(out.put_nowait, item)

        def work() -> None:
            try:
                for chunk in self._voice.synthesize(text):
                    if stop.is_set():
                        return
                    put(AudioFrame(chunk.audio_int16_bytes, chunk.sample_rate))
            except Exception as exc:
                put(exc)
            finally:
                put(None)

        loop.run_in_executor(None, work)
        try:
            while (item := await out.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
