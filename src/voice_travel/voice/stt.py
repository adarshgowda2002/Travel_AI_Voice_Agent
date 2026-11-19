"""Speech-to-text: audio in, best-effort text out. No interpretation, cleanup or intent handling.

Phase 2 transcribes one finished utterance at a time (no partials yet)."""
import asyncio
import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from voice_travel.config import Settings
from voice_travel.voice.vad import SAMPLE_RATE

NO_SPEECH_DROP = 0.6        # Whisper segments it judges to be non-speech are dropped (kills hallucinations on noise)


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float | None = None


class STT(Protocol):
    def is_busy(self) -> bool:
        """True while a transcription is running (partials are skipped then, so finals are never delayed)."""
        ...

    async def transcribe(self, pcm: bytes) -> Transcript:
        """pcm: int16 mono at 16 kHz. Returns empty text when nothing intelligible was said."""
        ...


class FasterWhisperSTT:
    def __init__(self, model: str = "small.en", device: str = "auto", compute_type: str = "int8",
                 language: str = "en") -> None:
        self._args = (model, device, compute_type)
        self._language = language
        self._model: Any = None
        self._lock = asyncio.Lock()           # one transcription at a time; the model is not re-entrant

    @classmethod
    def from_settings(cls, s: Settings) -> "FasterWhisperSTT":
        return cls(s.stt_model, s.stt_device, s.stt_compute_type, s.stt_language)

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred until first use
            model, device, compute_type = self._args
            self._model = WhisperModel(model, device=device, compute_type=compute_type)
        return self._model

    def _transcribe_blocking(self, pcm: bytes) -> Transcript:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._load().transcribe(audio, language=self._language, beam_size=1,
                                                  condition_on_previous_text=False, vad_filter=False)
        kept = [s for s in segments if s.no_speech_prob <= NO_SPEECH_DROP]
        text = " ".join(s.text.strip() for s in kept).strip()
        if not text:
            return Transcript("")
        confidence = sum(math.exp(s.avg_logprob) for s in kept) / len(kept)
        return Transcript(text, round(min(1.0, confidence), 3))

    def is_busy(self) -> bool:
        return self._lock.locked()

    async def warm_up(self) -> None:
        """Load the model before the first call so the first utterance isn't slow."""
        async with self._lock:
            await asyncio.to_thread(self._load)

    async def transcribe(self, pcm: bytes) -> Transcript:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_blocking, pcm)


assert SAMPLE_RATE == 16_000     # Whisper input rate; the segmenter emits exactly this
