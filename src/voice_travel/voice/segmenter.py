"""Turns a continuous audio stream into utterances using VAD probabilities. Pure and synchronous.

This is audio segmentation only. It does NOT decide conversational turns: silence ending an utterance
just means "this chunk of speech is complete"; the agent decides whether the user is done.
It never stops consuming audio, so listening never pauses (the always-listening guarantee)."""
from collections import deque
from dataclasses import dataclass

import numpy as np

from voice_travel.voice.vad import SAMPLE_RATE, VAD, WINDOW_SAMPLES

WINDOW_MS = WINDOW_SAMPLES * 1000 // SAMPLE_RATE          # 32
WINDOW_BYTES = WINDOW_SAMPLES * 2                         # int16


@dataclass(frozen=True)
class SegmenterConfig:
    on_threshold: float = 0.5          # speech starts at/above this
    off_threshold: float = 0.35        # while speaking, below this counts as silence (hysteresis)
    start_ms: int = 96                 # sustained speech needed to open an utterance
    silence_ms: int = 500              # sustained silence that closes it
    min_speech_ms: int = 250           # shorter bursts (clicks, coughs) are discarded
    preroll_ms: int = 320              # audio kept from before the detected start (soft onsets)
    tail_ms: int = 160                 # trailing silence kept on the utterance
    max_utterance_ms: int = 20_000     # force a cut so one endless monologue can't grow unbounded


@dataclass(frozen=True)
class SpeechStarted:
    pass


@dataclass(frozen=True)
class SpeechDiscarded:
    """Opened but too short to be an utterance."""


@dataclass(frozen=True)
class SpeechEnded:
    pcm: bytes                         # int16 mono 16 kHz
    speech_ms: int
    forced: bool = False               # cut at max_utterance_ms rather than by silence


SegmenterEvent = SpeechStarted | SpeechDiscarded | SpeechEnded


class UtteranceSegmenter:
    def __init__(self, vad: VAD, config: SegmenterConfig | None = None) -> None:
        self._vad = vad
        self._cfg = config or SegmenterConfig()
        self._pending = b""
        self._preroll: deque[bytes] = deque(maxlen=max(1, self._cfg.preroll_ms // WINDOW_MS))
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._speaking = False
        self._on_run = 0
        self._buffer: list[bytes] = []
        self._speech_windows = 0
        self._silence_run = 0

    def feed(self, pcm: bytes) -> list[SegmenterEvent]:
        """Any chunk size is fine; audio is re-windowed internally."""
        self._pending += pcm
        events: list[SegmenterEvent] = []
        while len(self._pending) >= WINDOW_BYTES:
            window, self._pending = self._pending[:WINDOW_BYTES], self._pending[WINDOW_BYTES:]
            events.extend(self._step(window))
        return events

    def snapshot(self) -> bytes:
        """Audio of the utterance in progress so far (empty when not speaking): used for partial transcripts."""
        return b"".join(self._buffer) if self._speaking else b""

    def flush(self) -> list[SegmenterEvent]:
        """Call ended mid-speech: close whatever was in progress instead of dropping it."""
        if not self._speaking:
            return []
        return self._finish(forced=False)

    def _step(self, window: bytes) -> list[SegmenterEvent]:
        samples = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
        prob = self._vad.probability(samples)
        cfg = self._cfg
        if not self._speaking:
            self._preroll.append(window)
            self._on_run = self._on_run + 1 if prob >= cfg.on_threshold else 0
            if self._on_run * WINDOW_MS >= cfg.start_ms:
                self._speaking = True
                self._buffer = list(self._preroll)
                self._speech_windows = self._on_run
                self._silence_run = 0
                return [SpeechStarted()]
            return []
        self._buffer.append(window)
        if prob < cfg.off_threshold:
            self._silence_run += 1
        else:
            self._silence_run = 0
            self._speech_windows += 1
        if self._silence_run * WINDOW_MS >= cfg.silence_ms:
            return self._finish(forced=False)
        if len(self._buffer) * WINDOW_MS >= cfg.max_utterance_ms:
            return self._finish(forced=True)
        return []

    def _finish(self, *, forced: bool) -> list[SegmenterEvent]:
        cfg = self._cfg
        speech_ms = self._speech_windows * WINDOW_MS
        trim = max(0, self._silence_run - cfg.tail_ms // WINDOW_MS)
        windows = self._buffer[:len(self._buffer) - trim] if trim else self._buffer
        pcm = b"".join(windows)
        self._reset_utterance()
        self._preroll.clear()
        self._vad.reset()
        if speech_ms < cfg.min_speech_ms:
            return [SpeechDiscarded()]
        return [SpeechEnded(pcm=pcm, speech_ms=speech_ms, forced=forced)]
