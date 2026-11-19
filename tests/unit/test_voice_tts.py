"""Real Piper voice checks. Skipped until the voice has been fetched (`fetch-voice-models`)."""
import asyncio
import time

import numpy as np
import pytest

from voice_travel.config import Settings
from voice_travel.voice.tts import PiperTTS

MODEL = Settings(_env_file=None).tts_voice_path
needs_voice = pytest.mark.skipif(not MODEL.exists(), reason="run fetch-voice-models first")


def test_missing_voice_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-voice-models"):
        PiperTTS(tmp_path / "nope.onnx")


@needs_voice
def test_synthesizes_one_chunk_per_sentence_with_real_audio():
    tts = PiperTTS(MODEL)

    async def go():
        return [f async for f in tts.synthesize("Hello there. This is a test of the voice.")]
    frames = asyncio.run(go())
    assert len(frames) == 2                                  # streamed sentence by sentence
    assert {f.sample_rate for f in frames} == {22050}
    audio = np.frombuffer(b"".join(f.pcm for f in frames), dtype=np.int16)
    assert 1.0 < len(audio) / 22050 < 8.0 and np.abs(audio).max() > 1000     # a few seconds of real speech


@needs_voice
def test_closing_early_stops_synthesis_promptly():
    """Barge-in closes the iterator after the first sentence: it must return fast, not finish the paragraph."""
    tts = PiperTTS(MODEL)
    text = " ".join(["This is another fairly long sentence about travelling to Dubai."] * 12)

    async def go():
        t0 = time.monotonic()
        stream = tts.synthesize(text)
        first = await stream.__anext__()
        await stream.aclose()
        return first, time.monotonic() - t0
    first, elapsed = asyncio.run(go())
    assert first.pcm and elapsed < 5
