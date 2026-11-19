"""Real Silero model checks. Skipped when the model has not been fetched (`fetch-voice-models`)."""
import numpy as np
import pytest

from voice_travel.config import Settings
from voice_travel.voice.vad import WINDOW_SAMPLES, SileroVAD

MODEL = Settings(_env_file=None).vad_model_path
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run fetch-voice-models first")


def test_silence_is_not_speech_and_state_carries_across_windows():
    vad = SileroVAD(MODEL)
    probs = [vad.probability(np.zeros(WINDOW_SAMPLES, dtype=np.float32)) for _ in range(20)]
    assert max(probs) < 0.2


def test_wrong_window_size_is_rejected():
    with pytest.raises(ValueError):
        SileroVAD(MODEL).probability(np.zeros(100, dtype=np.float32))


def test_missing_model_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-voice-models"):
        SileroVAD(tmp_path / "nope.onnx")
