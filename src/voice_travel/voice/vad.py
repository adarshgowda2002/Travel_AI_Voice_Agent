"""Voice activity detection. Answers one question per 32 ms window: how likely is this speech?
Segmenting that into utterances is segmenter.py's job; deciding what an utterance means is the agent's."""
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512                 # 32 ms at 16 kHz (the window size Silero v5 requires)
CONTEXT_SAMPLES = 64                 # Silero v5 expects the previous window's tail prepended


class VAD(Protocol):
    def reset(self) -> None: ...

    def probability(self, window: np.ndarray) -> float:
        """window: float32 mono, exactly WINDOW_SAMPLES samples in [-1, 1]."""
        ...


class SileroVAD:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found at {model_path}; run `fetch-voice-models`")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1      # tiny model: one thread per call
        self._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, window: np.ndarray) -> float:
        if window.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"expected {WINDOW_SAMPLES} samples, got {window.shape}")
        x = np.concatenate([self._context, window.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0][0])
