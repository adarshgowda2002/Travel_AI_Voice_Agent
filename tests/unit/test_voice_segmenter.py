import numpy as np
import pytest

from voice_travel.voice.segmenter import (
    WINDOW_BYTES,
    SegmenterConfig,
    SpeechDiscarded,
    SpeechEnded,
    SpeechStarted,
    UtteranceSegmenter,
)


class AmplitudeVAD:
    """Stand-in for Silero: loud window = speech. Makes segmentation logic testable without a model."""
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def probability(self, window):
        return 0.9 if float(np.abs(window).max()) > 0.1 else 0.0


def loud(ms: int) -> bytes:
    return (np.full(ms * 16, 10_000, dtype=np.int16)).tobytes()


def quiet(ms: int) -> bytes:
    return bytes(ms * 16 * 2)


def run(audio: bytes, chunk: int = WINDOW_BYTES, config: SegmenterConfig | None = None):
    seg = UtteranceSegmenter(AmplitudeVAD(), config)
    events = []
    for i in range(0, len(audio), chunk):
        events += seg.feed(audio[i:i + chunk])
    return seg, events + seg.flush()


def names(events):
    return [type(e).__name__ for e in events]


def test_one_utterance_with_preroll_and_trimmed_tail():
    # 640 ms silence, 640 ms speech, 1 s silence. Buffer = 7 pre-roll windows + 20 speech + 5 tail windows.
    _, events = run(quiet(640) + loud(640) + quiet(1000))
    assert names(events) == ["SpeechStarted", "SpeechEnded"]
    ended = events[1]
    assert isinstance(ended, SpeechEnded) and not ended.forced
    assert ended.speech_ms == 640
    assert len(ended.pcm) == (7 + 20 + 5) * WINDOW_BYTES


def test_chunk_size_does_not_change_the_result():
    audio = quiet(640) + loud(640) + quiet(1000)
    _, a = run(audio, chunk=WINDOW_BYTES)
    _, b = run(audio, chunk=320)           # 10 ms chunks: not aligned to the 32 ms windows
    assert a[1].pcm == b[1].pcm and names(a) == names(b)


def test_two_utterances_keep_listening_between_them():
    _, events = run(loud(640) + quiet(800) + loud(640) + quiet(800))
    assert names(events) == ["SpeechStarted", "SpeechEnded", "SpeechStarted", "SpeechEnded"]


def test_short_burst_is_discarded():
    _, events = run(quiet(320) + loud(128) + quiet(1000))          # ~128 ms: a click or a cough
    assert names(events) == ["SpeechStarted", "SpeechDiscarded"]


def test_below_start_threshold_never_opens():
    _, events = run(quiet(320) + loud(64) + quiet(1000))           # 64 ms < start_ms (96)
    assert events == []


def test_pause_shorter_than_silence_window_does_not_split():
    _, events = run(loud(512) + quiet(320) + loud(512) + quiet(1000))      # 320 ms pause < 500 ms
    assert names(events) == ["SpeechStarted", "SpeechEnded"]
    assert events[1].speech_ms == 1024


def test_endless_speech_is_cut_at_max_length():
    cfg = SegmenterConfig(max_utterance_ms=2000)
    _, events = run(loud(5000), config=cfg)
    ended = [e for e in events if isinstance(e, SpeechEnded)]
    assert ended and ended[0].forced
    assert len(ended[0].pcm) <= (2000 // 32 + 1) * WINDOW_BYTES


def test_flush_closes_speech_in_progress_instead_of_dropping_it():
    seg, events = run(loud(640))                    # the call ended mid-sentence
    assert names(events) == ["SpeechStarted", "SpeechEnded"]
    assert seg.flush() == []


def test_flush_with_nothing_in_progress_is_empty():
    seg, _ = run(quiet(500))
    assert seg.flush() == []


def test_vad_state_is_reset_after_each_utterance():
    vad = AmplitudeVAD()
    seg = UtteranceSegmenter(vad)
    seg.feed(loud(640) + quiet(1000))
    assert vad.resets == 1


def test_event_types_are_what_the_pipeline_expects():
    assert {SpeechStarted, SpeechEnded, SpeechDiscarded}
    with pytest.raises(AttributeError):
        SpeechEnded(pcm=b"", speech_ms=1).speech_ms = 2        # frozen
