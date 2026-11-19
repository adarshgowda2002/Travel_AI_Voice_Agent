"""The voice<->agent contract. Several tests encode guarantees from the reference architecture."""
import pytest
from pydantic import ValidationError

from voice_travel.voice import events as ev


def test_up_events_round_trip():
    for e in [ev.CallStarted(call_id="S1", metadata={"channel": "web"}), ev.CallEnded(call_id="S1"),
              ev.UtterancePartial(call_id="S1", utterance_id="u1", text="plan a"),
              ev.UtteranceFinal(call_id="S1", utterance_id="u1", text="plan a trip", confidence=0.9),
              ev.PlaybackStarted(call_id="S1", speech_id="p1"),
              ev.PlaybackEnded(call_id="S1", speech_id="p1", completed=False, played_ms=640),
              ev.Tick(call_id="S1", elapsed_ms=2000), ev.TransportError(call_id="S1", component="stt", detail="x")]:
        assert ev.decode_up(ev.encode(e)) == e


def test_down_commands_round_trip():
    for c in [ev.Speak(call_id="S1", speech_id="p1", text="Hello"), ev.CancelSpeech(call_id="S1", speech_id="p1"),
              ev.TickControl(call_id="S1", enabled=True), ev.Hangup(call_id="S1")]:
        assert ev.decode_down(ev.encode(c)) == c


def test_directions_are_not_interchangeable():
    """Authority flows one way: an agent command can't be decoded as a voice event, or vice versa."""
    with pytest.raises(ValidationError):
        ev.decode_up(ev.encode(ev.Speak(call_id="S1", speech_id="p1", text="Hi")))
    with pytest.raises(ValidationError):
        ev.decode_down(ev.encode(ev.UtteranceFinal(call_id="S1", utterance_id="u", text="hi")))


def test_partial_playback_is_representable():
    """Audio is best-effort: a barge-in cuts speech short and the agent is told how much played."""
    e = ev.PlaybackEnded(call_id="S1", speech_id="p1", completed=False, played_ms=300)
    assert not e.completed and e.played_ms == 300


def test_utterance_carries_no_turn_boundary_or_intent():
    """The voice layer reports speech only. Turn boundaries and intent belong to the agent."""
    fields = set(ev.UtteranceFinal.model_fields)
    assert fields == {"call_id", "ts", "type", "utterance_id", "text", "confidence", "during_playback"}


def test_speak_rejects_empty_text_and_unknown_fields():
    with pytest.raises(ValidationError):
        ev.Speak(call_id="S1", speech_id="p1", text="")
    with pytest.raises(ValidationError):
        ev.decode_down('{"type":"speak","call_id":"S1","speech_id":"p1","text":"hi","fallback":"oops"}')


def test_no_fallback_speech_event_exists():
    """The voice layer may report errors but has no event/command that makes it speak by itself."""
    assert not any("fallback" in name or "apolog" in name for name in dir(ev))
