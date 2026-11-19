"""The reference architecture's guarantees, as executable checks. A failure here means the voice layer
broke one of its non-negotiable rules (always listening, barge-in always on, never speaks on its own)."""
import inspect
import re
from pathlib import Path

from voice_fakes import FakeTTS, ScriptedSTT, ScriptTransport, loud, quiet, run_call, types, wait_for

from voice_travel.voice import pipeline as pipeline_module
from voice_travel.voice import stt as stt_module
from voice_travel.voice.stt import Transcript

VOICE_DIR = Path(pipeline_module.__file__).parent
SOURCES = {p.name: p.read_text() for p in VOICE_DIR.glob("*.py")}
AGENT_SIDE = {"dev_echo_agent.py"}        # the stand-in agent is allowed to speak: it IS the agent


def test_never_speaks_unprompted_even_after_speech_and_errors():
    """Caller speech, an STT failure and a transport hiccup must produce NO audio: only the agent speaks."""
    transport = ScriptTransport([(0, loud(640) + quiet(800) + loud(640) + quiet(800))])

    async def driver(bus, _c):
        await wait_for(bus, lambda e: e.type == "transport.error")
        await wait_for(bus, lambda e: e.type == "utterance.final")
    events = run_call(transport, driver=driver, tts=FakeTTS(),
                      stt=ScriptedSTT([RuntimeError("stt down"), Transcript("hello")]))
    assert transport.play_calls == 0
    assert not {"playback.started", "playback.ended"} & set(types(events))


def test_audio_keeps_flowing_in_while_the_agent_is_speaking():
    """Always-listening: speech during playback is captured, transcribed and flagged, not dropped."""
    from voice_fakes import speak
    transport = ScriptTransport([(0.2, loud(640) + quiet(800))])

    async def driver(bus, _c):
        await bus.publish_down(speak("a long answer"))
        await wait_for(bus, lambda e: e.type == "utterance.final")
    events = run_call(transport, driver=driver, tts=FakeTTS(frames=200),
                      stt=ScriptedSTT([Transcript("excuse me")]))
    final = next(e for e in events if e.type == "utterance.final")
    assert final.text == "excuse me" and final.during_playback


def test_barge_in_has_no_off_switch():
    params = inspect.signature(pipeline_module.CallPipeline.__init__).parameters
    assert not [name for name in params if "barge" in name or "interrupt" in name]


def test_playback_endings_are_always_reported():
    """Audio is best-effort: every started message ends with an explicit completed/played_ms report."""
    from voice_fakes import speak

    async def driver(bus, _c):
        await bus.publish_down(speak("one", speech_id="a"))
        await bus.publish_down(speak("two", speech_id="b"))
        await wait_for(bus, lambda e: e.type == "playback.ended" and e.speech_id == "b")
    events = run_call(ScriptTransport(), driver=driver, tts=FakeTTS(frames=2))
    started = {e.speech_id for e in events if e.type == "playback.started"}
    ended = {e.speech_id for e in events if e.type == "playback.ended"}
    assert started == ended == {"a", "b"}


def test_only_the_agent_side_ever_constructs_speech():
    """No module in the voice layer may build a Speak command (that is what hardcoded speech would look like)."""
    offenders = [name for name, src in SOURCES.items()
                 if name not in AGENT_SIDE | {"events.py"} and re.search(r"\bSpeak\(", src)]
    assert offenders == []


def test_no_hardcoded_fallback_or_filler_phrases():
    phrases = ["try again", "technical difficult", "sorry", "please wait", "one moment", "an error",
               "still checking", "i'm almost done"]
    offenders = [(name, p) for name, src in SOURCES.items() if name not in AGENT_SIDE
                 for p in phrases if p in src.lower()]
    assert offenders == []


def test_stt_module_does_no_interpretation():
    """STT is text only: no intent, cleanup or normalization helpers."""
    names = {n.lower() for n in dir(stt_module)}
    assert not {n for n in names if "intent" in n or "normalize" in n or "clean" in n}
