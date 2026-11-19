import asyncio

from voice_fakes import (
    CALL,
    FakeTTS,
    ScriptedSTT,
    ScriptTransport,
    loud,
    quiet,
    run_call,
    speak,
    types,
    wait_for,
)

from voice_travel.voice import events as ev
from voice_travel.voice.stt import Transcript

TWO_UTTERANCES = [(0, loud(640) + quiet(800) + loud(640) + quiet(800))]


def finals(events):
    return [e for e in events if e.type == "utterance.final"]


# ---- ingress ------------------------------------------------------------------
def test_ingress_emits_call_utterances_and_end_in_order():
    async def driver(bus, _c):
        await wait_for(bus, lambda e: e.type == "utterance.final" and e.text == "to dubai")
    events = run_call(ScriptTransport(TWO_UTTERANCES), driver=driver,
                      stt=ScriptedSTT([Transcript("plan a trip", 0.9), Transcript("to dubai", 0.8)]))
    assert types(events) == ["call.started", "utterance.final", "utterance.final", "call.ended"]
    assert [e.text for e in finals(events)] == ["plan a trip", "to dubai"]
    assert finals(events)[0].confidence == 0.9 and not finals(events)[0].during_playback


def test_empty_transcript_sends_nothing():
    async def driver(bus, _c):
        await wait_for(bus, lambda e: e.type == "utterance.final")
    events = run_call(ScriptTransport(TWO_UTTERANCES), driver=driver,
                      stt=ScriptedSTT([Transcript(""), Transcript("hello")]))
    assert [e.text for e in finals(events)] == ["hello"]


def test_stt_failure_is_reported_silently_and_the_call_continues():
    async def driver(bus, _c):
        await wait_for(bus, lambda e: e.type == "utterance.final")
    events = run_call(ScriptTransport(TWO_UTTERANCES), driver=driver,
                      stt=ScriptedSTT([RuntimeError("boom"), Transcript("still here")]))
    errors = [e for e in events if e.type == "transport.error"]
    assert errors[0].component == "stt" and errors[0].detail == "RuntimeError"
    assert [e.text for e in finals(events)] == ["still here"]
    assert "playback.started" not in types(events)           # an error never turns into speech


def test_transport_failure_is_reported_and_call_still_ends():
    events = run_call(ScriptTransport(fail_on_connect=True))
    assert types(events) == ["call.started", "transport.error", "call.ended"]
    assert events[1].component == "transport"


def test_listening_never_pauses_while_stt_is_busy():
    """The first transcription blocks until the SECOND utterance has been segmented. If ingestion waited
    on STT this would deadlock; it passes only because audio keeps flowing during transcription."""
    release = asyncio.Event

    class GatedSTT(ScriptedSTT):
        calls = 0
        gate = None

        async def transcribe(self, pcm):
            self.calls += 1
            if self.calls == 1:
                self.gate = self.gate or release()
                await asyncio.wait_for(self.gate.wait(), timeout=5)
            return Transcript(f"utterance {self.calls}")

    stt = GatedSTT()

    async def driver(bus, _c):
        await asyncio.sleep(0.3)            # both utterances are segmented by now; STT #1 is still blocked
        stt.gate = stt.gate or release()
        stt.gate.set()
        await wait_for(bus, lambda e: e.type == "utterance.final" and e.text == "utterance 2")
    events = run_call(ScriptTransport(TWO_UTTERANCES), stt=stt, driver=driver)
    assert [e.text for e in finals(events)] == ["utterance 1", "utterance 2"]


def test_partials_stream_while_talking_and_never_follow_their_final():
    class PartialSTT(ScriptedSTT):
        async def transcribe(self, pcm):
            return Transcript("so far" if len(pcm) < 40_000 else "the whole thing")

    async def driver(bus, _c):
        await wait_for(bus, lambda e: e.type == "utterance.final")
    transport = ScriptTransport([(0, loud(1200)), (0.7, quiet(800))])
    events = run_call(transport, stt=PartialSTT(), driver=driver, partials=True, partial_interval_ms=200)
    partials = [e for e in events if e.type == "utterance.partial"]
    final = finals(events)[0]
    assert partials and all(p.utterance_id == final.utterance_id for p in partials)
    assert types(events).index("utterance.final") > max(i for i, t in enumerate(types(events))
                                                        if t == "utterance.partial")


# ---- egress: speak, cancel, barge-in ---------------------------------------------
def test_speak_plays_and_reports_started_and_ended():
    transport = ScriptTransport()

    async def driver(bus, _c):
        await bus.publish_down(speak("Hello there"))
        await wait_for(bus, lambda e: e.type == "playback.ended")
    events = run_call(transport, driver=driver, tts=FakeTTS(frames=5))
    assert types(events) == ["call.started", "playback.started", "playback.ended", "call.ended"]
    ended = events[2]
    assert ended.speech_id == "p1" and ended.completed and ended.played_ms == 500
    assert transport.play_calls == 1


def test_several_speaks_play_in_order():
    async def driver(bus, _c):
        for i in (1, 2, 3):
            await bus.publish_down(speak(f"msg {i}", speech_id=f"p{i}"))
        await wait_for(bus, lambda e: e.type == "playback.ended" and e.speech_id == "p3")
    events = run_call(ScriptTransport(), driver=driver, tts=FakeTTS(frames=2))
    assert [e.speech_id for e in events if e.type == "playback.started"] == ["p1", "p2", "p3"]
    assert all(e.completed for e in events if e.type == "playback.ended")


def test_barge_in_cuts_the_agent_off_and_flags_the_utterance():
    """The caller talks over the agent: playback stops, queued speech is dropped, and everything is reported."""
    transport = ScriptTransport([(0.3, loud(640) + quiet(800))])        # user speaks 0.3 s into the agent's reply

    async def driver(bus, _c):
        await bus.publish_down(speak("a long answer", speech_id="p1"))
        await bus.publish_down(speak("and another", speech_id="p2"))
        await wait_for(bus, lambda e: e.type == "utterance.final")
    events = run_call(transport, driver=driver, tts=FakeTTS(frames=100),
                      stt=ScriptedSTT([Transcript("wait, change the date")]))
    ended = {e.speech_id: e for e in events if e.type == "playback.ended"}
    assert not ended["p1"].completed and 0 < ended["p1"].played_ms < 10_000
    assert not ended["p2"].completed and ended["p2"].played_ms == 0          # never started: flushed
    assert "p2" not in [e.speech_id for e in events if e.type == "playback.started"]
    assert finals(events)[0].during_playback
    assert transport.stop_calls >= 1


def test_cancel_speech_stops_the_playing_message():
    async def driver(bus, _c):
        await bus.publish_down(speak("long", speech_id="p1"))
        await wait_for(bus, lambda e: e.type == "playback.started")
        await bus.publish_down(ev.CancelSpeech(call_id=CALL, speech_id="p1"))
        await wait_for(bus, lambda e: e.type == "playback.ended")
    events = run_call(ScriptTransport(), driver=driver, tts=FakeTTS(frames=1000))
    ended = next(e for e in events if e.type == "playback.ended")
    assert not ended.completed


def test_tts_failure_is_silent_and_the_next_message_still_plays():
    async def driver(bus, _c):
        await bus.publish_down(speak("broken", speech_id="p1"))
        await bus.publish_down(speak("fine", speech_id="p2"))
        await wait_for(bus, lambda e: e.type == "playback.ended" and e.speech_id == "p2")
    events = run_call(ScriptTransport(), driver=driver, tts=FakeTTS(frames=2, fail_on={"broken"}))
    err = next(e for e in events if e.type == "transport.error")
    assert err.component == "playback" and err.detail == "RuntimeError"
    ended = {e.speech_id: e for e in events if e.type == "playback.ended"}
    assert not ended["p1"].completed and ended["p2"].completed


# ---- control: ticks, hangup, bad input -------------------------------------------
def test_ticks_flow_only_while_the_agent_asked_for_them():
    async def driver(bus, _c):
        await asyncio.sleep(0.5)                                          # no request yet: nothing
        assert "tick" not in types(await _seen(bus))
        await bus.publish_down(ev.TickControl(call_id=CALL, enabled=True, interval_ms=250))
        await wait_for(bus, lambda e: e.type == "tick" and e.elapsed_ms >= 500)
        await bus.publish_down(ev.TickControl(call_id=CALL, enabled=False))
        await asyncio.sleep(0.4)
        count = types(await _seen(bus)).count("tick")
        await asyncio.sleep(0.6)
        assert types(await _seen(bus)).count("tick") == count             # stopped
    events = run_call(ScriptTransport(), driver=driver)
    assert types(events).count("tick") >= 2


async def _seen(bus):
    return [e for _i, e in await bus.read_up(CALL) if e]


def test_hangup_ends_the_call():
    events = run_call(ScriptTransport())          # the harness hangs up immediately
    assert types(events) == ["call.started", "call.ended"]


def test_malformed_command_is_skipped_and_the_call_carries_on():
    async def driver(bus, client):
        await client.xadd(f"vta:call:{CALL}:events:down", {"data": '{"type":"speak","oops":true}'})
        await bus.publish_down(speak("still works", speech_id="p9"))
        await wait_for(bus, lambda e: e.type == "playback.ended" and e.speech_id == "p9")
    events = run_call(ScriptTransport(), driver=driver, tts=FakeTTS(frames=2))
    assert next(e for e in events if e.type == "playback.ended").completed
