import asyncio

import fakeredis

from voice_travel.voice import events as ev
from voice_travel.voice.bus import START, RedisStreamBus


def make_bus(maxlen=1000):
    return RedisStreamBus(fakeredis.FakeAsyncRedis(decode_responses=True), maxlen=maxlen)


def test_up_events_arrive_in_order():
    async def go():
        bus = make_bus()
        await bus.publish_up(ev.UtteranceFinal(call_id="S1", utterance_id="u1", text="one"))
        await bus.publish_up(ev.UtteranceFinal(call_id="S1", utterance_id="u2", text="two"))
        return await bus.read_up("S1")
    got = asyncio.run(go())
    assert [e.text for _id, e in got] == ["one", "two"]


def test_consumer_resumes_after_last_id():
    async def go():
        bus = make_bus()
        await bus.publish_up(ev.CallStarted(call_id="S1"))
        first = await bus.read_up("S1", after=START)
        await bus.publish_up(ev.CallEnded(call_id="S1"))
        return first, await bus.read_up("S1", after=first[-1][0])
    first, rest = asyncio.run(go())
    assert [type(e).__name__ for _i, e in first] == ["CallStarted"]
    assert [type(e).__name__ for _i, e in rest] == ["CallEnded"]


def test_calls_and_directions_are_isolated():
    async def go():
        bus = make_bus()
        await bus.publish_down(ev.Speak(call_id="S1", speech_id="p1", text="Hello"))
        await bus.publish_up(ev.CallStarted(call_id="S2"))
        return (await bus.read_down("S1"), await bus.read_up("S1"), await bus.read_down("S2"))
    down_s1, up_s1, down_s2 = asyncio.run(go())
    assert len(down_s1) == 1 and up_s1 == [] and down_s2 == []


def test_read_with_nothing_new_returns_empty():
    assert asyncio.run(make_bus().read_up("S1")) == []


def test_stream_is_trimmed():
    async def go():
        bus = make_bus(maxlen=10)
        for i in range(200):
            await bus.publish_up(ev.Tick(call_id="S1", elapsed_ms=i))
        return await bus.read_up("S1", count=1000)
    assert len(asyncio.run(go())) < 200
