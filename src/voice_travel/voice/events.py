"""Wire contract between the voice layer and the agent side. The ONLY coupling point between them.

Authority flows one way (agent -> voice layer -> transport). The voice layer reports what it heard and
what it played; it never decides what to say, never ends a conversational turn, and never speaks on its own.

  Voice -> Agent ("up"):   call.started, call.ended, utterance.partial, utterance.final,
                           playback.started, playback.ended, tick, transport.error
  Agent -> Voice ("down"): speak, cancel_speech, tick_control, hangup
"""
import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    ts: float = Field(default_factory=time.time)


# ---- voice layer -> agent ---------------------------------------------------
class CallStarted(_Message):
    type: Literal["call.started"] = "call.started"
    metadata: dict[str, str] = Field(default_factory=dict)


class CallEnded(_Message):
    type: Literal["call.ended"] = "call.ended"


class UtterancePartial(_Message):
    type: Literal["utterance.partial"] = "utterance.partial"
    utterance_id: str
    text: str


class UtteranceFinal(_Message):
    """One VAD-segmented utterance. NOT a conversational turn: the agent decides what it means
    (new request, interruption, clarification...). during_playback tells it the user spoke over the agent."""
    type: Literal["utterance.final"] = "utterance.final"
    utterance_id: str
    text: str
    confidence: float | None = None
    during_playback: bool = False


class PlaybackStarted(_Message):
    type: Literal["playback.started"] = "playback.started"
    speech_id: str


class PlaybackEnded(_Message):
    """Playback may be cut short (barge-in): completed=False with how much was actually played.
    The agent must not assume a spoken message was fully heard."""
    type: Literal["playback.ended"] = "playback.ended"
    speech_id: str
    completed: bool
    played_ms: int = Field(ge=0)


class Tick(_Message):
    """Elapsed-time signal while the agent has asked for ticks. The agent decides whether to say anything."""
    type: Literal["tick"] = "tick"
    elapsed_ms: int = Field(ge=0)


class TransportError(_Message):
    """Logged and reported silently. NEVER turned into speech by the voice layer."""
    type: Literal["transport.error"] = "transport.error"
    component: str
    detail: str


UpEvent = Annotated[
    CallStarted | CallEnded | UtterancePartial | UtteranceFinal | PlaybackStarted | PlaybackEnded | Tick
    | TransportError,
    Field(discriminator="type"),
]


# ---- agent -> voice layer ---------------------------------------------------
class Speak(_Message):
    """Render and play this text. The agent may send several per request (acknowledge, filler, answer)."""
    type: Literal["speak"] = "speak"
    speech_id: str
    text: str = Field(min_length=1)


class CancelSpeech(_Message):
    type: Literal["cancel_speech"] = "cancel_speech"
    speech_id: str


class TickControl(_Message):
    """Start/stop elapsed-time ticks (the agent asks for them while a task is pending)."""
    type: Literal["tick_control"] = "tick_control"
    enabled: bool
    interval_ms: int = Field(default=2000, ge=250)


class Hangup(_Message):
    type: Literal["hangup"] = "hangup"


DownCommand = Annotated[Speak | CancelSpeech | TickControl | Hangup, Field(discriminator="type")]

_UP: TypeAdapter[UpEvent] = TypeAdapter(UpEvent)
_DOWN: TypeAdapter[DownCommand] = TypeAdapter(DownCommand)


def encode(message: BaseModel) -> str:
    return message.model_dump_json()


def decode_up(raw: str | bytes) -> UpEvent:
    return _UP.validate_json(raw)


def decode_down(raw: str | bytes) -> DownCommand:
    return _DOWN.validate_json(raw)
