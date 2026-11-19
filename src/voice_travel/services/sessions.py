"""Voice session + conversation history (the turn manager persists through this)."""
from sqlalchemy import func, select, update

from voice_travel.infra.postgres import Conversation, Database
from voice_travel.infra.postgres import Session as VoiceSession
from voice_travel.services.errors import ServiceError, service_call
from voice_travel.utils import new_id, utcnow


class SessionService:
    def __init__(self, db: Database) -> None:
        self._db = db

    @service_call
    def start(self, user_id: str, trip_id: str | None = None) -> VoiceSession:
        voice_session = VoiceSession(id=new_id("S"), user_id=user_id, trip_id=trip_id, status="active",
                                     channel="voice", started_at=utcnow(), turns=0)
        with self._db.session() as s:
            s.add(voice_session)
        return voice_session

    @service_call
    def log_turn(self, session_id: str, user_text: str, agent_text: str, intent: str, agent: str | None = None,
                 latency_ms: int | None = None, barge_in: bool = False,
                 rag_doc_id: str | None = None) -> Conversation:
        """Append one exchange. UNIQUE(session_id, turn_no) guards against concurrent writers."""
        with self._db.session() as s:
            if s.get(VoiceSession, session_id) is None:
                raise ServiceError(f"session {session_id} does not exist")
            last = s.scalar(select(func.max(Conversation.turn_no)).where(Conversation.session_id == session_id))
            conv = Conversation(id=new_id("C"), session_id=session_id, turn_no=(last or 0) + 1,
                                user_text=user_text, agent_text=agent_text, intent=intent, agent=agent,
                                latency_ms=latency_ms, barge_in=barge_in, rag_doc_id=rag_doc_id, ts=utcnow())
            s.add(conv)
            s.execute(update(VoiceSession).where(VoiceSession.id == session_id)
                      .values(turns=VoiceSession.turns + 1))
        return conv

    @service_call
    def recent_turns(self, session_id: str, n: int = 5) -> list[Conversation]:
        """Last N exchanges, oldest first: the context window handed to the conversation agent."""
        with self._db.session() as s:
            q = (select(Conversation).where(Conversation.session_id == session_id)
                 .order_by(Conversation.turn_no.desc()).limit(n))
            return list(s.scalars(q))[::-1]

    @service_call
    def end(self, session_id: str) -> VoiceSession | None:
        with self._db.session() as s:
            voice_session = s.get(VoiceSession, session_id)
            if voice_session is None:
                return None
            now = utcnow()
            voice_session.status, voice_session.ended_at = "ended", now
            voice_session.duration_sec = int((now - voice_session.started_at).total_seconds())
        return voice_session
