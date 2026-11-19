"""ORM models: the SINGLE definition of the schema. Migrations are generated from these (Alembic)."""
import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    event,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from voice_travel.utils import phone_key

# jsonb on PostgreSQL, plain JSON elsewhere (SQLite in tests)
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('end_user', 'partner', 'admin')", name="ck_users_role"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    phone: Mapped[str | None] = mapped_column(String)
    phone_key: Mapped[str | None] = mapped_column(String, index=True)   # derived from phone, see listener below
    home_city: Mapped[str | None] = mapped_column(String)
    role: Mapped[str] = mapped_column(String, nullable=False, default="end_user")


Index("ix_users_name_lower", func.lower(User.name))


@event.listens_for(User, "before_insert")
@event.listens_for(User, "before_update")
def _derive_phone_key(mapper, connection, target: User) -> None:
    target.phone_key = phone_key(target.phone)


class Trip(Base):
    __tablename__ = "trips"
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'planned', 'booked', 'cancelled')", name="ck_trips_status"),
        CheckConstraint("end_date >= start_date", name="ck_trips_dates"),
        Index("ix_trips_user_status", "user_id", "status"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    destination: Mapped[str] = mapped_column(String, nullable=False)
    start_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    end_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    budget: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class Itinerary(Base):
    __tablename__ = "itineraries"
    __table_args__ = (UniqueConstraint("trip_id", "day_no", name="uq_itinerary_trip_day"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.id", ondelete="CASCADE"), nullable=False)
    day_no: Mapped[int] = mapped_column(Integer, nullable=False)
    activities: Mapped[str | None] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(String)


class Session(Base):
    """A voice session (LiveKit room)."""
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_user_started", "user_id", "started_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    trip_id: Mapped[str | None] = mapped_column(ForeignKey("trips.id"))
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    channel: Mapped[str] = mapped_column(String, nullable=False, default="voice")
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_sec: Mapped[int | None] = mapped_column(Integer)


class Booking(Base):
    """Flight/hotel booking. session_id is NULL for web bookings."""
    __tablename__ = "bookings"
    __table_args__ = (
        CheckConstraint("channel IN ('voice', 'web')", name="ck_bookings_channel"),
        CheckConstraint("type IN ('flight', 'hotel')", name="ck_bookings_type"),
        CheckConstraint("status IN ('confirmed', 'cancelled')", name="ck_bookings_status"),
        CheckConstraint("(channel = 'voice') = (session_id IS NOT NULL)", name="ck_bookings_channel_session"),
        Index("ix_bookings_trip", "trip_id"),
        Index("ix_bookings_user_status", "user_id", "status"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("sessions.id"))
    channel: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str | None] = mapped_column(String)
    reference: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class Conversation(Base):
    """One exchange: user utterance + agent reply."""
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_no", name="uq_conv_session_turn"),
        Index("ix_conv_intent", "intent"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    turn_no: Mapped[int] = mapped_column(Integer, nullable=False)
    user_text: Mapped[str] = mapped_column(String, nullable=False)
    agent_text: Mapped[str] = mapped_column(String, nullable=False)
    intent: Mapped[str] = mapped_column(String, nullable=False)
    agent: Mapped[str | None] = mapped_column(String)
    barge_in: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rag_doc_id: Mapped[str | None] = mapped_column(String)       # Qdrant doc id when retrieval was used
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class AudioLog(Base):
    """Audio metadata; the file itself lives in MinIO/S3 at file_path."""
    __tablename__ = "audio_logs"
    __table_args__ = (
        CheckConstraint("type IN ('voice_in', 'voice_out')", name="ck_audio_type"),
        Index("ix_audio_session_turn", "session_id", "turn_no"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    turn_no: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(Numeric(8, 2, asdecimal=False))
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    accent: Mapped[str | None] = mapped_column(String)
    noise_profile: Mapped[str | None] = mapped_column(String)
    stt_model: Mapped[str | None] = mapped_column(String)
    stt_confidence: Mapped[float | None] = mapped_column(Numeric(4, 3, asdecimal=False))
    source: Mapped[str | None] = mapped_column(String)


class AgentTask(Base):
    """Every tool call made by an agent (flights.search, booking.create...)."""
    __tablename__ = "agent_tasks"
    __table_args__ = (
        CheckConstraint("status IN ('success', 'failed')", name="ck_tasks_status"),
        Index("ix_tasks_session", "session_id", "created_at"),
        # partial index: ops alerting only ever reads failed tasks
        Index("ix_tasks_failed", "created_at", postgresql_where=text("status = 'failed'"),
              sqlite_where=text("status = 'failed'")),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    agent: Mapped[str] = mapped_column(String, nullable=False)
    tool: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    request: Mapped[dict | None] = mapped_column(JSONType)
    result: Mapped[dict | None] = mapped_column(JSONType)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class AuditLog(Base):
    """Append-only activity log."""
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_user_ts", "user_id", "timestamp"),
        Index("ix_audit_type_ts", "event_type", "timestamp"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    details: Mapped[dict | None] = mapped_column(JSONType)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
