"""Services against the real generated dataset (SQLite by default, Postgres via TEST_DATABASE_URL)."""
import datetime as dt

import pytest
from sqlalchemy import func, select

from voice_travel.infra.postgres import AgentTask, Booking, Conversation, Itinerary, Session, Trip, User
from voice_travel.services import ServiceError


def count(db, model, *where):
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where))


def first(db, model, *where):
    with db.session() as s:
        return s.scalars(select(model).where(*where).limit(1)).first()


# ---- load fidelity ----------------------------------------------------------
def test_counts_match_dataset(db):
    assert count(db, User) == 120
    assert count(db, Session) == 500
    assert count(db, Conversation) == 2506
    assert count(db, Booking) == 333


def test_no_orphans(db):
    with db.session() as s:
        orphan_convs = s.scalar(select(func.count()).select_from(Conversation)
                                .where(Conversation.session_id.not_in(select(Session.id))))
        orphan_bookings = s.scalar(select(func.count()).select_from(Booking)
                                   .where(Booking.trip_id.not_in(select(Trip.id))))
    assert (orphan_convs, orphan_bookings) == (0, 0)


# ---- identity ---------------------------------------------------------------
def test_phone_lookup_handles_formats(db, identity):
    u = first(db, User)
    digits10 = u.phone_key
    for fmt in (u.phone, digits10, "0" + digits10, "+91" + digits10):
        assert identity.fetch_user_by_phone(fmt).id == u.id, fmt
    assert identity.fetch_user_by_phone("12345") is None


def test_verify_identity(db, identity):
    u = first(db, User)
    assert identity.verify_identity(u.name.upper(), u.phone)
    assert not identity.verify_identity("Somebody Else", u.phone)


# ---- bookings ---------------------------------------------------------------
def test_latest_summary_joins_user_trip_booking(db, bookings):
    b = first(db, Booking, Booking.status == "confirmed")
    u = first(db, User, User.id == b.user_id)
    summary = bookings.fetch_latest_summary(u.name)
    assert summary is not None and "destination" in summary
    assert bookings.fetch_latest_summary("Nobody Here") is None


def test_create_modify_cancel_flow(db, bookings):
    trip = first(db, Trip, Trip.status == "draft")
    sess = first(db, Session, Session.user_id == trip.user_id)
    b = bookings.create(trip.id, trip.user_id, sess.id, "flight", "IndiGo", 20000)
    assert first(db, Trip, Trip.id == trip.id).status == "booked"

    old = first(db, Trip, Trip.id == trip.id)
    res = bookings.modify_date(b.reference, old.start_date + dt.timedelta(days=5))
    assert res["change_fee"] == 2500
    new = first(db, Trip, Trip.id == trip.id)
    assert new.end_date - new.start_date == old.end_date - old.start_date      # trip length preserved
    assert first(db, Booking, Booking.id == b.id).amount == 22500

    assert bookings.cancel(b.reference)["refund"] == 20250
    assert first(db, Trip, Trip.id == trip.id).status == "cancelled"
    assert bookings.cancel(b.reference) is None                               # already cancelled
    assert bookings.modify_date(b.reference, dt.date(2027, 1, 1)) is None


def test_create_for_missing_trip_raises_and_writes_nothing(db, bookings):
    before = count(db, Booking)
    with pytest.raises(ServiceError):
        bookings.create("T9999", "U0001", "S0001", "flight", "IndiGo", 1)
    assert count(db, Booking) == before


def test_constraint_violation_rolls_back_whole_transaction(db, bookings):
    """A voice booking without a session violates ck_bookings_channel_session: nothing may persist."""
    trip = first(db, Trip, Trip.status == "draft")
    before = count(db, Booking)
    with pytest.raises(ServiceError):
        bookings.create(trip.id, trip.user_id, None, "flight", "IndiGo", 1000)
    assert count(db, Booking) == before
    assert first(db, Trip, Trip.id == trip.id).status == "draft"


# ---- sessions ---------------------------------------------------------------
def test_session_lifecycle_and_recent_turns(db, sessions):
    sess = sessions.start(first(db, User).id)
    for i in range(6):
        sessions.log_turn(sess.id, f"user {i}", f"agent {i}", "search_flights", agent="flight_agent", latency_ms=900)
    assert [c.turn_no for c in sessions.recent_turns(sess.id, 3)] == [4, 5, 6]   # last 3, oldest first
    ended = sessions.end(sess.id)
    assert (ended.status, ended.turns) == ("ended", 6)
    with pytest.raises(ServiceError):
        sessions.log_turn("S_DOES_NOT_EXIST", "x", "y", "defer")


# ---- trips / ops ------------------------------------------------------------
def test_itinerary_roundtrip(db, trips):
    trip = first(db, Trip, Trip.status == "draft")
    trips.save_itinerary(trip.id, ["Eiffel Tower", "Louvre", "Seine Cruise"])
    assert [i.day_no for i in trips.fetch_itinerary(trip.id)] == [1, 2, 3]
    assert first(db, Trip, Trip.id == trip.id).status == "planned"
    trips.save_itinerary(trip.id, ["Only day"])                               # replaces, not appends
    assert count(db, Itinerary, Itinerary.trip_id == trip.id) == 1


def test_ops_logging(db, ops):
    sess = first(db, Session)
    before = count(db, AgentTask)
    ops.log_agent_task(sess.id, "flight_agent", "booking.create", {"a": 1}, {"error": "provider_timeout"}, "failed")
    assert count(db, AgentTask) == before + 1
    assert any((t.result or {}).get("error") == "provider_timeout" for t in ops.failed_tasks(500))
    assert ops.log_event("session_started", first(db, User).id, {"k": "v"}).details["k"] == "v"
