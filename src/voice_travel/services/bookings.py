"""Booking lookups and the write side of the booking tools.
Every write runs in ONE transaction: bookings and trips change together or not at all."""
import datetime as dt
import secrets
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session as OrmSession

from voice_travel.domain.pricing import CHANGE_FEE_INR, refund_amount
from voice_travel.infra.postgres import Booking, Database, Trip, User
from voice_travel.services.errors import ServiceError, service_call
from voice_travel.utils import new_id, utcnow

_REF_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I: unambiguous when read aloud


def _new_reference() -> str:
    return "TRV" + "".join(secrets.choice(_REF_ALPHABET) for _ in range(6))


def _confirmed_for_update(s: OrmSession, reference: str) -> Booking | None:
    """Lock the booking row (no double-modify race on Postgres) and return it only if still confirmed."""
    q = select(Booking).where(Booking.reference == reference.strip().upper()).with_for_update()
    booking = s.scalars(q).first()
    return booking if booking and booking.status == "confirmed" else None


class BookingService:
    def __init__(self, db: Database) -> None:
        self._db = db

    @service_call
    def fetch_by_reference(self, reference: str) -> Booking | None:
        with self._db.session() as s:
            return s.scalars(select(Booking).where(Booking.reference == reference.strip().upper())).first()

    @service_call
    def fetch_user_bookings(self, user_id: str, status: str | None = None, limit: int = 10) -> list[Booking]:
        q = select(Booking).where(Booking.user_id == user_id)
        if status:
            q = q.where(Booking.status == status)
        with self._db.session() as s:
            return list(s.scalars(q.order_by(Booking.created_at.desc()).limit(limit)))

    @service_call
    def fetch_latest_summary(self, name: str) -> dict[str, Any] | None:
        """users JOIN bookings JOIN trips: answers 'what is my latest booking?'."""
        q = (select(Booking, Trip)
             .join(User, User.id == Booking.user_id)
             .join(Trip, Trip.id == Booking.trip_id)
             .where(func.lower(User.name) == name.strip().lower())
             .order_by(Booking.created_at.desc()).limit(1))
        with self._db.session() as s:
            row = s.execute(q).first()
        if row is None:
            return None
        b, t = row
        return {"reference": b.reference, "type": b.type, "status": b.status, "amount": b.amount,
                "currency": b.currency, "provider": b.provider, "destination": t.destination,
                "start_date": t.start_date.isoformat()}

    @service_call
    def create(self, trip_id: str, user_id: str, session_id: str, type: str, provider: str,
               amount: int, reference: str | None = None) -> Booking:
        """booking.create tool: insert the booking and mark the trip booked, atomically."""
        with self._db.session() as s:
            if s.get(Trip, trip_id) is None:
                raise ServiceError(f"trip {trip_id} does not exist")
            now = utcnow()
            booking = Booking(id=new_id("B"), trip_id=trip_id, user_id=user_id, session_id=session_id,
                              channel="voice", type=type, provider=provider,
                              reference=reference or _new_reference(), status="confirmed", amount=amount,
                              currency="INR", created_at=now, updated_at=now)
            s.add(booking)
            s.execute(update(Trip).where(Trip.id == trip_id).values(status="booked"))
        return booking

    @service_call
    def modify_date(self, reference: str, new_start: dt.date) -> dict[str, Any] | None:
        """booking.modify tool: move the trip to new_start (keeping its length) and add the change fee."""
        with self._db.session() as s:
            booking = _confirmed_for_update(s, reference)
            if booking is None:
                return None
            trip = s.get(Trip, booking.trip_id)
            assert trip is not None   # FK guarantees it
            length = trip.end_date - trip.start_date
            trip.start_date, trip.end_date = new_start, new_start + length
            booking.amount += CHANGE_FEE_INR
            booking.updated_at = utcnow()
            return {"reference": booking.reference, "new_start": new_start.isoformat(),
                    "change_fee": CHANGE_FEE_INR}

    @service_call
    def cancel(self, reference: str) -> dict[str, Any] | None:
        """booking.cancel tool. Returns the refund, or None if missing / already cancelled."""
        with self._db.session() as s:
            booking = _confirmed_for_update(s, reference)
            if booking is None:
                return None
            booking.status, booking.updated_at = "cancelled", utcnow()
            s.flush()
            remaining = s.scalar(select(func.count()).select_from(Booking)
                                 .where(Booking.trip_id == booking.trip_id, Booking.status == "confirmed"))
            if not remaining:
                s.execute(update(Trip).where(Trip.id == booking.trip_id).values(status="cancelled"))
            return {"reference": booking.reference, "refund": refund_amount(booking.amount)}
