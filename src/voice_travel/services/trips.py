from sqlalchemy import delete, select, update

from voice_travel.infra.postgres import Database, Itinerary, Trip
from voice_travel.services.errors import service_call
from voice_travel.utils import new_id


class TripService:
    def __init__(self, db: Database) -> None:
        self._db = db

    @service_call
    def fetch_user_trips(self, user_id: str, status: str | None = None, limit: int = 10) -> list[Trip]:
        q = select(Trip).where(Trip.user_id == user_id)
        if status:
            q = q.where(Trip.status == status)
        with self._db.session() as s:
            return list(s.scalars(q.order_by(Trip.created_at.desc()).limit(limit)))

    @service_call
    def fetch_trip(self, trip_id: str) -> Trip | None:
        with self._db.session() as s:
            return s.get(Trip, trip_id)

    @service_call
    def fetch_latest_trip(self, user_id: str) -> Trip | None:
        with self._db.session() as s:
            q = select(Trip).where(Trip.user_id == user_id).order_by(Trip.created_at.desc()).limit(1)
            return s.scalars(q).first()

    @service_call
    def fetch_itinerary(self, trip_id: str) -> list[Itinerary]:
        with self._db.session() as s:
            q = select(Itinerary).where(Itinerary.trip_id == trip_id).order_by(Itinerary.day_no)
            return list(s.scalars(q))

    @service_call
    def save_itinerary(self, trip_id: str, day_activities: list[str]) -> int:
        """Replace the trip's itinerary atomically (itinerary.save / .modify tools). Returns days saved."""
        with self._db.session() as s:
            s.execute(delete(Itinerary).where(Itinerary.trip_id == trip_id))
            s.add_all(Itinerary(id=new_id("IT"), trip_id=trip_id, day_no=n, activities=a)
                      for n, a in enumerate(day_activities, 1))
            s.execute(update(Trip).where(Trip.id == trip_id, Trip.status == "draft").values(status="planned"))
        return len(day_activities)
