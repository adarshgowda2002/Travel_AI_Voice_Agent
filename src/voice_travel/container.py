"""Composition root: the one place that wires concrete implementations together.
Everything else receives its dependencies through constructors."""
from dataclasses import dataclass

import redis

from voice_travel.config import Settings, get_settings
from voice_travel.infra.postgres import Database
from voice_travel.infra.redis import create_redis
from voice_travel.services import BookingService, IdentityService, OpsLogService, SessionService, TripService


@dataclass(frozen=True)
class Container:
    settings: Settings
    db: Database
    redis: redis.Redis | None
    identity: IdentityService
    trips: TripService
    bookings: BookingService
    sessions: SessionService
    ops: OpsLogService


def build_container(settings: Settings | None = None, *, connect_redis: bool = True) -> Container:
    settings = settings or get_settings()
    db = Database(settings.database_url)
    client = create_redis(settings.redis_url) if connect_redis and settings.redis_url else None
    return Container(settings=settings, db=db, redis=client, identity=IdentityService(db),
                     trips=TripService(db), bookings=BookingService(db), sessions=SessionService(db),
                     ops=OpsLogService(db))
