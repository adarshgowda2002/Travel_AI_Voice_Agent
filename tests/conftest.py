import os

import pytest

from voice_travel.config import get_settings
from voice_travel.infra.postgres import Database
from voice_travel.seeding.postgres import load_all
from voice_travel.services import BookingService, IdentityService, OpsLogService, SessionService, TripService

# Set TEST_DATABASE_URL to a SCRATCH Postgres database to run the integration suite against real Postgres.
# Warning: the suite drops and recreates the app tables.
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite+pysqlite://")


@pytest.fixture(scope="session")
def db():
    database = Database(TEST_DATABASE_URL)
    load_all(database, get_settings().dataset_dir, reset=True)
    yield database
    database.dispose()


@pytest.fixture
def identity(db):
    return IdentityService(db)


@pytest.fixture
def trips(db):
    return TripService(db)


@pytest.fixture
def bookings(db):
    return BookingService(db)


@pytest.fixture
def sessions(db):
    return SessionService(db)


@pytest.fixture
def ops(db):
    return OpsLogService(db)
