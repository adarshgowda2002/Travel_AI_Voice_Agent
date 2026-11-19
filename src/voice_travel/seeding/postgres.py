"""Load data/out/*.csv into the database through the ORM tables.

    seed-postgres            # create missing tables, load (fails on duplicates rather than doubling data)
    seed-postgres --reset    # DROP and recreate the app tables first (destructive)

Schema is owned by Alembic in real environments (`alembic upgrade head`); --reset/create_all here
is for local dev and tests only.
"""
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, Numeric, Table, func, insert, select
from sqlalchemy.types import JSON, TypeEngine

from voice_travel.config import get_settings
from voice_travel.infra.postgres import (
    AgentTask,
    AudioLog,
    AuditLog,
    Base,
    Booking,
    Conversation,
    Database,
    Itinerary,
    Session,
    Trip,
    User,
)
from voice_travel.logging import configure_logging
from voice_travel.seeding.csv_io import read_csv
from voice_travel.utils import phone_key

BATCH = 2000
# Foreign-key-safe load order
PLAN: list[tuple[type[Base], str]] = [
    (User, "users.csv"), (Trip, "trips.csv"), (Session, "sessions.csv"), (Conversation, "conversations.csv"),
    (Booking, "bookings.csv"), (Itinerary, "itineraries.csv"), (AgentTask, "agent_tasks.csv"),
    (AudioLog, "audio_logs.csv"), (AuditLog, "audit_logs.csv"),
]


def _convert(col_type: TypeEngine, value: str) -> Any:
    """CSV text -> Python value according to the column's SQL type. Empty string -> NULL."""
    if value == "":
        return None
    if isinstance(col_type, JSON):
        return json.loads(value)
    if isinstance(col_type, Boolean):
        return value in ("1", "true", "True")
    if isinstance(col_type, DateTime):
        return dt.datetime.fromisoformat(value)
    if isinstance(col_type, Date):
        return dt.date.fromisoformat(value)
    if isinstance(col_type, Integer):
        return int(value)
    if isinstance(col_type, Numeric | Float):
        return float(value)
    return value


def load_all(db: Database, dataset_dir: Path, reset: bool = False) -> dict[str, int]:
    """Create tables and load every CSV in ONE transaction (all tables or none). Returns row counts."""
    if reset:
        Base.metadata.drop_all(db.engine)
    Base.metadata.create_all(db.engine)
    counts: dict[str, int] = {}
    with db.engine.begin() as conn:
        for model, filename in PLAN:
            table = cast(Table, model.__table__)
            types = {c.name: c.type for c in table.columns}
            rows = []
            for raw in read_csv(dataset_dir, filename):
                row = {k: _convert(types[k], v) for k, v in raw.items()}
                if model is User:                       # bulk insert bypasses ORM events
                    row["phone_key"] = phone_key(row.get("phone"))
                rows.append(row)
            for i in range(0, len(rows), BATCH):
                conn.execute(insert(table), rows[i:i + BATCH])
            counts[table.name] = conn.scalar(select(func.count()).select_from(table)) or 0
    return counts


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    db = Database(settings.database_url)
    db.ping()
    for name, n in load_all(db, settings.dataset_dir, reset="--reset" in sys.argv).items():
        print(f"  {name:<16}{n:>7}")


if __name__ == "__main__":
    main()
