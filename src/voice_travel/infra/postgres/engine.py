"""Database access object. Created once at the composition root and INJECTED into services,
so there is no global connection and tests can pass an in-memory database."""
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        if url.startswith("sqlite"):
            # in-memory SQLite (tests) needs a single shared connection
            self.engine: Engine = create_engine(
                url, echo=echo, connect_args={"check_same_thread": False}, poolclass=StaticPool)
        else:
            self.engine = create_engine(url, echo=echo, pool_pre_ping=True, pool_size=10, max_overflow=10)
        self._sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[OrmSession]:
        """One transaction per unit of work: commit on success, roll back on any error.
        A write touching several tables is therefore atomic."""
        s = self._sessionmaker()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def dispose(self) -> None:
        self.engine.dispose()
