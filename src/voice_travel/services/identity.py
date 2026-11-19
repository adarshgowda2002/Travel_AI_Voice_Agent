"""Caller identification and authentication for the voice session."""
from sqlalchemy import func, select

from voice_travel.infra.postgres import Database, User
from voice_travel.services.errors import service_call
from voice_travel.utils import phone_key


class IdentityService:
    def __init__(self, db: Database) -> None:
        self._db = db

    @service_call
    def fetch_user_by_phone(self, phone: str) -> User | None:
        key = phone_key(phone)
        if not key:
            return None
        with self._db.session() as s:
            return s.scalars(select(User).where(User.phone_key == key).limit(1)).first()

    @service_call
    def fetch_user_by_name(self, name: str) -> User | None:
        with self._db.session() as s:
            return s.scalars(select(User).where(func.lower(User.name) == name.strip().lower()).limit(1)).first()

    def verify_identity(self, name: str, phone: str) -> bool:
        """Both name (case-insensitive) and phone must match the same user."""
        user = self.fetch_user_by_phone(phone)
        return bool(user and name and user.name.strip().lower() == name.strip().lower())
