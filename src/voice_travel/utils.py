import datetime as dt
import re
import uuid

PHONE_KEY_LEN = 10


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:10].upper()}"


def utcnow() -> dt.datetime:
    """Naive UTC timestamp (columns are 'timestamp without time zone')."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def phone_key(phone: str | None) -> str:
    """Canonical lookup key: the last 10 digits, so '+91 90000 00001', '9000000001' and
    '09000000001' all map to the same value. Returns '' when there are fewer than 10 digits."""
    digits = re.sub(r"\D", "", phone or "")
    return digits[-PHONE_KEY_LEN:] if len(digits) >= PHONE_KEY_LEN else ""
