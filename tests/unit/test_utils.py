import datetime as dt

import pytest

from voice_travel.utils import new_id, phone_key, utcnow


@pytest.mark.parametrize("raw", ["+91 90000 00001", "9000000001", "09000000001", "+919000000001", "(900) 000-0001"])
def test_phone_key_is_format_independent(raw):
    assert phone_key(raw) == "9000000001"


@pytest.mark.parametrize("raw", ["", None, "12345", "abc"])
def test_phone_key_rejects_short_or_missing(raw):
    assert phone_key(raw) == ""


def test_new_id_is_prefixed_and_unique():
    ids = {new_id("B") for _ in range(1000)}
    assert len(ids) == 1000
    assert all(i.startswith("B") for i in ids)


def test_utcnow_is_naive():
    assert isinstance(utcnow(), dt.datetime)
    assert utcnow().tzinfo is None
