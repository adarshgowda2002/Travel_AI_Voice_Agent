import pytest

from voice_travel.domain.pricing import CHANGE_FEE_INR, refund_amount


def test_change_fee_matches_policy_doc():
    assert CHANGE_FEE_INR == 2500


@pytest.mark.parametrize(("paid", "expected"), [(22500, 20250), (20000, 18000), (0, 0), (123456, 111100)])
def test_refund_is_90_percent_rounded_to_50(paid, expected):
    assert refund_amount(paid) == expected
