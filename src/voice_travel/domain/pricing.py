"""Business rules for money. Pure functions: no I/O. The ONLY place these constants live
(they mirror the policy docs served by RAG: 'change_fee' and 'refund')."""

CHANGE_FEE_INR = 2500
REFUND_RATE = 0.9
_ROUND_TO = 50


def refund_amount(paid: int) -> int:
    """Refund for a cancelled booking, rounded to the nearest 50 INR."""
    return int(round(paid * REFUND_RATE / _ROUND_TO) * _ROUND_TO)
