import functools
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("voice_travel.services")

P = ParamSpec("P")
R = TypeVar("R")


class ServiceError(Exception):
    """The data layer failed (DB down, constraint violated). 'Not found' is NOT an error: services return
    None so the agent can answer naturally instead of treating a missing record as an outage."""


def service_call(fn: Callable[P, R]) -> Callable[P, R]:
    """Uniform logging + translation of database errors into ServiceError for every service method."""
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            result = fn(*args, **kwargs)
            logger.debug("%s ok", fn.__qualname__)
            return result
        except ServiceError:
            raise
        except SQLAlchemyError as exc:
            logger.exception("%s failed", fn.__qualname__)
            raise ServiceError(f"{fn.__qualname__} failed: {type(exc).__name__}") from exc
    return wrapper
