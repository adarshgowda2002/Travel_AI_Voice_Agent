import logging
from urllib.parse import urlsplit

import redis
from pydantic import SecretStr

logger = logging.getLogger(__name__)


def safe_url(url: str) -> str:
    """host:port only. The URL contains a password: never log it."""
    p = urlsplit(url)
    return f"{p.scheme}://{p.hostname}:{p.port}"


def create_redis(url: SecretStr | str) -> redis.Redis:
    """Build a client and verify it. decode_responses so values are str, not bytes."""
    raw = url.get_secret_value() if isinstance(url, SecretStr) else url
    try:
        client = redis.Redis.from_url(raw, decode_responses=True, socket_connect_timeout=5,
                                      socket_timeout=5, health_check_interval=30)
        client.ping()
    except redis.exceptions.RedisError:
        logger.error("Redis connection failed (%s)", safe_url(raw))
        raise
    logger.info("Connected to Redis %s", safe_url(raw))
    return client
