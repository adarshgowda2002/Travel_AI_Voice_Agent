from pydantic import SecretStr

from voice_travel.config import Settings


def test_redis_url_is_secret_and_not_in_repr():
    s = Settings(redis_url="redis://default:topsecret@host:1234")
    assert isinstance(s.redis_url, SecretStr)
    assert "topsecret" not in repr(s)
    assert "topsecret" not in str(s)


def test_defaults_point_at_local_dev_stack():
    s = Settings(_env_file=None)
    assert s.database_url.startswith("postgresql+psycopg://")
    assert s.redis_url is None


def test_production_refuses_development_secrets_and_open_api():
    import pytest
    with pytest.raises(ValueError, match="development defaults") as exc:
        Settings(_env_file=None, environment="production")
    assert "VOICE_API_KEY" in str(exc.value)


def test_production_starts_with_real_secrets():
    s = Settings(_env_file=None, environment="production", livekit_api_key="prodkey",
                 livekit_api_secret="x" * 40, voice_api_key="k" * 32)
    assert s.environment == "production"
