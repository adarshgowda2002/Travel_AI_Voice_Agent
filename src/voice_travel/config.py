"""Single source of configuration. The ONLY module that reads the environment / .env file."""
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_LIVEKIT_KEY = "devkey"
DEV_LIVEKIT_SECRET = "devsecret-devsecret-devsecret-0001"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # "production" refuses to start with development secrets or an open voice API (see _check_production).
    environment: Literal["development", "production"] = "development"
    database_url: str = "postgresql+psycopg://voice:voice@localhost:5432/voice_travel"
    redis_url: SecretStr | None = None          # SecretStr: never shown in repr/logs
    # LiveKit. Defaults match livekit.yaml / docker-compose (local dev ONLY): override both in .env elsewhere.
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = DEV_LIVEKIT_KEY
    livekit_api_secret: SecretStr = SecretStr(DEV_LIVEKIT_SECRET)
    # Bearer key required on the session endpoints. Unset = open (development only).
    voice_api_key: SecretStr | None = None
    voice_token_ttl_sec: int = 3600
    voice_join_timeout_sec: int = 120            # transport-level: give up if the caller never publishes audio
    vad_model_path: Path = PROJECT_ROOT / "models" / "silero_vad.onnx"     # fetch-voice-models
    stt_model: str = "small.en"                  # faster-whisper model name or local path
    stt_device: str = "auto"
    stt_compute_type: str = "int8"
    stt_language: str = "en"
    stt_partials: bool = True                    # live partial transcripts (extra STT work while the caller talks)
    stt_partial_interval_ms: int = 1000
    tts_voice_path: Path = PROJECT_ROOT / "models" / "en_US-lessac-medium.onnx"     # Piper voice
    log_level: str = "INFO"
    dataset_dir: Path = PROJECT_ROOT / "data" / "out"

    @model_validator(mode="after")
    def _check_production(self) -> Self:
        if self.environment != "production":
            return self
        problems = []
        if (self.livekit_api_key == DEV_LIVEKIT_KEY
                or self.livekit_api_secret.get_secret_value() == DEV_LIVEKIT_SECRET):
            problems.append("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are the development defaults")
        if self.voice_api_key is None or not self.voice_api_key.get_secret_value():
            problems.append("VOICE_API_KEY is not set (the session API would be open to anyone)")
        if problems:
            raise ValueError("refusing to start in production: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
