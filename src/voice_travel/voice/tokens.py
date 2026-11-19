"""LiveKit access tokens: who may join which room. Signed with the API secret, never sent to the worker."""
import datetime as dt

from livekit import api

from voice_travel.config import Settings


def room_name(call_id: str) -> str:
    return f"call_{call_id}"


def create_join_token(settings: Settings, *, call_id: str, identity: str) -> str:
    grants = api.VideoGrants(room_join=True, room=room_name(call_id), can_publish=True, can_subscribe=True,
                             can_publish_data=True)
    return (api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret.get_secret_value())
            .with_identity(identity)
            .with_ttl(dt.timedelta(seconds=settings.voice_token_ttl_sec))
            .with_grants(grants)
            .to_jwt())
