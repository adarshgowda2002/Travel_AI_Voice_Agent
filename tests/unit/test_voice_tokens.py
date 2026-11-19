from livekit import api

from voice_travel.config import Settings
from voice_travel.voice.tokens import create_join_token, room_name


def test_token_is_scoped_to_one_room_and_identity():
    s = Settings(_env_file=None)
    token = create_join_token(s, call_id="S1", identity="U0001")
    claims = api.TokenVerifier(s.livekit_api_key, s.livekit_api_secret.get_secret_value()).verify(token)
    assert claims.identity == "U0001"
    assert claims.video.room == room_name("S1") == "call_S1"
    assert claims.video.room_join and claims.video.can_publish and claims.video.can_subscribe
    assert not claims.video.room_admin


def test_livekit_secret_is_not_in_repr():
    s = Settings(_env_file=None)
    assert "devsecret" not in repr(s)
