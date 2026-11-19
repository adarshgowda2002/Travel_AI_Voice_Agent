"""Fetch model files that are not shipped in the repo (verified by SHA-256). `fetch-voice-models`."""
import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

from voice_travel.config import get_settings
from voice_travel.logging import configure_logging

logger = logging.getLogger(__name__)

# Silero VAD v5.1.2 (MIT). Pinned tag + hash so a moved or changed upstream file is rejected.
SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"


# Piper voice en_US-lessac-medium, pinned to the rhasspy/piper-voices v1.0.0 tag.
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/"
PIPER_FILES = {
    "en_US-lessac-medium.onnx": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    "en_US-lessac-medium.onnx.json": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, dest: Path, sha256: str) -> Path:
    """Download url to dest unless dest already matches sha256. The temp file is only moved into place
    after the hash matches, so a truncated or tampered download never becomes the model."""
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
          urllib.request.urlopen(url, timeout=120) as resp):   # noqa: S310 (fixed https URLs above)
        shutil.copyfileobj(resp, tmp)
    tmp_path = Path(tmp.name)
    if _sha256(tmp_path) != sha256:
        tmp_path.unlink()
        raise RuntimeError(f"{dest.name} download failed its SHA-256 check")
    tmp_path.replace(dest)
    logger.info("Fetched %s", dest)
    return dest


def fetch_silero(dest: Path) -> Path:
    return _fetch(SILERO_URL, dest, SILERO_SHA256)


def fetch_piper_voice(onnx_path: Path) -> Path:
    """The voice is two files side by side: NAME.onnx and NAME.onnx.json."""
    for name, sha in PIPER_FILES.items():
        _fetch(PIPER_BASE + name, onnx_path.parent / name, sha)
    return onnx_path


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    fetch_silero(settings.vad_model_path)
    fetch_piper_voice(settings.tts_voice_path)
