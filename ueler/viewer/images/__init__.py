from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterator

_ASSET_ROOT = Path(__file__).resolve().parent


def get_asset_path(name: str) -> str:
    """Return the filesystem path to the packaged viewer asset."""
    asset = _ASSET_ROOT / name
    if not asset.exists():
        raise FileNotFoundError(f"Viewer asset '{name}' not found under {_ASSET_ROOT}.")
    return str(asset)


def iter_asset_bytes(name: str, chunk_size: int = 8192) -> Iterator[bytes]:
    """Yield bytes for the requested asset in chunks."""
    with open_asset(name, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def load_asset_bytes(name: str) -> bytes:
    """Load the requested asset into memory and return its bytes."""
    with open_asset(name, "rb") as handle:
        return handle.read()


def open_asset(name: str, mode: str = "rb") -> BinaryIO:
    """Open a packaged viewer asset and return the file handle."""
    if "b" not in mode:
        raise ValueError("Viewer assets should be accessed in binary mode.")
    asset = _ASSET_ROOT / name
    if not asset.exists():
        raise FileNotFoundError(f"Viewer asset '{name}' not found under {_ASSET_ROOT}.")
    return asset.open(mode)


__all__ = [
    "get_asset_path",
    "iter_asset_bytes",
    "load_asset_bytes",
    "open_asset",
]
