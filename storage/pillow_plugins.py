"""Helpers to register Pillow plugins (e.g., AVIF/HEIF support)."""

from __future__ import annotations

import logging
from threading import Lock

_REGISTER_LOCK = Lock()
_REGISTERED = False


def ensure_heif_support() -> None:
    """
    Register Pillow plugins that enable HEIF/AVIF decoding.

    Calling this multiple times is safe; the registration happens only once.
    """

    global _REGISTERED
    if _REGISTERED:
        return

    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        try:
            import pillow_heif  # type: ignore[import-not-found]
        except ImportError:
            logging.warning(
                "pillow_heif is not installed – HEIF/AVIF images cannot be decoded."
            )
            return

        try:
            pillow_heif.register_heif_opener()  # type: ignore[attr-defined]
        except AttributeError:
            logging.debug("pillow_heif.register_heif_opener not available")
        try:
            pillow_heif.register_avif_opener()  # type: ignore[attr-defined]
        except AttributeError:
            logging.debug("pillow_heif.register_avif_opener not available")

        _REGISTERED = True
