from __future__ import annotations

from typing import Any


APP_NAME = "ML Local App"
APP_VERSION = "0.1.8"
APP_RELEASE_CHANNEL = "prototype"
APP_RELEASE_DATE = "2026-04-14"


def get_app_metadata() -> dict[str, Any]:
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "release_channel": APP_RELEASE_CHANNEL,
        "release_date": APP_RELEASE_DATE,
    }


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "APP_RELEASE_CHANNEL",
    "APP_RELEASE_DATE",
    "get_app_metadata",
]
