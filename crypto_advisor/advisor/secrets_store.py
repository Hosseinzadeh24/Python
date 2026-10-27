"""API key storage. The key is never written to a project file and never printed in full.

Order of lookup: environment variable WALLEX_API_KEY, then the OS keyring
(Windows Credential Manager on Windows 11), where the GUI saves it.
"""
from __future__ import annotations

import os

SERVICE = "crypto_advisor_wallex"
USER = "api_key"
ENV = "WALLEX_API_KEY"


def _keyring():
    try:
        import keyring
        return keyring
    except Exception:
        return None


def load_key() -> str | None:
    env = os.environ.get(ENV, "").strip()
    if env:
        return env
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, USER) or None
    except Exception:
        return None


def save_key(key: str) -> None:
    kr = _keyring()
    if kr is None:
        raise RuntimeError("keyring is not installed: run  pip install keyring")
    kr.set_password(SERVICE, USER, key.strip())


def delete_key() -> None:
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(SERVICE, USER)
    except Exception:
        pass


def source() -> str:
    if os.environ.get(ENV, "").strip():
        return "environment variable"
    return "Windows Credential Manager" if load_key() else "not set"


def mask(key: str | None) -> str:
    if not key:
        return "(no key)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 6)}{key[-2:]}"
