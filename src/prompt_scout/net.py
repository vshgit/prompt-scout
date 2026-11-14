"""HTTP downloads with certifi's CA bundle.

Python from python.org on macOS does not use the system keychain, so plain urllib
fails with CERTIFICATE_VERIFY_FAILED until "Install Certificates.command" is run.
Using certifi makes downloads work everywhere without that step.
"""

from __future__ import annotations

import ssl
import urllib.request
from functools import lru_cache

import certifi


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def get(url: str, timeout: int = 60, max_bytes: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "prompt-scout"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
        return resp.read(max_bytes) if max_bytes else resp.read()
