from __future__ import annotations

import json
import time
import urllib.request
from functools import wraps
from typing import Any


def install(runtime_helper_module: Any) -> None:
    """Give runtime-helper metadata requests the same absolute transfer deadline.

    ``urlopen(timeout=...)`` only bounds individual socket stalls. A peer can keep a
    response alive indefinitely by sending a small amount of JSON before every socket
    timeout. Read in bounded chunks and enforce one monotonic total deadline so helper
    discovery cannot hold runtime preparation forever.
    """

    current = runtime_helper_module._json_request
    if getattr(current, "_mmm_absolute_json_deadline", False):
        return

    @wraps(current)
    def json_request(url: str) -> Any:
        total_timeout = min(
            60.0,
            float(runtime_helper_module._download_timeout_seconds()),
        )
        deadline = time.monotonic() + total_timeout
        request = urllib.request.Request(url, headers=runtime_helper_module._headers())
        chunks: list[bytes] = []
        with urllib.request.urlopen(
            request,
            timeout=min(30.0, total_timeout),
        ) as response:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Runtime helper metadata request exceeded {total_timeout:g}s: {url}"
                    )
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Runtime helper metadata request exceeded {total_timeout:g}s: {url}"
                    )
        return json.loads(b"".join(chunks).decode("utf-8"))

    json_request._mmm_absolute_json_deadline = True  # type: ignore[attr-defined]
    json_request.__wrapped__ = current  # type: ignore[attr-defined]
    runtime_helper_module._json_request = json_request


__all__ = ["install"]
