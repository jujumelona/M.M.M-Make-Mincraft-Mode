from __future__ import annotations

import json
import time
import urllib.request
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO


def _remaining(deadline: float, *, url: str, label: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError(f"{label} exceeded its absolute deadline: {url}")
    return value


def _set_socket_timeout(response: Any, seconds: float) -> None:
    """Tighten the current CPython HTTP socket to the remaining absolute deadline."""

    candidates = [
        getattr(getattr(response, "fp", None), "raw", None),
        getattr(response, "fp", None),
    ]
    for candidate in candidates:
        sock = getattr(candidate, "_sock", None)
        if sock is None:
            continue
        setter = getattr(sock, "settimeout", None)
        if callable(setter):
            setter(max(0.001, seconds))
            return


def _read_one(response: Any, size: int) -> bytes:
    # HTTPResponse.read1 performs at most one underlying buffered/raw read instead of
    # trying to fill ``size`` with an unbounded sequence of successful trickle reads.
    read1 = getattr(response, "read1", None)
    if callable(read1):
        return read1(size)
    return response.read(size)


def _iter_deadlined_chunks(
    response: Any,
    *,
    deadline: float,
    per_read_timeout: float,
    url: str,
    label: str,
    chunk_size: int,
):
    while True:
        remaining = _remaining(deadline, url=url, label=label)
        _set_socket_timeout(response, min(per_read_timeout, remaining))
        chunk = _read_one(response, chunk_size)
        if not chunk:
            return
        yield chunk
        _remaining(deadline, url=url, label=label)


def install(runtime_helper_module: Any) -> None:
    """Bound both helper binaries and metadata by one monotonic transfer deadline.

    Socket timeouts alone only bound one stalled recv. ``read()`` may internally
    perform many successful recv calls while a peer trickles bytes forever. The
    installed paths use ``read1`` and re-tighten the socket timeout to the remaining
    host deadline before every read, so neither binary nor JSON helper discovery can
    be kept alive indefinitely by slow progress.
    """

    current_download = runtime_helper_module._download
    if not getattr(current_download, "_mmm_absolute_transfer_deadline", False):

        @wraps(current_download)
        def download(url: str, target: Path) -> None:
            total_timeout = float(runtime_helper_module._download_timeout_seconds())
            deadline = time.monotonic() + total_timeout
            request = urllib.request.Request(
                url,
                headers=runtime_helper_module._headers(),
            )
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.unlink(missing_ok=True)
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=min(60.0, total_timeout),
                ) as response, temporary.open("wb") as handle:
                    for chunk in _iter_deadlined_chunks(
                        response,
                        deadline=deadline,
                        per_read_timeout=60.0,
                        url=url,
                        label="Runtime helper download",
                        chunk_size=1024 * 1024,
                    ):
                        handle.write(chunk)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

        download._mmm_absolute_transfer_deadline = True  # type: ignore[attr-defined]
        download.__wrapped__ = current_download  # type: ignore[attr-defined]
        runtime_helper_module._download = download

    current_json = runtime_helper_module._json_request
    if getattr(current_json, "_mmm_absolute_json_deadline", False):
        return

    @wraps(current_json)
    def json_request(url: str) -> Any:
        total_timeout = min(
            60.0,
            float(runtime_helper_module._download_timeout_seconds()),
        )
        deadline = time.monotonic() + total_timeout
        request = urllib.request.Request(
            url,
            headers=runtime_helper_module._headers(),
        )
        chunks: list[bytes] = []
        with urllib.request.urlopen(
            request,
            timeout=min(30.0, total_timeout),
        ) as response:
            chunks.extend(
                _iter_deadlined_chunks(
                    response,
                    deadline=deadline,
                    per_read_timeout=30.0,
                    url=url,
                    label="Runtime helper metadata request",
                    chunk_size=256 * 1024,
                )
            )
        return json.loads(b"".join(chunks).decode("utf-8"))

    json_request._mmm_absolute_json_deadline = True  # type: ignore[attr-defined]
    json_request.__wrapped__ = current_json  # type: ignore[attr-defined]
    runtime_helper_module._json_request = json_request


__all__ = ["install"]
