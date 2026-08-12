from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai import minecraft_mcp_runtime_helper_contract as runtime_helpers


def _jar_bytes(mod_id: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("fabric.mod.json", json.dumps({"id": mod_id}))
    return stream.getvalue()


def test_download_timeout_configuration_must_be_finite_positive(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RUNTIME_HELPER_DOWNLOAD_TIMEOUT_SECONDS", "0")
    with pytest.raises(RuntimeError, match="finite and positive"):
        runtime_helpers._download_timeout_seconds()


def test_corrupt_fabric_api_cache_is_redownloaded(monkeypatch, tmp_path: Path) -> None:
    version = "0.0-test"
    target = tmp_path / f"fabric-api-{version}.jar"
    target.write_bytes(b"corrupt")
    good = _jar_bytes("fabric-api")
    calls: list[str] = []

    def fake_download(url: str, path: Path) -> None:
        calls.append(url)
        path.write_bytes(good)

    monkeypatch.setattr(runtime_helpers, "_download", fake_download)
    receipt = runtime_helpers._fabric_api_artifact(tmp_path, version)

    assert len(calls) == 1
    assert target.read_bytes() == good
    assert receipt["status"] == "STAGED"
