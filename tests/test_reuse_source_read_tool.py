from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from minecraft_mod_ai.production_tools import ProductionToolService
from minecraft_mod_ai.spec import SpecValidationError


def _fixture(tmp_path: Path):
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    donor = project / ".minecraft_ai" / "reuse" / "donors" / ("a" * 20)
    source = donor / "src/main/java/example/Trade.java"
    source.parent.mkdir(parents=True)
    raw = b"package example;\nfinal class Trade { static int price = 7; }\n"
    source.write_bytes(raw)
    manifest = {
        "repository": "example/trade",
        "commit_sha": "b" * 40,
        "license_id": "MIT",
        "capability": "trade.transaction",
        "files": [{"path": str(source), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}],
    }
    (donor / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ProductionToolService(workspace_root=workspace), project, source, raw


def test_read_reuse_source_is_manifest_authorized_and_paginated(tmp_path):
    service, project, source, raw = _fixture(tmp_path)
    first = service.read_reuse_source(str(project), str(source), limit_bytes=12)
    assert first["schema_version"] == "mmm/reuse-source-read-v1"
    assert first["content"] == raw[:12].decode()
    assert first["eof"] is False
    second = service.read_reuse_source(str(project), str(source), offset_bytes=first["next_offset_bytes"], limit_bytes=32768)
    assert second["eof"] is True
    assert second["content"] == raw[12:].decode()


def test_read_reuse_source_rejects_unmanifested_file(tmp_path):
    service, project, source, _raw = _fixture(tmp_path)
    other = source.with_name("Other.java")
    other.write_text("class Other {}", encoding="utf-8")
    with pytest.raises(SpecValidationError, match="not authorized"):
        service.read_reuse_source(str(project), str(other))
