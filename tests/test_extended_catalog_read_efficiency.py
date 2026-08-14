from __future__ import annotations

import json
from pathlib import Path

import pytest

import minecraft_mod_ai.extended_content_generator as extended
import minecraft_mod_ai.extended_registration_contract as contract


def _write_directory_catalog(root: Path) -> Path:
    directory = root / ".minecraft_ai/extended-module-records"
    directory.mkdir(parents=True)
    records = {
        "alpha": {
            "module_id": "alpha",
            "kind": "item",
            "config": {"value": 1},
        },
        "beta": {
            "module_id": "beta",
            "kind": "block",
            "config": {"value": 2},
        },
    }
    for module_id, record in records.items():
        (directory / f"{module_id}.json").write_text(
            json.dumps(record, sort_keys=True),
            encoding="utf-8",
        )
    (root / ".minecraft_ai/extended-modules.json").write_text(
        json.dumps(
            {
                "schema_version": "mmm/extended-module-directory-v1",
                "module_count": 2,
                "directory": ".minecraft_ai/extended-module-records",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return directory


def test_unchanged_directory_records_are_not_reopened(monkeypatch, tmp_path: Path) -> None:
    directory = _write_directory_catalog(tmp_path)
    with contract._RECORD_CACHE_LOCK:
        contract._RECORD_CACHE.clear()

    first = list(extended.iter_extended_module_records(tmp_path))
    assert [record["module_id"] for record in first] == ["alpha", "beta"]

    original_read_text = Path.read_text
    record_reads = 0

    def counted_read_text(path: Path, *args, **kwargs):
        nonlocal record_reads
        if path.parent == directory:
            record_reads += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    second = list(extended.iter_extended_module_records(tmp_path))

    assert second == first
    assert record_reads == 0


def test_only_changed_record_is_reparsed(monkeypatch, tmp_path: Path) -> None:
    directory = _write_directory_catalog(tmp_path)
    with contract._RECORD_CACHE_LOCK:
        contract._RECORD_CACHE.clear()
    list(extended.iter_extended_module_records(tmp_path))

    alpha = directory / "alpha.json"
    record = json.loads(alpha.read_text(encoding="utf-8"))
    record["config"]["value"] = 99
    alpha.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    original_read_text = Path.read_text
    reopened: list[str] = []

    def counted_read_text(path: Path, *args, **kwargs):
        if path.parent == directory:
            reopened.append(path.name)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    records = list(extended.iter_extended_module_records(tmp_path))

    assert reopened == ["alpha.json"]
    assert records[0]["config"]["value"] == 99


def test_changed_record_identity_mismatch_still_fails_closed(tmp_path: Path) -> None:
    directory = _write_directory_catalog(tmp_path)
    with contract._RECORD_CACHE_LOCK:
        contract._RECORD_CACHE.clear()
    list(extended.iter_extended_module_records(tmp_path))

    alpha = directory / "alpha.json"
    record = json.loads(alpha.read_text(encoding="utf-8"))
    record["module_id"] = "forged"
    alpha.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    with pytest.raises(extended.ExtendedContentError, match="record is invalid"):
        list(extended.iter_extended_module_records(tmp_path))
