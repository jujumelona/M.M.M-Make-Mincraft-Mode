from __future__ import annotations

import os
from pathlib import Path

import pytest

from minecraft_mod_ai import research_validation_fingerprint_performance as perf
from minecraft_mod_ai import validation_execution_contract as validation


def _project(root: Path) -> tuple[Path, Path]:
    (root / "src/main/java/demo").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True)
    (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    (root / "settings.gradle").write_text("rootProject.name='demo'\n", encoding="utf-8")
    java = root / "src/main/java/demo/Main.java"
    resource = root / "src/main/resources/data.json"
    java.write_text("package demo; public final class Main {}\n", encoding="utf-8")
    resource.write_text('{"value":1}\n', encoding="utf-8")
    return java, resource


def test_validation_fingerprint_runtime_is_installed() -> None:
    marker = "_mmm_stat_validated_validation_fingerprint_v1"
    assert getattr(validation.project_build_fingerprint, marker, False)
    assert getattr(validation._java_fingerprint, marker, False)


def test_build_fingerprint_reuses_unchanged_file_digests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    java, resource = _project(tmp_path)
    perf.clear_digest_cache()
    calls: list[Path] = []
    original = perf._hash_file_uncached

    def tracked(path: Path) -> bytes:
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(perf, "_hash_file_uncached", tracked)
    first = validation.project_build_fingerprint(tmp_path)
    first_count = len(calls)
    assert first_count >= 4

    second = validation.project_build_fingerprint(tmp_path)
    assert second == first
    assert len(calls) == first_count

    resource.write_text('{"value":2}\n', encoding="utf-8")
    third = validation.project_build_fingerprint(tmp_path)
    assert third != second
    assert len(calls) == first_count + 1
    assert calls[-1] == resource

    # Java is still unchanged and remains reusable after a different input changed.
    assert java not in calls[first_count:]


@pytest.mark.skipif(os.name == "nt", reason="Windows NTFS ctime is creation time, not metadata change time")
def test_same_size_content_change_with_restored_mtime_is_not_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _java, resource = _project(tmp_path)
    perf.clear_digest_cache()
    calls: list[Path] = []
    original = perf._hash_file_uncached

    def tracked(path: Path) -> bytes:
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(perf, "_hash_file_uncached", tracked)
    before_stat = resource.stat()
    first = validation.project_build_fingerprint(tmp_path)
    first_count = len(calls)

    # Preserve size and restore mtime deliberately. ctime/inode metadata is part of
    # the cache key, so a rewritten file is still re-read instead of trusting mtime.
    resource.write_text('{"value":9}\n', encoding="utf-8")
    os.utime(resource, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    second = validation.project_build_fingerprint(tmp_path)

    assert second != first
    assert len(calls) == first_count + 1
    assert calls[-1] == resource


def test_java_fingerprint_reuses_configs_and_sources_independently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    java, _resource = _project(tmp_path)
    perf.clear_digest_cache()
    calls: list[Path] = []
    original = perf._hash_file_uncached

    def tracked(path: Path) -> bytes:
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(perf, "_hash_file_uncached", tracked)
    first, files = validation._java_fingerprint(tmp_path, None)
    first_count = len(calls)
    assert files == ("src/main/java/demo/Main.java",)
    assert first_count >= 3

    second, files_again = validation._java_fingerprint(tmp_path, None)
    assert second == first
    assert files_again == files
    assert len(calls) == first_count

    java.write_text("package demo; public final class Main { int x; }\n", encoding="utf-8")
    third, _ = validation._java_fingerprint(tmp_path, None)
    assert third != second
    assert len(calls) == first_count + 1
    assert calls[-1] == java
