"""Verified WorkNode Java producers must survive the graph-to-coder boundary."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.integrity_dispatcher import _generator_source_context
from minecraft_mod_ai.request_runtime_records import verified_completed_work_java_paths


def _source(root, relative, contents):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return hashlib.sha256(contents.encode()).hexdigest()


def test_completed_dependency_is_visible_without_an_extra_template_port(tmp_path):
    relative = "src/main/java/demo/Producer.java"
    sha = _source(tmp_path, relative, (
        "package demo; public class Producer { "
        "public static void signal() {} }"))
    other = "src/main/java/demo/Unrelated.java"
    _source(tmp_path, other, "package demo; public class Unrelated {}")
    context = {"_verified_prerequisite_source_hashes": {relative: sha}}
    spec = {"language": "java", "target_path": "src/main/java/demo/Consumer.java",
            "bindings": {}}
    job = SimpleNamespace(requires=(), canonical_leaf="minecraft/event/callback")
    rows = _generator_source_context(job, spec=spec, context=context,
                                     port_registry=None, base_dir=tmp_path)
    assert [r["path"] for r in rows] == [relative]
    assert rows[0]["sha256"] == sha
    assert "signal" in rows[0]["source"]
    assert other not in str(rows)


def test_stale_dependency_receipt_never_exposes_replaced_java_source(tmp_path):
    relative = "src/main/java/demo/Producer.java"
    sha = _source(tmp_path, relative, "package demo; public class Producer {}")
    _source(tmp_path, relative, "package demo; public class Replacement {}")
    with pytest.raises(ValueError, match="WORK_SOURCE_RECEIPT_HASH_MISMATCH"):
        verified_completed_work_java_paths(
            {"_verified_prerequisite_source_hashes": {relative: sha}}, tmp_path)


def test_unverified_or_current_output_cannot_become_a_dependency(tmp_path):
    target = "src/main/java/demo/Consumer.java"
    sha = _source(tmp_path, target, "package demo; public class Consumer {}")
    rows = verified_completed_work_java_paths(
        {"_verified_prerequisite_source_hashes": {target: sha}}, tmp_path,
        target_path=target)
    assert rows == ()


def test_symlinked_materialized_source_is_not_accepted(tmp_path):
    real = "src/main/java/demo/Real.java"
    sha = _source(tmp_path, real, "package demo; public class Real {}")
    link = "src/main/java/demo/Link.java"
    (tmp_path / link).symlink_to(tmp_path / real)
    with pytest.raises(ValueError, match="WORK_SOURCE_RECEIPT_SYMLINK"):
        verified_completed_work_java_paths(
            {"_verified_prerequisite_source_hashes": {link: sha}}, tmp_path)
