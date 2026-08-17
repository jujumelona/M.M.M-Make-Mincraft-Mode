from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import rag_index


def _metadata(source_commit: str) -> dict[str, str]:
    return {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": "17",
        "license": "project-local",
        "source_commit": source_commit,
    }


def test_unchanged_rag_build_does_not_rewrite_relations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "A.java").write_text("public final class A {}\n", encoding="utf-8")
    index = rag_index.ProjectRAGIndex(tmp_path / "project-index.sqlite")

    index.build([root], metadata=_metadata("sha256:first"), semantic=False)

    calls = 0

    def unexpected_relation_write(*_args, **_kwargs) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(rag_index, "_insert_relations", unexpected_relation_write)
    result = index.build(
        [root],
        metadata=_metadata("sha256:second"),
        semantic=False,
    )

    assert result["incremental"] is True
    assert result["changed_files"] == 0
    assert result["removed_files"] == 0
    assert calls == 0
