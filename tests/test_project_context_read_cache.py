from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.project_index import ProjectIndex


def _count_binary_source_opens(monkeypatch, target: Path):
    original_open = Path.open
    calls = {"count": 0}

    def counting_open(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == target and "r" in mode and "b" in mode:
            calls["count"] += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    return calls


def test_repeated_select_reuses_unchanged_source_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "src/main/java/example/Target.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example; final class Target { int targetSymbol = 1; }\n",
        encoding="utf-8",
    )
    index = ProjectIndex(tmp_path)
    calls = _count_binary_source_opens(monkeypatch, source)

    first = index.select(query="targetSymbol", byte_budget=4096)
    second = index.select(query="targetSymbol", byte_budget=4096)

    assert first == second
    assert calls["count"] == 1


def test_update_files_invalidates_cached_source_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/main/java/example/Target.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example; final class Target { int targetSymbol = 1; }\n",
        encoding="utf-8",
    )
    index = ProjectIndex(tmp_path)
    first = index.select(query="targetSymbol", byte_budget=4096)

    source.write_text(
        "package example; final class Target { int targetSymbol = 2222; }\n",
        encoding="utf-8",
    )
    index.update_files(["src/main/java/example/Target.java"])
    second = index.select(query="targetSymbol", byte_budget=4096)

    assert "targetSymbol = 1" in first["files"][0]["content"]
    assert "targetSymbol = 2222" in second["files"][0]["content"]


def test_paginated_context_reads_large_source_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "src/main/java/example/Large.java"
    source.parent.mkdir(parents=True)
    original = (
        "package example;\n"
        + "// paginationTarget exact grounded evidence\n" * 420
        + "final class Large {}\n"
    )
    source.write_text(original, encoding="utf-8")
    index = ProjectIndex(tmp_path)
    calls = _count_binary_source_opens(monkeypatch, source)

    cursor = ""
    fragments: list[str] = []
    pages = 0
    while True:
        page = index.select_page(
            query="paginationTarget",
            byte_budget=1400,
            cursor=cursor,
        )
        pages += 1
        fragments.extend(item["content"] for item in page["files"])
        cursor = page["next_cursor"]
        if not cursor:
            break

    assert pages > 1
    assert "".join(fragments) == original
    assert calls["count"] == 1


def test_paginated_context_cache_does_not_hide_stale_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/main/java/example/Large.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example;\n" + "// staleTarget\n" * 420,
        encoding="utf-8",
    )
    index = ProjectIndex(tmp_path)
    first = index.select_page(query="staleTarget", byte_budget=1400)
    assert first["next_cursor"]

    source.write_text(
        "package example;\n" + "// staleTarget changed and longer\n" * 420,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Project source changed after its context index was built",
    ):
        index.select_page(
            query="staleTarget",
            byte_budget=1400,
            cursor=first["next_cursor"],
        )


def test_cached_source_identity_detects_same_size_edit_with_restored_mtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/main/java/example/SameSize.java"
    source.parent.mkdir(parents=True)
    original = "package example; final class SameSize { int value = 1; }\n"
    changed = "package example; final class SameSize { int value = 2; }\n"
    assert len(original.encode("utf-8")) == len(changed.encode("utf-8"))
    source.write_text(original, encoding="utf-8")

    index = ProjectIndex(tmp_path)
    first = index.select_page(query="SameSize", byte_budget=4096)
    before = source.stat()

    source.write_text(changed, encoding="utf-8")
    import os
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(
        ValueError,
        match="Project source changed after its context index was built",
    ):
        index.select_page(
            query="SameSize",
            byte_budget=4096,
            cursor=first["next_cursor"] or "",
        )
