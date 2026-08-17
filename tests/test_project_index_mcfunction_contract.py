from __future__ import annotations

import hashlib
from pathlib import Path

from minecraft_mod_ai.project_index import ProjectIndex


def test_mcfunction_is_indexed_with_host_owned_sha_and_exact_source(tmp_path: Path) -> None:
    function = tmp_path / "src/main/resources/data/example/functions/start.mcfunction"
    function.parent.mkdir(parents=True)
    content = "say ready\nfunction example:next\n"
    function.write_text(content, encoding="utf-8")

    index = ProjectIndex(tmp_path)
    relative = function.relative_to(tmp_path).as_posix()

    indexed = index._by_path[relative]
    assert indexed.suffix == ".mcfunction"
    assert indexed.sha256 == "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    selected = index.select(
        query="example start function",
        diagnostic_paths=(relative,),
        byte_budget=4096,
    )
    record = next(item for item in selected["files"] if item["path"] == relative)
    assert record["sha256"] == indexed.sha256
    assert record["content"] == content


def test_incremental_mcfunction_update_refreshes_snapshot_sha(tmp_path: Path) -> None:
    function = tmp_path / "src/main/resources/data/example/functions/start.mcfunction"
    function.parent.mkdir(parents=True)
    function.write_text("say one\n", encoding="utf-8")
    index = ProjectIndex(tmp_path)
    relative = function.relative_to(tmp_path).as_posix()
    before = index._by_path[relative].sha256

    function.write_text("say two\n", encoding="utf-8")
    index.update_files((function,))

    after = index._by_path[relative].sha256
    assert after != before
    assert after == "sha256:" + hashlib.sha256(b"say two\n").hexdigest()
