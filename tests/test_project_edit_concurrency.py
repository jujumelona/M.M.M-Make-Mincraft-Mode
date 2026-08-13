from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from minecraft_mod_ai import project_edit
from minecraft_mod_ai.project_edit import FabricProjectInfo, write_text_files


def _info(root: Path) -> FabricProjectInfo:
    return FabricProjectInfo(
        root=root,
        mod_id="parallel_test",
        main_entrypoint="example.Main",
        package_name="example",
        main_class="Main",
        main_java=root / "src/main/java/example/Main.java",
        fabric_mod_json=root / "src/main/resources/fabric.mod.json",
        main_entrypoints=("example.Main",),
    )


def test_write_text_files_serializes_shared_file_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Parallel system shards may share generated support files without create races."""

    root = tmp_path / "project"
    root.mkdir()
    info = _info(root)
    shared = "src/main/java/example/system/MmmPersistentStore.java"
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    active = 0
    max_active = 0
    original_apply = project_edit.TransactionalSourcePatcher.apply

    def tracked_apply(self, operations):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            # Keep the transaction open long enough that the second worker would
            # overlap if write_text_files did not own the project lock while planning.
            time.sleep(0.05)
            return original_apply(self, operations)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(
        project_edit.TransactionalSourcePatcher,
        "apply",
        tracked_apply,
    )

    def worker(name: str) -> dict[str, object]:
        barrier.wait(timeout=2)
        return write_text_files(
            info,
            {
                shared: "final class MmmPersistentStore {}\n",
                f"src/main/java/example/system/{name}.java": (
                    f"final class {name} {{}}\n"
                ),
            },
            replace_existing=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(worker, "SystemA")
        right = pool.submit(worker, "SystemB")
        assert left.result(timeout=5)["status"] in {"APPLIED", "UNCHANGED"}
        assert right.result(timeout=5)["status"] in {"APPLIED", "UNCHANGED"}

    assert max_active == 1
    assert (root / shared).read_text(encoding="utf-8") == (
        "final class MmmPersistentStore {}\n"
    )
    assert (root / "src/main/java/example/system/SystemA.java").is_file()
    assert (root / "src/main/java/example/system/SystemB.java").is_file()
