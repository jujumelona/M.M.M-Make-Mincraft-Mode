from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path

from minecraft_mod_ai.extended_content_generator import generate_extended_content
from minecraft_mod_ai.project_write_lock import project_write_lock


def test_project_write_lock_is_scoped_per_project(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    barrier = threading.Barrier(2)
    entered: list[str] = []
    guard = threading.Lock()

    def work(project_root: Path, label: str) -> None:
        with project_write_lock(project_root):
            with guard:
                entered.append(label)
            barrier.wait(timeout=2)

    threads = [
        threading.Thread(target=work, args=(left, "left")),
        threading.Thread(target=work, args=(right, "right")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(entered) == ["left", "right"]


def test_project_write_lock_serializes_same_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    guard = threading.Lock()
    active = 0
    max_active = 0

    def work() -> None:
        nonlocal active, max_active
        with project_write_lock(root):
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1

    threads = [threading.Thread(target=work), threading.Thread(target=work)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active == 1


def test_extended_content_uses_short_commit_fencing_not_function_wrapper() -> None:
    source = inspect.getsource(generate_extended_content)
    assert "_serialized_extended_content" not in source
    assert "with project_write_lock(info.root):" in source


def test_project_write_lock_is_reentrant(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with project_write_lock(root):
        with project_write_lock(root):
            assert True
