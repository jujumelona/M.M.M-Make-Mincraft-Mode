from __future__ import annotations

import threading
import time
from pathlib import Path

from minecraft_mod_ai.extended_content_generator import _serialized_extended_content
from minecraft_mod_ai.project_write_lock import project_write_lock


def test_extended_content_lock_is_scoped_per_project(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    barrier = threading.Barrier(2)
    entered: list[str] = []
    guard = threading.Lock()

    @_serialized_extended_content
    def work(*, project_root: Path, label: str) -> None:
        with guard:
            entered.append(label)
        barrier.wait(timeout=2)

    threads = [
        threading.Thread(target=work, kwargs={"project_root": left, "label": "left"}),
        threading.Thread(target=work, kwargs={"project_root": right, "label": "right"}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(entered) == ["left", "right"]


def test_extended_content_lock_serializes_same_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    guard = threading.Lock()
    active = 0
    max_active = 0

    @_serialized_extended_content
    def work(*, project_root: Path) -> None:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1

    threads = [
        threading.Thread(target=work, kwargs={"project_root": root}),
        threading.Thread(target=work, kwargs={"project_root": root}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active == 1


def test_extended_content_project_lock_is_reentrant(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    @_serialized_extended_content
    def work(*, project_root: Path) -> bool:
        with project_write_lock(project_root):
            return True

    assert work(project_root=root) is True
