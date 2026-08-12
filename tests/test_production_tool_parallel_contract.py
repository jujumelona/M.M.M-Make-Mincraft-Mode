from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.production_tool_parallel_contract import install


class _FakeService:
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def index_project_rag(
        self,
        roots,
        *,
        index_path="rag/project-index.json",
        metadata,
        semantic=False,
    ):
        del roots, metadata, semantic
        target = self._resolve(index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.active_lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            time.sleep(0.08)
            target.write_text("built", encoding="utf-8")
            return {"index_path": str(target)}
        finally:
            with self.active_lock:
                type(self).active -= 1


# Install once at module import so either test also passes when collected or run in
# isolation. The installer is idempotent and marks the patched method.
install(SimpleNamespace(ProductionToolService=_FakeService))


def test_same_rag_target_has_one_builder_and_second_rechecks_exists(
    tmp_path: Path,
) -> None:
    _FakeService.active = 0
    _FakeService.max_active = 0
    service = _FakeService(tmp_path)
    barrier = threading.Barrier(3)
    outcomes = []

    def worker() -> None:
        barrier.wait()
        try:
            outcomes.append(
                service.index_project_rag(
                    ["src"],
                    metadata={"target": "test"},
                )
            )
        except BaseException as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert _FakeService.max_active == 1
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert sum(isinstance(item, FileExistsError) for item in outcomes) == 1


def test_different_rag_targets_can_still_build_in_parallel(tmp_path: Path) -> None:
    # Pick two canonical targets that map to different lock stripes so the test
    # proves the contract serializes conflicts rather than all RAG work globally.
    from minecraft_mod_ai.production_tool_parallel_contract import _index_lock

    root = tmp_path.resolve()
    first = root / "a.json"
    second = None
    first_lock = _index_lock(first)
    for index in range(1, 256):
        candidate = root / f"b-{index}.json"
        if _index_lock(candidate) is not first_lock:
            second = candidate
            break
    assert second is not None

    _FakeService.active = 0
    _FakeService.max_active = 0
    service = _FakeService(root)
    barrier = threading.Barrier(3)

    def worker(path: Path) -> None:
        barrier.wait()
        service.index_project_rag(
            ["src"],
            index_path=path.name,
            metadata={"target": "test"},
        )

    threads = [
        threading.Thread(target=worker, args=(first,)),
        threading.Thread(target=worker, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert _FakeService.max_active == 2
