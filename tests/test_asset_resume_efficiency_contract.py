from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.asset_resume_efficiency_contract import (
    _CachedImageRouter,
    install,
)
from minecraft_mod_ai.project_write_lock import project_write_lock


class _Router:
    def __init__(self) -> None:
        self.calls = 0

    def generate_image(
        self,
        role: str,
        *,
        prompt: str,
        output_path: str | Path,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Path:
        self.calls += 1
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}|{prompt}|{width}|{height}|{seed}".encode())
        return path


def test_exact_image_source_is_reused_after_retry(tmp_path) -> None:
    backend = _Router()
    router = _CachedImageRouter(backend)
    output = tmp_path / "tile.png"

    first = router.generate_image(
        "image_generator",
        prompt="same prompt",
        output_path=output,
        width=512,
        height=512,
        seed=123,
    )
    second = router.generate_image(
        "image_generator",
        prompt="same prompt",
        output_path=output,
        width=512,
        height=512,
        seed=123,
    )

    assert first == output.resolve()
    assert second == output.resolve()
    assert backend.calls == 1
    assert output.with_name(output.name + ".mmm-image-source.json").is_file()


def test_changed_prompt_or_seed_invalidates_image_source_cache(tmp_path) -> None:
    backend = _Router()
    router = _CachedImageRouter(backend)
    output = tmp_path / "tile.png"

    router.generate_image(
        "image_generator",
        prompt="one",
        output_path=output,
        width=512,
        height=512,
        seed=1,
    )
    router.generate_image(
        "image_generator",
        prompt="two",
        output_path=output,
        width=512,
        height=512,
        seed=1,
    )
    router.generate_image(
        "image_generator",
        prompt="two",
        output_path=output,
        width=512,
        height=512,
        seed=2,
    )

    assert backend.calls == 3


def test_corrupted_cached_source_is_regenerated(tmp_path) -> None:
    backend = _Router()
    router = _CachedImageRouter(backend)
    output = tmp_path / "tile.png"

    router.generate_image(
        "image_generator",
        prompt="stable",
        output_path=output,
        width=512,
        height=512,
        seed=7,
    )
    output.write_bytes(b"corrupted")
    router.generate_image(
        "image_generator",
        prompt="stable",
        output_path=output,
        width=512,
        height=512,
        seed=7,
    )

    assert backend.calls == 2


def _proposal(relative: str):
    return SimpleNamespace(
        game_design={
            "_asset_generation_plan": {
                "schema_version": "mmm/resource-asset-generation-plan-v2",
                "assets": [
                    {
                        "asset_id": "test_asset",
                        "container": "mod",
                        "textures": [{"target_path": relative}],
                        "documents": [],
                    }
                ],
            }
        }
    )


def _canonical_module(current):
    module = SimpleNamespace()

    def atomic_write(target: Path, data: bytes) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    module._atomic_write_bytes = atomic_write
    module.generate_assets = current
    return module


def test_expensive_asset_phase_does_not_hold_project_write_lock(tmp_path) -> None:
    import threading

    project_root = tmp_path / "project"
    project_root.mkdir()
    relative = "src/main/resources/assets/example/textures/item/test.png"
    target = project_root / relative
    lock_was_free = False
    module = None

    def current(router, proposal, project_root, run_root):
        nonlocal lock_was_free
        entered = threading.Event()

        def contender():
            with project_write_lock(project_root):
                entered.set()

        thread = threading.Thread(target=contender)
        thread.start()
        lock_was_free = entered.wait(timeout=1)
        thread.join(timeout=1)
        module._atomic_write_bytes(target, b"generated")
        return {"status": "TEXTURE_PRODUCTION_PASS"}

    module = _canonical_module(current)
    install(module)
    receipt = module.generate_assets(
        object(),
        _proposal(relative),
        project_root,
        tmp_path / "run",
    )

    assert lock_was_free is True
    assert target.read_bytes() == b"generated"
    assert receipt["status"] == "TEXTURE_PRODUCTION_PASS"
    assert module.generate_assets._mmm_path_scoped_asset_commit is True


def test_asset_commit_refuses_stale_overwrite(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    relative = "src/main/resources/assets/example/textures/item/test.png"
    target = project_root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"initial")
    module = None

    def current(router, proposal, project_root, run_root):
        with project_write_lock(project_root):
            target.write_bytes(b"concurrent-writer")
        module._atomic_write_bytes(target, b"generated")
        return {"status": "TEXTURE_PRODUCTION_PASS"}

    module = _canonical_module(current)
    install(module)

    with pytest.raises(RuntimeError, match="changed while generation was in flight"):
        module.generate_assets(
            object(),
            _proposal(relative),
            project_root,
            tmp_path / "run",
        )

    assert target.read_bytes() == b"concurrent-writer"


def test_unplanned_run_local_write_is_not_fenced_as_project_output(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    relative = "src/main/resources/assets/example/textures/item/test.png"
    run_local = tmp_path / "run" / "candidate.png"
    module = None

    def current(router, proposal, project_root, run_root):
        module._atomic_write_bytes(run_local, b"candidate")
        return {"status": "TEXTURE_PRODUCTION_PASS"}

    module = _canonical_module(current)
    install(module)
    module.generate_assets(
        object(),
        _proposal(relative),
        project_root,
        tmp_path / "run",
    )

    assert run_local.read_bytes() == b"candidate"
