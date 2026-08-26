from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.asset_resume_efficiency_contract as asset_contract
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


def _services(current):
    return SimpleNamespace(
        _generate_single_asset_source=current,
        _generate_tiled_asset_source=current,
    )


def _asset_call(service, project_root: Path, concept_dir: Path):
    target = project_root / "src/main/resources/assets/example/textures/item/test.png"
    request = SimpleNamespace(
        asset_id="test_asset",
        target_path="src/main/resources/assets/example/textures/item/test.png",
    )
    return service(
        object(),
        request=request,
        concept_dir=concept_dir,
        target=target,
    ), target


def test_expensive_asset_phase_does_not_hold_project_write_lock(tmp_path) -> None:
    project_root = tmp_path / "project"
    concept_dir = tmp_path / "run" / "asset-concepts"
    project_root.mkdir()
    lock_was_free = False

    def current(router, *, request, concept_dir, target):
        nonlocal lock_was_free
        import threading

        entered = threading.Event()

        def contender():
            with project_write_lock(project_root):
                entered.set()

        thread = threading.Thread(target=contender)
        thread.start()
        lock_was_free = entered.wait(timeout=1)
        thread.join(timeout=1)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"generated")
        return {"status": "GENERATED"}

    services = _services(current)
    install(services)
    receipt, target = _asset_call(
        services._generate_single_asset_source,
        project_root,
        concept_dir,
    )

    assert lock_was_free is True
    assert target.read_bytes() == b"generated"
    assert receipt["asset_commit_mode"] == "staged_atomic_replace"


def test_asset_commit_refuses_stale_overwrite(tmp_path) -> None:
    project_root = tmp_path / "project"
    concept_dir = tmp_path / "run" / "asset-concepts"
    target = project_root / "src/main/resources/assets/example/textures/item/test.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"initial")

    def current(router, *, request, concept_dir, target: Path):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"generated")
        final_target = project_root / request.target_path
        with project_write_lock(project_root):
            final_target.write_bytes(b"concurrent-writer")
        return {"status": "GENERATED"}

    services = _services(current)
    install(services)

    with pytest.raises(RuntimeError, match="changed while generation was in flight"):
        _asset_call(
            services._generate_single_asset_source,
            project_root,
            concept_dir,
        )

    assert target.read_bytes() == b"concurrent-writer"


def test_same_filesystem_atomic_commit_does_not_copy_staged_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    staged = tmp_path / "run" / "asset-concepts" / ".final-staging" / "asset.png"
    target = project_root / "src/main/resources/assets/example/textures/item/asset.png"
    staged.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    payload = b"final-image" * 4096
    staged.write_bytes(payload)

    def forbidden_temp_copy(*_args, **_kwargs):
        raise AssertionError("same-filesystem commit must not allocate a copy temp file")

    monkeypatch.setattr(asset_contract.tempfile, "mkstemp", forbidden_temp_copy)

    digest = asset_contract._atomic_commit(staged, target, project_root)

    assert not staged.exists()
    assert target.read_bytes() == payload
    assert digest == asset_contract._sha256(target)
