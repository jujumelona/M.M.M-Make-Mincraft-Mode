from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.asset_resume_efficiency_contract import _CachedImageRouter


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
        path.write_bytes(f"{role}|{prompt}|{width}|{height}|{seed}".encode("utf-8"))
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
