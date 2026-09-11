from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.write_text(content, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_resource_contracts() -> None:
    path = "minecraft_mod_ai/resource_contracts.py"
    text = _read(path)
    text = _replace_once(
        text,
        "from pathlib import Path, PurePosixPath",
        "from pathlib import PurePosixPath",
        label="resource_contracts pathlib import",
    )
    _write(path, text)


RESUME_CONTRACT = r'''from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any

_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _backend_identity(router: Any, role: str) -> str:
    config = router.registry.role(router.profile, role)
    payload = _jsonable(config)
    if isinstance(payload, dict):
        payload.pop("api_key", None)
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _key(
    *,
    role: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    backend_identity: str,
) -> str:
    body = json.dumps(
        {
            "version": _VERSION,
            "role": role,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "backend_identity": backend_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _meta_path(output: Path) -> Path:
    return output.with_name(output.name + ".mmm-image-source.json")


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _valid_cached(output: Path, *, cache_key: str) -> bool:
    meta = _meta_path(output)
    if (
        not output.is_file()
        or output.is_symlink()
        or not meta.is_file()
        or meta.is_symlink()
    ):
        return False
    try:
        value = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(value, dict):
        return False
    if value.get("version") != _VERSION or value.get("key") != cache_key:
        return False
    expected = str(value.get("sha256", ""))
    return bool(expected) and _sha256(output) == expected


class _CachedImageRouter:
    "Content-address expensive deterministic image candidates across retries."

    def __init__(self, router: Any) -> None:
        self._router = router
        self._backend_identities: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

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
        output = Path(output_path).expanduser().resolve()
        if role not in self._backend_identities:
            self._backend_identities[role] = _backend_identity(self._router, role)
        identity = self._backend_identities[role]
        cache_key = _key(
            role=role,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            backend_identity=identity,
        )
        if _valid_cached(output, cache_key=cache_key):
            return output

        generated = Path(
            self._router.generate_image(
                role,
                prompt=prompt,
                output_path=output,
                width=width,
                height=height,
                seed=seed,
            )
        ).expanduser().resolve()
        if generated != output or not output.is_file() or output.is_symlink():
            raise RuntimeError(
                "Image backend did not create the exact requested resumable output path."
            )
        _write_meta(
            _meta_path(output),
            {
                "version": _VERSION,
                "key": cache_key,
                "sha256": _sha256(output),
                "width": int(width),
                "height": int(height),
                "seed": int(seed),
                "backend_identity": identity,
            },
        )
        return output


def install(resource_asset_module: Any) -> None:
    "Bind retry-safe image source caching to the single canonical asset producer."

    current = resource_asset_module.generate_assets
    if getattr(current, "_mmm_resumable_image_sources", False):
        return

    @wraps(current)
    def resumable(
        router: Any,
        *args: Any,
        __current=current,
        **kwargs: Any,
    ):
        return __current(_CachedImageRouter(router), *args, **kwargs)

    resumable._mmm_resumable_image_sources = True  # type: ignore[attr-defined]
    setattr(resource_asset_module, "generate_assets", resumable)


__all__ = ["_CachedImageRouter", "install"]
'''


def patch_resume_contract() -> None:
    _write("minecraft_mod_ai/asset_resume_efficiency_contract.py", RESUME_CONTRACT)


def patch_runtime_bootstrap() -> None:
    path = "minecraft_mod_ai/runtime_bootstrap.py"
    text = _read(path)
    text = _replace_once(
        text,
        "from . import agentic_optimization_contract, complete_orchestrator_services",
        "from . import agentic_optimization_contract, resource_asset_production",
        label="runtime_bootstrap planner import",
    )
    text = _replace_once(
        text,
        "install_asset_resume_efficiency(complete_orchestrator_services)",
        "install_asset_resume_efficiency(resource_asset_production)",
        label="runtime_bootstrap asset resume owner",
    )
    _write(path, text)


ATOMIC_HELPER = r'''
def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

'''


def patch_resource_asset_production() -> None:
    path = "minecraft_mod_ai/resource_asset_production.py"
    text = _read(path)
    text = _replace_once(
        text,
        "import hashlib\nimport json\nimport re\nimport zipfile\n",
        "import hashlib\nimport json\nimport os\nimport re\nimport tempfile\nimport zipfile\n",
        label="resource_asset_production imports",
    )
    generate_anchor = (
        "\ndef generate_assets("
        "router: Any, proposal: CompleteProposal, project_root: Path, run_root: Path"
        ") -> dict[str, Any]:\n"
    )
    text = _replace_once(
        text,
        generate_anchor,
        "\n" + ATOMIC_HELPER + generate_anchor.lstrip("\n"),
        label="resource_asset_production atomic helper",
    )
    text = _replace_once(
        text,
        '            target.parent.mkdir(parents=True, exist_ok=True)\n'
        '            target.write_text(encoded, encoding="utf-8")',
        '            _atomic_write_bytes(target, encoded.encode("utf-8"))',
        label="resource document atomic write",
    )
    text = _replace_once(
        text,
        '        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")',
        '        _atomic_write_bytes(\n'
        '            metadata_path,\n'
        '            (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\\n").encode("utf-8"),\n'
        '        )',
        label="standalone pack metadata atomic write",
    )
    text = _replace_once(
        text,
        "                target.parent.mkdir(parents=True, exist_ok=True)\n"
        "                target.write_bytes(winner.read_bytes())",
        "                _atomic_write_bytes(target, winner.read_bytes())",
        label="texture atomic write",
    )
    _write(path, text)


EXTRA_TEST = r'''

def test_resume_cache_binds_only_to_canonical_asset_producer() -> None:
    from minecraft_mod_ai import (
        complete_orchestrator_services,
        resource_asset_production,
    )

    assert getattr(
        resource_asset_production.generate_assets,
        "_mmm_resumable_image_sources",
        False,
    )
    assert not hasattr(
        complete_orchestrator_services,
        "_generate_single_asset_source",
    )
    assert not hasattr(
        complete_orchestrator_services,
        "_generate_tiled_asset_source",
    )
'''


def patch_tests() -> None:
    path = "tests/test_resource_architecture.py"
    text = _read(path)
    if "test_resume_cache_binds_only_to_canonical_asset_producer" in text:
        raise RuntimeError("resource architecture resume integration test already exists")
    _write(path, text.rstrip() + EXTRA_TEST + "\n")


def main() -> None:
    patch_resource_contracts()
    patch_resume_contract()
    patch_runtime_bootstrap()
    patch_resource_asset_production()
    patch_tests()


if __name__ == "__main__":
    main()
