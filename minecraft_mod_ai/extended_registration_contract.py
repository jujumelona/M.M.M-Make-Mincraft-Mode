from __future__ import annotations

import copy
import json
import sys
import threading
from collections import OrderedDict
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .project_write_lock import project_write_lock


_TEXTURE_CACHE_LOCK = threading.RLock()
_TEXTURE_CACHE: OrderedDict[tuple[str, str, int, int], bytes] = OrderedDict()
_TEXTURE_CACHE_LIMIT = 512
_RECORD_CACHE_LOCK = threading.RLock()
_RECORD_CACHE: OrderedDict[
    str,
    dict[str, tuple[tuple[int, int, int], dict[str, Any]]],
] = OrderedDict()
_RECORD_CACHE_LIMIT = 16


def _render_static_registration(root_name: str | None) -> str:
    lines = [
        '    private static List<Block> registerGeneratedUnits() {',
        '        // GeneratedContentUnit registrations use the bounded static tree; Files.list(directory) is not executed.',
        '        List<Block> machineBlocks = new ArrayList<>();',
    ]
    if root_name:
        lines.extend(
            [
                f'        {root_name}.register();',
                f'        machineBlocks.addAll({root_name}.machineBlocks());',
            ]
        )
    lines.extend(
        [
            '        return machineBlocks;',
            '    }',
        ]
    )
    return "\n".join(lines)


def _replace_registration_method(source: str, root_name: str | None) -> str:
    start_tokens = (
        '    @SuppressWarnings("unchecked")\n    private static List<Block> registerGeneratedUnits() {',
        '    private static List<Block> registerGeneratedUnits() {',
    )
    start = -1
    for token in start_tokens:
        start = source.find(token)
        if start >= 0:
            break
    if start < 0:
        raise RuntimeError("GeneratedExtendedContent registration method is missing.")
    end_marker = "\n\n    public record MachineDefinition("
    end = source.find(end_marker, start)
    if end < 0:
        raise RuntimeError("GeneratedExtendedContent registration method boundary is missing.")
    return source[:start] + _render_static_registration(root_name) + source[end:]


def _generated_unit_names(root: Path, package_name: str) -> list[str]:
    """Read registrar identity from generated Java names, not every JSON record."""

    directory = (
        root
        / "src/main/java"
        / Path(*package_name.split("."))
        / "extended"
    )
    if not directory.is_dir() or directory.is_symlink():
        return []
    names: list[str] = []
    for path in directory.glob("GeneratedContentUnit*.java"):
        if path.is_file() and not path.is_symlink():
            names.append(path.stem)
    names.sort()
    return names


def _texture_pattern_key(color: str, seed: str, kind: str, size: int) -> tuple[str, str, int, int]:
    # make_texture_png uses seed_value only through modulo 2 (checker) and modulo 7
    # (highlight). LCM(2, 7)=14, so every seed with the same residue produces the
    # exact same raw pixels and therefore the exact same deterministic PNG bytes.
    seed_residue = sum((index + 1) * ord(char) for index, char in enumerate(seed)) % 14
    return str(color), str(kind), int(size), seed_residue


def _install_texture_equivalence_cache(extended_module: Any) -> None:
    """Reuse byte-identical deterministic textures instead of recompressing them."""

    from . import generator as generator_module

    original = generator_module.make_texture_png
    if getattr(original, "_mmm_texture_equivalence_cache", False):
        if getattr(extended_module, "make_texture_png", None) is not original:
            extended_module.make_texture_png = original
        return

    @wraps(original)
    def make_texture_png(
        color: str,
        seed: str,
        *,
        kind: str,
        size: int = 16,
    ) -> bytes:
        key = _texture_pattern_key(color, seed, kind, size)
        with _TEXTURE_CACHE_LOCK:
            cached = _TEXTURE_CACHE.get(key)
            if cached is not None:
                _TEXTURE_CACHE.move_to_end(key)
                return cached

        rendered = original(color, seed, kind=kind, size=size)
        with _TEXTURE_CACHE_LOCK:
            cached = _TEXTURE_CACHE.get(key)
            if cached is not None:
                _TEXTURE_CACHE.move_to_end(key)
                return cached
            _TEXTURE_CACHE[key] = rendered
            while len(_TEXTURE_CACHE) > _TEXTURE_CACHE_LIMIT:
                _TEXTURE_CACHE.popitem(last=False)
        return rendered

    make_texture_png._mmm_texture_equivalence_cache = True  # type: ignore[attr-defined]
    make_texture_png._mmm_texture_cache = _TEXTURE_CACHE  # type: ignore[attr-defined]
    make_texture_png.__wrapped__ = original  # type: ignore[attr-defined]
    generator_module.make_texture_png = make_texture_png

    # Retarget only package-local aliases that imported the exact original function.
    for module_name, loaded in tuple(sys.modules.items()):
        if not (
            module_name == "minecraft_mod_ai"
            or module_name.startswith("minecraft_mod_ai.")
        ):
            continue
        if loaded is None:
            continue
        try:
            namespace = vars(loaded)
        except TypeError:
            continue
        if namespace.get("make_texture_png") is original:
            namespace["make_texture_png"] = make_texture_png

    extended_module.make_texture_png = make_texture_png


def _install_extended_record_cache(extended_module: Any) -> None:
    """Reuse parsed directory-catalog records only while their exact file metadata is unchanged."""

    original = extended_module.iter_extended_module_records
    if getattr(original, "_mmm_validated_record_cache", False):
        return

    @wraps(original)
    def iter_extended_module_records(project_root: str | Path) -> Iterator[dict[str, Any]]:
        root = Path(project_root).expanduser().resolve()
        catalog = root / ".minecraft_ai/extended-modules.json"
        if not catalog.is_file() or catalog.is_symlink():
            yield from original(project_root)
            return
        try:
            header = json.loads(catalog.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            yield from original(project_root)
            return
        if (
            not isinstance(header, dict)
            or header.get("schema_version") != extended_module._DIRECTORY_CATALOG_SCHEMA
        ):
            yield from original(project_root)
            return

        relative = header.get("directory")
        expected = header.get("module_count")
        if not isinstance(relative, str) or type(expected) is not int:
            raise extended_module.ExtendedContentError(
                "Extended module directory catalog is invalid."
            )
        normalized = PurePosixPath(relative.replace("\\", "/"))
        if (
            normalized.is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            raise extended_module.ExtendedContentError(
                "Extended module directory path is unsafe."
            )
        directory = (root / Path(*normalized.parts)).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise extended_module.ExtendedContentError(
                "Extended module directory escaped the project."
            ) from exc
        if not directory.is_dir() or directory.is_symlink():
            raise extended_module.ExtendedContentError(
                "Extended module directory is missing or unsafe."
            )

        directory_key = str(directory)
        with _RECORD_CACHE_LOCK:
            cached_records = _RECORD_CACHE.get(directory_key, {})
        refreshed: dict[str, tuple[tuple[int, int, int], dict[str, Any]]] = {}
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            if not path.is_file() or path.is_symlink():
                raise extended_module.ExtendedContentError(
                    "Extended module record is unsafe."
                )
            stat = path.stat()
            signature = (
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
            path_key = str(path)
            cached = cached_records.get(path_key)
            if cached is not None and cached[0] == signature:
                item = cached[1]
            else:
                item = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(item, dict)
                    or not item.get("module_id")
                    or path.stem != str(item["module_id"])
                ):
                    raise extended_module.ExtendedContentError(
                        "Extended module record is invalid."
                    )
            refreshed[path_key] = (signature, item)
            records.append(copy.deepcopy(item))

        if len(records) != expected:
            raise extended_module.ExtendedContentError(
                "Extended module directory count does not match."
            )
        with _RECORD_CACHE_LOCK:
            _RECORD_CACHE[directory_key] = refreshed
            _RECORD_CACHE.move_to_end(directory_key)
            while len(_RECORD_CACHE) > _RECORD_CACHE_LIMIT:
                _RECORD_CACHE.popitem(last=False)
        yield from records

    iter_extended_module_records._mmm_validated_record_cache = True  # type: ignore[attr-defined]
    iter_extended_module_records.__wrapped__ = original  # type: ignore[attr-defined]
    extended_module.iter_extended_module_records = iter_extended_module_records


def _install_static_registration(extended_module: Any) -> None:
    original = extended_module.generate_extended_content
    if getattr(original, "_mmm_static_registrar_tree", False):
        return

    @wraps(original)
    def generated_with_static_registrar(*args: Any, **kwargs: Any) -> dict[str, Any]:
        project_root = kwargs.get("project_root")
        mod_id = kwargs.get("mod_id")
        package_name = kwargs.get("package_name")
        if project_root is None and args:
            project_root = args[0]
        if mod_id is None and len(args) > 1:
            mod_id = args[1]
        if package_name is None and len(args) > 2:
            package_name = args[2]
        if not project_root or not mod_id or not package_name:
            raise RuntimeError("Static registrar binding requires project_root/mod_id/package_name.")

        root = Path(project_root).expanduser().resolve()
        # The base generator owns its own project transaction boundary. Do not hold a
        # second outer project lock across its validation/materialization work. Only
        # the shared registrar read/merge/commit below needs this wrapper's lock.
        receipt = original(*args, **kwargs)
        if not isinstance(receipt, dict) or receipt.get("status") != "GENERATED":
            return receipt

        with project_write_lock(root):
            # The base generator has already materialized one deterministic
            # GeneratedContentUnit*.java file for every Java-backed module.
            # Reopening and JSON-decoding the complete bounded record directory
            # here was a second O(N) I/O pass. Filenames are the exact registrar
            # class identities and are sufficient for the compile-time tree.
            leaf_names = _generated_unit_names(root, str(package_name))
            dispatch_root, dispatch_files = extended_module._registrar_tree_files(
                str(package_name),
                leaf_names,
                fanout=32,
            )

            info = extended_module.inspect_fabric_project(root)
            dispatch_receipt = None
            if dispatch_files:
                dispatch_receipt = extended_module.write_text_files(
                    info,
                    dispatch_files,
                    replace_existing=True,
                )

            root_path = (
                root
                / "src/main/java"
                / Path(*str(package_name).split("."))
                / "extended/GeneratedExtendedContent.java"
            )
            before = root_path.read_bytes()
            try:
                before_text = before.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError("GeneratedExtendedContent.java is not UTF-8.") from exc
            after = _replace_registration_method(before_text, dispatch_root)
            if after != before_text:
                from .source_patch import TransactionalSourcePatcher, sha256_bytes

                registration_receipt = TransactionalSourcePatcher(root).apply(
                    [
                        {
                            "operation": "replace",
                            "path": root_path.relative_to(root).as_posix(),
                            "expected_sha256": sha256_bytes(before),
                            "content": after,
                        }
                    ]
                )
            else:
                registration_receipt = {"status": "UNCHANGED"}

            dispatch_paths = [str(root / relative) for relative in sorted(dispatch_files)]
            result = dict(receipt)
            result["registrar_dispatch_count"] = len(dispatch_files)
            result["static_registration_unit_count"] = len(leaf_names)
            result["static_registration_root"] = dispatch_root
            result["static_registration_receipt"] = registration_receipt
            result["files"] = list(
                dict.fromkeys([*receipt.get("files", []), *dispatch_paths])
            )
            result["touched_paths"] = list(
                dict.fromkeys([*receipt.get("touched_paths", []), *dispatch_paths])
            )
            if dispatch_receipt is not None:
                result["registrar_dispatch_receipt"] = dispatch_receipt
            return result

    generated_with_static_registrar._mmm_static_registrar_tree = True
    generated_with_static_registrar._mmm_project_scoped_serialization = True
    extended_module.generate_extended_content = generated_with_static_registrar


def install(extended_module: Any) -> None:
    """Install deterministic registration and bounded exact generation reuse."""

    _install_texture_equivalence_cache(extended_module)
    _install_extended_record_cache(extended_module)
    _install_static_registration(extended_module)