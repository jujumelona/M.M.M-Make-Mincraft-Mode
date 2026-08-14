from __future__ import annotations

import threading
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from .project_write_lock import project_write_lock


_EXTENDED_PROJECT_ROOT: ContextVar[Path | None] = ContextVar(
    "mmm_extended_project_root",
    default=None,
)


class _ProjectScopedExtendedLock:
    """Preserve one-writer semantics without serializing unrelated projects."""

    _mmm_project_scoped_extended_lock = True

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._local = threading.local()

    def __enter__(self):
        root = _EXTENDED_PROJECT_ROOT.get()
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        if root is None:
            self._fallback.acquire()
            stack.append(("fallback", self._fallback))
        else:
            manager = project_write_lock(root)
            manager.__enter__()
            stack.append(("project", manager))
        return self

    def __exit__(self, exc_type, exc, tb):
        stack = getattr(self._local, "stack", None)
        if not stack:
            raise RuntimeError("Extended-content lock exit without matching enter.")
        kind, value = stack.pop()
        if kind == "fallback":
            value.release()
            return False
        return value.__exit__(exc_type, exc, tb)


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


def _install_static_registration(extended_module: Any) -> None:
    original_lock = extended_module._EXTENDED_CONTENT_LOCK
    if not getattr(original_lock, "_mmm_project_scoped_extended_lock", False):
        extended_module._EXTENDED_CONTENT_LOCK = _ProjectScopedExtendedLock(original_lock)

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
        token = _EXTENDED_PROJECT_ROOT.set(root)
        try:
            # Hold the same per-project re-entrant lock across catalog generation and
            # static-tree binding. The wrapped base generator re-enters this lock, so
            # same-project callers remain atomic while unrelated roots can proceed.
            with extended_module._EXTENDED_CONTENT_LOCK:
                receipt = original(*args, **kwargs)
                if not isinstance(receipt, dict) or receipt.get("status") != "GENERATED":
                    return receipt

                records = [
                    item
                    for item in extended_module.iter_extended_module_records(root)
                    if str(item.get("kind", "")) in extended_module._JAVA_KINDS
                ]
                leaf_names = [
                    extended_module._unit_class_name(str(item["module_id"]))
                    for item in records
                ]
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
                before = root_path.read_text(encoding="utf-8")
                after = _replace_registration_method(before, dispatch_root)
                if after != before:
                    from .source_patch import TransactionalSourcePatcher, sha256_file

                    registration_receipt = TransactionalSourcePatcher(root).apply(
                        [
                            {
                                "operation": "replace",
                                "path": root_path.relative_to(root).as_posix(),
                                "expected_sha256": sha256_file(root_path),
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
        finally:
            _EXTENDED_PROJECT_ROOT.reset(token)

    generated_with_static_registrar._mmm_static_registrar_tree = True
    generated_with_static_registrar._mmm_project_scoped_serialization = True
    extended_module.generate_extended_content = generated_with_static_registrar


def install(extended_module: Any) -> None:
    """Replace runtime classpath reflection with a bounded compile-time registrar tree."""

    _install_static_registration(extended_module)
