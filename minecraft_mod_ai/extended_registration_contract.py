from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any


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
    original = extended_module.generate_extended_content
    if getattr(original, "_mmm_static_registrar_tree", False):
        return

    @wraps(original)
    def generated_with_static_registrar(*args: Any, **kwargs: Any) -> dict[str, Any]:
        receipt = original(*args, **kwargs)
        if not isinstance(receipt, dict) or receipt.get("status") != "GENERATED":
            return receipt

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

        result = dict(receipt)
        result["registrar_dispatch_count"] = len(dispatch_files)
        result["static_registration_unit_count"] = len(leaf_names)
        result["static_registration_root"] = dispatch_root
        result["static_registration_receipt"] = registration_receipt
        if dispatch_receipt is not None:
            result["registrar_dispatch_receipt"] = dispatch_receipt
        return result

    generated_with_static_registrar._mmm_static_registrar_tree = True
    extended_module.generate_extended_content = generated_with_static_registrar


def install(extended_module: Any) -> None:
    """Replace runtime classpath reflection with a bounded compile-time registrar tree."""

    _install_static_registration(extended_module)
