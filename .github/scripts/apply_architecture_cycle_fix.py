from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "minecraft_mod_ai"


def read(name: str) -> str:
    return (PKG / name).read_text(encoding="utf-8")


def write(name: str, text: str) -> None:
    (PKG / name).write_text(text, encoding="utf-8")


write(
    "project_platform_identity.py",
    '''from __future__ import annotations

"""Read the host-owned platform identity from one concrete project tree.

This module deliberately owns only local project evidence. Executable provider
resolution remains in :mod:`platform_catalog`, keeping low-level tool runtime code
independent from the platform registry and its planning/research dependency graph.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectPlatformIdentity:
    minecraft_version: str
    loader: str
    lock_values: Mapping[str, Any] | None
    gradle_properties: Mapping[str, str]


def _project_platform_lock(root: Path) -> Path | None:
    direct = root / ".minecraft_ai" / "platform-lock.json"
    if direct.is_file() and not direct.is_symlink():
        return direct
    if root.name != "project":
        return None
    checkpoint_root = root.parent
    checkpoint_directory = checkpoint_root.parent
    metadata_root = checkpoint_directory.parent
    key = checkpoint_root.name
    valid_key = len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    if checkpoint_directory.name != ".mmm-custom-checkpoints" or metadata_root.name != ".minecraft_ai" or not valid_key:
        return None
    inherited = metadata_root / "platform-lock.json"
    return inherited if inherited.is_file() and not inherited.is_symlink() else None


def _read_gradle_properties(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"gradle.properties is missing: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _fabric_descriptor_identifies_project(root: Path) -> bool:
    descriptor = root / "src" / "main" / "resources" / "fabric.mod.json"
    if not descriptor.is_file() or descriptor.is_symlink():
        return False
    try:
        raw = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    return (
        type(raw.get("schemaVersion")) is int
        and raw["schemaVersion"] >= 1
        and isinstance(raw.get("id"), str)
        and bool(raw["id"].strip())
        and isinstance(raw.get("version"), str)
        and bool(raw["version"].strip())
    )


def project_platform_identity(project_root: str | Path) -> ProjectPlatformIdentity:
    root = Path(project_root).expanduser().resolve()
    lock_file = _project_platform_lock(root)
    if lock_file is not None:
        raw = json.loads(lock_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Generated platform lock must be an object.")
        version = str(raw.get("minecraft_version") or "").strip()
        loader = str(raw.get("loader") or "").strip().casefold()
        if not version or not loader:
            raise ValueError("Generated platform lock must bind minecraft_version and loader.")
        return ProjectPlatformIdentity(version, loader, raw, {})

    properties = _read_gradle_properties(root / "gradle.properties")
    version = properties.get("minecraft_version", "").strip()
    if not version:
        raise ValueError("Existing project minecraft_version is missing.")
    loader = properties.get("loader", "").strip().casefold()
    if not loader:
        fabric_properties = properties.get("loader_version") and properties.get("fabric_version")
        if fabric_properties or _fabric_descriptor_identifies_project(root):
            loader = "fabric"
        else:
            raise ValueError("Existing project loader could not be identified unambiguously.")
    return ProjectPlatformIdentity(version, loader, None, properties)


__all__ = ["ProjectPlatformIdentity", "project_platform_identity"]
''',
)

agent = read("agent_tool_runtime.py")
old = '''        if stage == "generation" and _looks_like_bound_project(Path(self.workspace_root)):\n            from .platform_catalog import adapter_from_project\n\n            # Child services cannot see the parent's selected adapter. Bind from this\n            # runtime's project lock, never a process-global target from another worker.\n            target = adapter_from_project(Path(self.workspace_root))\n            env["MMM_MCP_MINECRAFT_VERSION"] = target.minecraft_version\n            env["MMM_MCP_LOADER"] = target.loader\n'''
new = '''        if stage == "generation" and _looks_like_bound_project(Path(self.workspace_root)):\n            from .project_platform_identity import project_platform_identity\n\n            # Child services inherit only the exact project identity. Executable-provider\n            # validation belongs to the child/platform owner, not this transport layer.\n            target = project_platform_identity(Path(self.workspace_root))\n            env["MMM_MCP_MINECRAFT_VERSION"] = target.minecraft_version\n            env["MMM_MCP_LOADER"] = target.loader\n'''
if old not in agent:
    raise SystemExit("agent_tool_runtime platform import anchor not found")
write("agent_tool_runtime.py", agent.replace(old, new, 1))

catalog = read("platform_catalog.py")
catalog = catalog.replace("import json\n", "", 1)
import_anchor = '''from .platform_live_discovery import (\n    PlatformDiscoveryError,\n    _emit_discovery_log,\n)\n'''
if import_anchor not in catalog:
    raise SystemExit("platform_catalog import anchor not found")
catalog = catalog.replace(
    import_anchor,
    import_anchor + "from .project_platform_identity import project_platform_identity\n",
    1,
)
start = catalog.index("def _project_platform_lock(")
end = catalog.index("def adapter_from_project(", start)
catalog = catalog[:start] + catalog[end:]
start = catalog.index("def adapter_from_project(")
end = catalog.index("\ndef platform_catalog_receipt(", start)
replacement = '''def adapter_from_project(project_root: str | Path) -> TargetContract:\n    identity = project_platform_identity(project_root)\n    adapter = adapter_for_target(identity.minecraft_version, identity.loader)\n    if identity.lock_values is not None:\n        fields = [\n            "minecraft_version", "loader", "java_version", "fabric_loader",\n            "fabric_api", "fabric_loom", "gradle",\n        ]\n        if adapter.mappings_applicable:\n            fields.append("yarn_mappings")\n        for field in fields:\n            if str(identity.lock_values.get(field) or "") != str(getattr(adapter, field) or ""):\n                raise ValueError(\n                    f"Generated platform lock disagrees with executable provider: {field}"\n                )\n        return adapter\n\n    if identity.loader == "fabric":\n        expected = {\n            "loader_version": adapter.fabric_loader,\n            "fabric_version": adapter.fabric_api,\n            "loom_version": adapter.fabric_loom,\n        }\n        if adapter.mappings_kind == "yarn":\n            expected["yarn_mappings"] = adapter.mappings_version\n        for key, expected_value in expected.items():\n            actual = identity.gradle_properties.get(key)\n            if actual and actual != expected_value:\n                raise ValueError(\n                    f"Project Gradle property {key} disagrees with executable provider discovery."\n                )\n    return adapter\n\n'''
catalog = catalog[:start] + replacement + catalog[end + 1:]
# Remove the old private parser now owned by project_platform_identity.
old_reader = catalog.find("def _read_gradle_properties(")
if old_reader != -1:
    reader_end = catalog.index("\ndef _loader_id(", old_reader)
    catalog = catalog[:old_reader] + catalog[reader_end + 1:]
write("platform_catalog.py", catalog)

write(
    "tool_validation_surface.py",
    '''from __future__ import annotations

"""Pure tool-schema validation-surface composition.

No transport, adapter, router, or runtime owner is imported here. Both the native
adapter and its installation-time contract depend downward on this module.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def assert_unique_schema_names(schemas: Sequence[Any], *, surface: str) -> None:
    seen: set[str] = set()
    for schema in schemas:
        name = tool_name(schema)
        if not name:
            continue
        if name in seen:
            raise RuntimeError(f"duplicate tool schema name {name!r} in {surface} surface")
        seen.add(name)


def validation_surface(visible: Sequence[Any], authorized: Sequence[Any]) -> tuple[Any, ...]:
    assert_unique_schema_names(visible, surface="model-visible")
    assert_unique_schema_names(authorized, surface="authorized-validation")
    result = list(visible)
    visible_names = {name for schema in visible if (name := tool_name(schema))}
    result.extend(
        schema
        for schema in authorized
        if not (name := tool_name(schema)) or name not in visible_names
    )
    return tuple(result)


__all__ = ["assert_unique_schema_names", "tool_name", "validation_surface"]
''',
)

contract = read("tool_validation_surface_contract.py")
start = contract.index("from collections.abc import Mapping, Sequence")
end = contract.index("\ndef install()", start)
imports = '''from .tool_validation_surface import (\n    assert_unique_schema_names as _assert_unique_schema_names,\n    validation_surface as _validation_surface,\n)\n\n'''
contract = contract[:start] + imports + contract[end + 1:]
write("tool_validation_surface_contract.py", contract)

llama = read("model_adapters/llama_cpp_adapter.py")
old = "    from ..tool_validation_surface_contract import _validation_surface\n\n    visible = tuple(request.tools)"
new = "    from ..tool_validation_surface import validation_surface\n\n    visible = tuple(request.tools)"
if old not in llama:
    raise SystemExit("llama validation surface import anchor not found")
llama = llama.replace(old, new, 1)
llama = llama.replace("    validation_surface = _validation_surface(visible, authorized)\n    return _tool_schema_map(tuple(dict(tool) for tool in validation_surface))", "    schemas = validation_surface(visible, authorized)\n    return _tool_schema_map(tuple(dict(tool) for tool in schemas))", 1)
write("model_adapters/llama_cpp_adapter.py", llama)
