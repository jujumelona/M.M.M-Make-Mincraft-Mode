from __future__ import annotations

"""Model-free production backend for authored Typed PlanIR modules."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .production_state_compiler import (
    normalize_structured_state_section,
    render_production_state_java,
)
from .project_edit import (
    ensure_client_entrypoint,
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
from .typed_host_capabilities import render_typed_host_capabilities_java
from .typed_plan_ir import typed_plan_capability_ids, typed_plan_uses_state
from .typed_plan_java import render_typed_plan_java


def _typed_program_path(package_name: str) -> str:
    return (
        "src/main/java/"
        + str(package_name).replace(".", "/")
        + "/AuthoredProgram.java"
    )


def _assert_host_owned_or_absent(
    root: Path,
    relative: str,
    *,
    marker: str,
) -> None:
    target = root / relative
    if not target.exists():
        return
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"TYPED_PLAN_TARGET_INVALID: {relative}")
    current = target.read_text(encoding="utf-8")
    if marker not in current:
        raise ValueError(f"TYPED_PLAN_OWNERSHIP_CONFLICT: {relative}")




def _state_variable_names(section: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = normalize_structured_state_section(section)
    rows = normalized["specification"].get("variables")
    if not isinstance(rows, Sequence) or isinstance(
        rows, (str, bytes, bytearray)
    ):
        return ()
    names: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _java_object_literal(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "Boolean.TRUE"
    if value is False:
        return "Boolean.FALSE"
    if isinstance(value, int):
        if -(2**31) <= value <= 2**31 - 1:
            return f"Integer.valueOf({value})"
        return f"Long.valueOf({value}L)"
    if isinstance(value, float):
        return f"Double.valueOf({repr(float(value))}d)"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise ValueError(
        f"TYPED_STATE_STORE_MIGRATION_VALUE_INVALID: {type(value).__name__}"
    )


def _state_persistence_java(
    package_name: str,
    namespace: str,
    state_names: tuple[str, ...],
    config: Mapping[str, Any],
) -> str:
    namespace_literal = json.dumps(namespace, ensure_ascii=True)
    schema_version = str(config.get("schema_version") or "1")
    schema_literal = json.dumps(schema_version, ensure_ascii=True)

    restore_lines: list[str] = []
    persist_lines: list[str] = []
    transfer_lines: list[str] = []
    for name in state_names:
        literal = json.dumps(name, ensure_ascii=True)
        restore_lines.extend([
            f"            if (data.containsKey({literal})) {{",
            f"                AuthoredStateModel.setState({literal}, data.get({literal}));",
            "            }",
        ])
        persist_lines.extend([
            f"            value = AuthoredStateModel.getState({literal});",
            f"            if (value != null) data.put({literal}, value);",
        ])
        transfer_lines.extend([
            f"            value = AuthoredStateModel.getState({literal}, oldContext);",
            f"            if (value != null) AuthoredStateModel.setState({literal}, value, newContext);",
        ])

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    destinations_by_source: dict[str, str] = {}
    for raw in config.get("migrations", []):
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("from_version") or "")
        destination = str(raw.get("to_version") or "")
        prior = destinations_by_source.get(source)
        if prior is not None and prior != destination:
            raise ValueError(
                "TYPED_STATE_STORE_MIGRATION_BRANCHING_FORBIDDEN: "
                f"{source!r} -> {prior!r}/{destination!r}"
            )
        destinations_by_source[source] = destination
        grouped.setdefault((source, destination), []).append(raw)

    migration_branches: list[str] = []
    for source, destination in grouped:
        operations: list[str] = []
        for migration in grouped[(source, destination)]:
            operation = str(migration.get("operation") or "")
            source_key = json.dumps(
                str(migration.get("source_key") or ""),
                ensure_ascii=True,
            )
            destination_key = json.dumps(
                str(migration.get("destination_key") or ""),
                ensure_ascii=True,
            )
            if operation == "preserve":
                operations.append("                // preserve existing values")
            elif operation == "rename_key":
                operations.extend([
                    f"                if (data.containsKey({source_key})) {{",
                    f"                    Object migratedValue = data.remove({source_key});",
                    f"                    data.put({destination_key}, migratedValue);",
                    "                }",
                ])
            elif operation == "delete_key":
                operations.append(f"                data.remove({source_key});")
            elif operation == "set_default":
                operations.append(
                    f"                data.putIfAbsent({destination_key}, "
                    + _java_object_literal(migration.get("value"))
                    + ");"
                )
            else:
                raise ValueError(
                    f"TYPED_STATE_STORE_MIGRATION_OPERATION_INVALID: {operation!r}"
                )
        operations.extend([
            f"                current = {json.dumps(destination, ensure_ascii=True)};",
            "                progressed = true;",
        ])
        prefix = "if" if not migration_branches else "else if"
        migration_branches.append(
            f'            {prefix} ({json.dumps(source, ensure_ascii=True)}.equals(current)) {{\n'
            + "\n".join(operations)
            + "\n            }"
        )

    migration_body = "\n".join(migration_branches)
    if not migration_body:
        migration_body = "            // No migration edges are required."

    transfer_registration = ""
    if config.get("transfer_on_respawn") is True:
        transfer_registration = """
        net.fabricmc.fabric.api.entity.event.v1.ServerPlayerEvents.COPY_FROM.register(
            (oldPlayer, newPlayer, alive) -> transferState(oldPlayer, newPlayer)
        );"""

    transfer_method = ""
    if config.get("transfer_on_respawn") is True:
        transfer_method = f"""
    private static void transferState(
            net.minecraft.server.network.ServerPlayerEntity oldPlayer,
            net.minecraft.server.network.ServerPlayerEntity newPlayer
    ) {{
        java.util.Map<String, Object> oldContext = java.util.Map.of(
                "player",
                oldPlayer.getUuidAsString()
        );
        java.util.Map<String, Object> newContext = java.util.Map.of(
                "player",
                newPlayer.getUuidAsString()
        );
        Object value;
{chr(10).join(transfer_lines)}
    }}
"""

    restore = "\n".join(restore_lines)
    persist = "\n".join(persist_lines)
    return f"""package {package_name};

import {package_name}.system.MmmPersistentStore;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;

// MMM:TYPED_STATE_PERSISTENCE_OWNER
public final class AuthoredStatePersistence {{
    private static final String CURRENT_SCHEMA_VERSION = {schema_literal};
    private static boolean registered;

    private AuthoredStatePersistence() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        ServerLifecycleEvents.SERVER_STARTED.register(server -> {{
            MmmPersistentStore.load(server);
            java.util.Map<String, Object> data =
                    MmmPersistentStore.namespace({namespace_literal});
            Object rawVersion = data.get("__mmm_schema_version");
            String storedVersion = rawVersion == null
                    ? CURRENT_SCHEMA_VERSION
                    : String.valueOf(rawVersion);
            String migratedVersion = migrate(data, storedVersion);
            if (!CURRENT_SCHEMA_VERSION.equals(migratedVersion)) {{
                throw new IllegalStateException(
                        "No typed migration path from " + storedVersion
                        + " to " + CURRENT_SCHEMA_VERSION
                );
            }}
            data.put("__mmm_schema_version", CURRENT_SCHEMA_VERSION);
{restore}
            if (!CURRENT_SCHEMA_VERSION.equals(storedVersion)) {{
                MmmPersistentStore.save(server);
            }}
        }});
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {{
            java.util.Map<String, Object> data =
                    MmmPersistentStore.namespace({namespace_literal});
            data.clear();
            data.put("__mmm_schema_version", CURRENT_SCHEMA_VERSION);
            Object value;
{persist}
            MmmPersistentStore.save(server);
        }});{transfer_registration}
    }}

    private static String migrate(
            java.util.Map<String, Object> data,
            String version
    ) {{
        String current = version;
        int guard = 0;
        while (!CURRENT_SCHEMA_VERSION.equals(current) && guard++ < 64) {{
            boolean progressed = false;
{migration_body}
            if (!progressed) break;
        }}
        return current;
    }}
{transfer_method}
}}
"""

def _assert_exact_or_absent(
    root: Path,
    relative: str,
    *,
    expected: str,
) -> None:
    target = root / relative
    if not target.exists():
        return
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"TYPED_PLAN_TARGET_INVALID: {relative}")
    if target.read_text(encoding="utf-8") != expected:
        raise ValueError(f"TYPED_PLAN_OWNERSHIP_CONFLICT: {relative}")


def _persistence_files(
    *,
    package_name: str,
    mod_id: str,
    section: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    state_names = _state_variable_names(section)
    if not state_names:
        raise ValueError(
            "TYPED_STATE_STORE_VARIABLES_REQUIRED: persistent state requires "
            "canonical state_model.variables."
        )
    namespace = str(config.get("namespace") or "authored_state").strip()
    package_path = package_name.replace(".", "/")
    from .system_templates_common import _persistent_store_java

    return {
        f"src/main/java/{package_path}/system/MmmPersistentStore.java":
            _persistent_store_java(package_name, mod_id),
        f"src/main/java/{package_path}/AuthoredStatePersistence.java":
            _state_persistence_java(
                package_name,
                namespace,
                state_names,
                config,
            ),
    }


def generate_typed_plan_module(
    project_root: str | Path,
    *,
    module: ProductionModule,
) -> dict[str, Any]:
    """Compile a persisted Typed PlanIR directly into Java with no model access."""

    module.validate()
    config = module.config
    raw_plan = config.get("typed_plan_ir")
    if not isinstance(raw_plan, Mapping) or not raw_plan:
        raise ValueError("TYPED_PLAN_IR_REQUIRED")

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("TYPED_PLAN_PROJECT_REQUIRED")
    info = inspect_fabric_project(root)

    package_name = str(
        config.get("typed_plan_package") or info.package_name
    ).strip()
    if package_name != info.package_name:
        raise ValueError(
            "TYPED_PLAN_PACKAGE_MISMATCH: "
            f"{package_name!r} != {info.package_name!r}"
        )
    expected_path = _typed_program_path(package_name)
    configured_path = str(
        config.get("typed_plan_path") or expected_path
    ).replace("\\", "/").strip()
    if configured_path != expected_path:
        raise ValueError(
            "TYPED_PLAN_PATH_MISMATCH: "
            f"{configured_path!r} != {expected_path!r}"
        )

    target = root / expected_path
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError("TYPED_PLAN_TARGET_INVALID")
        current = target.read_text(encoding="utf-8")
        if "// MMM:TYPED_PLAN_OWNER" not in current:
            raise ValueError("TYPED_PLAN_OWNERSHIP_CONFLICT")

    raw_capabilities = config.get("typed_plan_capabilities")
    capabilities = (
        dict(raw_capabilities)
        if isinstance(raw_capabilities, Mapping)
        else {}
    )
    source = render_typed_plan_java(
        raw_plan,
        package=package_name,
        capabilities=capabilities,
    )
    files = {expected_path: source}

    capability_ids = typed_plan_capability_ids(raw_plan)
    if capability_ids:
        missing = [
            capability_id
            for capability_id in capability_ids
            if capability_id not in capabilities
        ]
        if missing:
            raise ValueError(
                "TYPED_PLAN_CAPABILITY_BINDING_REQUIRED: "
                + ", ".join(missing)
            )
        capability_path = (
            "src/main/java/"
            + package_name.replace(".", "/")
            + "/AuthoredHostCapabilities.java"
        )
        _assert_host_owned_or_absent(
            root,
            capability_path,
            marker="// MMM:TYPED_HOST_CAPABILITIES_OWNER",
        )
        files[capability_path] = render_typed_host_capabilities_java(
            package_name
        )

    raw_state_store = config.get("typed_state_store")
    if raw_state_store is not None:
        if not isinstance(raw_state_store, Mapping):
            raise ValueError("TYPED_STATE_STORE_CONFIG_INVALID")
        raw_state = config.get("typed_plan_state_section")
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("TYPED_STATE_STORE_STATE_AUTHORITY_REQUIRED")
        files.update(
            _persistence_files(
                package_name=package_name,
                mod_id=info.mod_id,
                section=raw_state,
                config=raw_state_store,
            )
        )

    _assert_host_owned_or_absent(
        root,
        expected_path,
        marker="// MMM:TYPED_PLAN_OWNER",
    )

    if raw_state_store is not None:
        package_path = package_name.replace(".", "/")
        store_path = (
            f"src/main/java/{package_path}/system/MmmPersistentStore.java"
        )
        bridge_path = (
            f"src/main/java/{package_path}/AuthoredStatePersistence.java"
        )
        _assert_exact_or_absent(
            root,
            store_path,
            expected=files[store_path],
        )
        _assert_host_owned_or_absent(
            root,
            bridge_path,
            marker="// MMM:TYPED_STATE_PERSISTENCE_OWNER",
        )

    raw_state = config.get("typed_plan_state_section")
    state_authority_present = isinstance(raw_state, Mapping) and bool(raw_state)
    if (
        typed_plan_uses_state(raw_plan)
        or raw_state_store is not None
        or state_authority_present
    ):
        if not state_authority_present:
            raise ValueError("TYPED_PLAN_STATE_AUTHORITY_REQUIRED")
        state_path = (
            "src/main/java/"
            + package_name.replace(".", "/")
            + "/AuthoredStateModel.java"
        )
        _assert_host_owned_or_absent(
            root,
            state_path,
            marker="// MMM:TYPED_PLAN_STATE_OWNER",
        )
        files[state_path] = render_production_state_java(
            raw_state,
            package_name=package_name,
        )

    patch_receipt = write_text_files(
        info,
        files,
        replace_existing=True,
    )
    if raw_state_store is not None:
        ensure_main_initializer_call(
            info,
            import_line=f"import {package_name}.AuthoredStatePersistence",
            call_line="AuthoredStatePersistence.register()",
            marker="typed-state-persistence",
        )
    touched_paths = sorted(files)
    return {
        "schema_version": "mmm/custom-module-result-v3",
        "module_id": module.module_id,
        "kind": module.kind,
        "status": "SOURCE_GENERATED",
        "touched_paths": touched_paths,
        "patch_receipt": patch_receipt,
        "operation_count": len(patch_receipt.get("operations") or ()),
        "required_gates": list(module.required_gates),
        "generation_verification": {
            "status": "PASS",
            "mode": "host_typed_plan_ir_deferred_to_pipeline",
            "compile_deferred": True,
            "model_calls": 0,
        },
        "runtime_tests": [
            "Execute the requested GameTest/runtime gates."
        ],
    }


__all__ = ["generate_typed_plan_module"]
