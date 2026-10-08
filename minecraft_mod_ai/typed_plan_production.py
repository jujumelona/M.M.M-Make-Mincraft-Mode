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
from .typed_host_generation_contract import (
    normalize_typed_network_sync_config,
    normalize_typed_resource_policy_config,
    normalize_typed_state_store_config,
)
from .typed_platform_ir import network_sync_requires_state
from .typed_plan_java import render_typed_plan_java


def _typed_program_path(package_name: str) -> str:
    return (
        "src/main/java/"
        + str(package_name).replace(".", "/")
        + "/AuthoredProgram.java"
    )


def _normalized_typed_host_configs(
    config: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    raw_state = config.get("typed_plan_state_section")
    state_section = raw_state if isinstance(raw_state, Mapping) else None
    structured_raw = config.get("typed_plan_structured_sections")
    structured = (
        structured_raw
        if isinstance(structured_raw, Mapping)
        else None
    )

    raw_state_store = config.get("typed_state_store")
    state_store = None
    if raw_state_store is not None:
        if not isinstance(raw_state_store, Mapping):
            raise ValueError("TYPED_STATE_STORE_CONFIG_INVALID")
        state_store = normalize_typed_state_store_config(
            raw_state_store,
            structured_sections=structured,
            state_section=state_section,
        )

    raw_network_sync = config.get("typed_network_sync")
    network_sync = None
    if raw_network_sync is not None:
        if not isinstance(raw_network_sync, Mapping):
            raise ValueError("TYPED_NETWORK_SYNC_CONFIG_INVALID")
        network_sync = normalize_typed_network_sync_config(raw_network_sync)

    raw_resource_policy = config.get("typed_resource_policy")
    resource_policy = None
    if raw_resource_policy is not None:
        if not isinstance(raw_resource_policy, Mapping):
            raise ValueError("TYPED_RESOURCE_POLICY_CONFIG_INVALID")
        resource_policy = normalize_typed_resource_policy_config(
            raw_resource_policy
        )

    return state_section, state_store, network_sync, resource_policy


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




def _state_variable_contracts(
    section: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return canonical network-visible state keys and their declared value types.

    Network policy worksheet rows describe semantic synchronization policy. They are
    not an identifier namespace. Concrete payload keys come only from the canonical
    state_model.variables authority.
    """

    normalized = normalize_structured_state_section(section)
    rows = normalized["specification"]["variables"]
    if not isinstance(rows, Sequence) or isinstance(
        rows, (str, bytes, bytearray)
    ):
        raise ValueError("TYPED_STATE_VARIABLES_ARRAY_REQUIRED")

    contracts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"TYPED_STATE_VARIABLE_ROW_INVALID: variables[{index}]"
            )
        name = str(row["name"]).strip()
        type_name = str(row["type"]).strip()
        if not name:
            raise ValueError(
                f"TYPED_STATE_VARIABLE_NAME_REQUIRED: variables[{index}]"
            )
        if name in seen:
            raise ValueError(
                f"TYPED_STATE_VARIABLE_DUPLICATE: {name!r}"
            )
        if type_name not in {"boolean", "int", "long", "double", "string"}:
            raise ValueError(
                f"TYPED_STATE_VARIABLE_TYPE_INVALID: {name!r} -> {type_name!r}"
            )
        seen.add(name)
        contracts.append((name, type_name))
    return tuple(contracts)


def _state_variable_names(section: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(name for name, _type_name in _state_variable_contracts(section))


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
    minecraft_version: str = "",
) -> str:
    namespace_literal = json.dumps(namespace, ensure_ascii=True)
    schema_version = str(config["schema_version"])
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
    for index, raw in enumerate(config["migrations"]):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"TYPED_STATE_STORE_MIGRATION_ROW_INVALID: {index}"
            )
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
    source = f"""package {package_name};

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
    if str(minecraft_version).strip().startswith("26."):
        # Changed 2026-10-08: keep respawn transfer bound to Mojang 26.1 symbols.
        return source.replace(
            "net.minecraft.server.network.ServerPlayerEntity",
            "net.minecraft.server.level.ServerPlayer",
        ).replace("getUuidAsString()", "getUUID().toString()")
    return source


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
    minecraft_version: str = "",
) -> dict[str, str]:
    state_names = _state_variable_names(section)
    if not state_names:
        raise ValueError(
            "TYPED_STATE_STORE_VARIABLES_REQUIRED: persistent state requires "
            "canonical state_model.variables."
        )
    namespace = str(config["namespace"]).strip()
    package_path = package_name.replace(".", "/")
    from .system_templates_common import _persistent_store_java

    return {
        f"src/main/java/{package_path}/system/MmmPersistentStore.java":
            _persistent_store_java(
                package_name, mod_id, minecraft_version=minecraft_version
            ),
        f"src/main/java/{package_path}/AuthoredStatePersistence.java":
            _state_persistence_java(
                package_name,
                namespace,
                state_names,
                config,
                minecraft_version=minecraft_version,
            ),
    }



def _structured_rows(
    structured: Mapping[str, Any],
    section: str,
    concern: str,
) -> tuple[Mapping[str, Any], ...]:
    raw_section = structured.get(section)
    if raw_section is None:
        return ()
    if not isinstance(raw_section, Mapping):
        raise ValueError(
            f"TYPED_STRUCTURED_SECTION_INVALID: {section}"
        )
    specification = raw_section.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError(
            f"TYPED_STRUCTURED_SPECIFICATION_REQUIRED: {section}"
        )
    rows = specification.get(concern, ())
    if not isinstance(rows, Sequence) or isinstance(
        rows, (str, bytes, bytearray)
    ):
        raise ValueError(
            f"TYPED_STRUCTURED_CONCERN_INVALID: {section}.{concern}"
        )
    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"TYPED_STRUCTURED_ROW_INVALID: {section}.{concern}[{index}]"
            )
        result.append(row)
    return tuple(result)


def _network_payload_guard(field: str, type_name: str) -> str:
    literal = json.dumps(field, ensure_ascii=True)
    if type_name in {"int", "long", "double"}:
        condition = "value instanceof Number"
    elif type_name == "boolean":
        condition = "value instanceof Boolean"
    elif type_name == "string":
        condition = "value instanceof String"
    else:
        raise ValueError(
            f"TYPED_NETWORK_STATE_TYPE_INVALID: {field!r} -> {type_name!r}"
        )
    return f"            case {literal} -> {condition};"


def _network_policy_files(
    *,
    package_name: str,
    mod_id: str,
    state_section: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    stateful = network_sync_requires_state(
        tuple(config.get("__covers", ()))
    )
    if not stateful:
        package_path = package_name.replace(".", "/")
        source = f"""package {package_name};

// MMM:TYPED_NETWORK_SYNC_OWNER
public final class AuthoredNetworkSync {{
    private AuthoredNetworkSync() {{}}

    public static void register() {{
        // Server-authoritative boundary: no client mutation channel is generated.
    }}

    public static boolean clientMutationAllowed() {{
        return false;
    }}
}}
"""
        return {
            f"src/main/java/{package_path}/AuthoredNetworkSync.java": source,
        }

    state_contracts = _state_variable_contracts(state_section)
    if not state_contracts:
        raise ValueError(
            "TYPED_NETWORK_STATE_REQUIRED: network synchronization requires "
            "at least one canonical state variable."
        )

    # Concrete transport fields are host-owned canonical state identifiers. The
    # authority_and_network worksheet describes policy and intent only; its prose
    # fields must never be reinterpreted as executable state keys.
    fields = [name for name, _type_name in state_contracts]
    field_types = {
        name: type_name
        for name, type_name in state_contracts
    }

    interval = int(config["sync_interval_ticks"])
    max_bytes = int(config["max_payload_bytes"])
    channel_path = "typed_state_sync"

    write_lines: list[str] = []
    guard_lines: list[str] = []
    for name in fields:
        literal = json.dumps(name, ensure_ascii=True)
        write_lines.extend([
            f"        value = AuthoredStateModel.getState({literal}, context);",
            f"        if (value != null && accepted({literal}, value)) {{",
            f"            payload.add({literal}, GSON.toJsonTree(value));",
            "        }",
        ])
        guard_lines.append(
            _network_payload_guard(name, field_types[name])
        )

    server_source = f"""package {package_name};

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.networking.v1.PacketByteBufs;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.network.PacketByteBuf;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.Identifier;

import java.nio.charset.StandardCharsets;
import java.util.Map;

// MMM:TYPED_NETWORK_SYNC_OWNER
public final class AuthoredNetworkSync {{
    public static final Identifier CHANNEL =
            new Identifier({json.dumps(mod_id)}, {json.dumps(channel_path)});
    private static final Gson GSON = new GsonBuilder().create();
    private static final int SYNC_INTERVAL_TICKS = {interval};
    private static final int MAX_PAYLOAD_BYTES = {max_bytes};
    private static boolean registered;
    private static int ticks;

    private AuthoredNetworkSync() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        ServerPlayConnectionEvents.JOIN.register(
                (handler, sender, server) -> sync(handler.player)
        );
        ServerTickEvents.END_SERVER_TICK.register(server -> {{
            ticks++;
            if (ticks < SYNC_INTERVAL_TICKS) return;
            ticks = 0;
            for (ServerPlayerEntity player :
                    server.getPlayerManager().getPlayerList()) {{
                sync(player);
            }}
        }});
    }}

    private static void sync(ServerPlayerEntity player) {{
        JsonObject payload = new JsonObject();
        Map<String, Object> context = Map.of(
                "player",
                player.getUuidAsString()
        );
        Object value;
{chr(10).join(write_lines)}
        String encoded = GSON.toJson(payload);
        if (encoded.getBytes(StandardCharsets.UTF_8).length
                > MAX_PAYLOAD_BYTES) {{
            throw new IllegalStateException(
                    "Typed synchronization payload exceeds configured bound"
            );
        }}
        PacketByteBuf buffer = PacketByteBufs.create();
        buffer.writeString(encoded, MAX_PAYLOAD_BYTES);
        ServerPlayNetworking.send(player, CHANNEL, buffer);
    }}

    private static boolean accepted(String field, Object value) {{
        return switch (field) {{
{chr(10).join(guard_lines)}
            default -> false;
        }};
    }}
}}
"""

    client_source = f"""package {package_name};

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.minecraft.util.Identifier;

// MMM:TYPED_NETWORK_CLIENT_OWNER
public final class AuthoredNetworkClient implements ClientModInitializer {{
    private static final Identifier CHANNEL =
            new Identifier({json.dumps(mod_id)}, {json.dumps(channel_path)});
    private static final Gson GSON = new GsonBuilder().create();
    private static final int MAX_PAYLOAD_BYTES = {max_bytes};
    private static JsonObject snapshot = new JsonObject();

    @Override
    public void onInitializeClient() {{
        ClientPlayNetworking.registerGlobalReceiver(
                CHANNEL,
                (client, handler, buffer, responseSender) -> {{
                    String encoded = buffer.readString(MAX_PAYLOAD_BYTES);
                    JsonObject parsed = JsonParser.parseString(encoded)
                            .getAsJsonObject();
                    client.execute(() -> update(parsed));
                }}
        );
    }}

    private static synchronized void update(JsonObject value) {{
        snapshot = value.deepCopy();
    }}

    public static synchronized JsonObject snapshot() {{
        return snapshot.deepCopy();
    }}

    public static synchronized Object value(String key) {{
        if (!snapshot.has(key)) return null;
        return GSON.fromJson(snapshot.get(key), Object.class);
    }}
}}
"""

    package_path = package_name.replace(".", "/")
    return {
        f"src/main/java/{package_path}/AuthoredNetworkSync.java":
            server_source,
        f"src/main/java/{package_path}/AuthoredNetworkClient.java":
            client_source,
    }


def _merged_lang_text(
    root: Path,
    relative: str,
    additions: Mapping[str, str],
) -> str:
    target = root / relative
    existing: dict[str, Any] = {}
    if target.is_file() and not target.is_symlink():
        try:
            parsed = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"TYPED_RESOURCE_LANGUAGE_INVALID: {relative}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"TYPED_RESOURCE_LANGUAGE_INVALID: {relative}"
            )
        existing = {
            str(key): value
            for key, value in parsed.items()
        }
    for key, value in additions.items():
        existing[str(key)] = str(value)
    return json.dumps(
        existing,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _resource_policy_files(
    *,
    root: Path,
    package_name: str,
    mod_id: str,
    structured: Mapping[str, Any],
) -> dict[str, str]:
    missing_rows = _structured_rows(
        structured,
        "resources_and_ui",
        "missing_resources",
    )
    accessibility_rows = _structured_rows(
        structured,
        "resources_and_ui",
        "accessibility",
    )
    package_path = package_name.replace(".", "/")
    files: dict[str, str] = {}

    if missing_rows:
        files[
            f"src/main/resources/assets/{mod_id}/models/item/"
            "mmm_missing_resource.json"
        ] = json.dumps(
            {
                "parent": "minecraft:item/generated",
                "textures": {
                    "layer0": "minecraft:item/barrier"
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        files[
            f"src/main/resources/data/{mod_id}/mmm_policies/"
            "missing_resources.json"
        ] = json.dumps(
            {
                "schema_version": "mmm/missing-resource-policy-v1",
                "fallback_model": (
                    f"{mod_id}:item/mmm_missing_resource"
                ),
                "rules": [dict(row) for row in missing_rows],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    if accessibility_rows:
        en: dict[str, str] = {}
        ko: dict[str, str] = {}
        message_keys: list[str] = []
        for index, row in enumerate(accessibility_rows, 1):
            key = f"text.{mod_id}.accessibility_{index}"
            feedback = str(row.get("feedback") or "").strip()
            localization = str(
                row.get("localization") or ""
            ).strip()
            observation = str(
                row.get("observation") or ""
            ).strip()
            en[key] = feedback or observation or localization
            ko[key] = localization or feedback or observation
            message_keys.append(key)

        en_path = (
            f"src/main/resources/assets/{mod_id}/lang/en_us.json"
        )
        ko_path = (
            f"src/main/resources/assets/{mod_id}/lang/ko_kr.json"
        )
        files[en_path] = _merged_lang_text(root, en_path, en)
        files[ko_path] = _merged_lang_text(root, ko_path, ko)

        send_lines = "\n".join(
            "            handler.player.sendMessage("
            f"net.minecraft.text.Text.translatable({json.dumps(key)}), false);"
            for key in message_keys
        )
        files[
            f"src/main/java/{package_path}/"
            "AuthoredAccessibility.java"
        ] = f"""package {package_name};

import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;

// MMM:TYPED_ACCESSIBILITY_OWNER
public final class AuthoredAccessibility {{
    private static boolean registered;

    private AuthoredAccessibility() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        ServerPlayConnectionEvents.JOIN.register(
                (handler, sender, server) -> {{
{send_lines}
                }}
        );
    }}
}}
"""
    return files

def validate_typed_plan_generation_contract(
    module: ProductionModule,
    *,
    package_name: str,
    mod_id: str,
) -> dict[str, Any]:
    """Dry-compile every model-free typed-host source contract before dispatch.

    Filesystem ownership/merge checks remain generation-time concerns. Semantic
    source rendering does not: the exact PlanIR/state/persistence/network
    renderers are exercised here so proposal lowering cannot defer contract
    mismatches into a generation worker.
    """

    module.validate()
    config = module.config
    raw_plan = config.get("typed_plan_ir")
    if not isinstance(raw_plan, Mapping) or not raw_plan:
        raise ValueError("TYPED_PLAN_IR_REQUIRED")

    configured_package = str(
        config.get("typed_plan_package") or package_name
    ).strip()
    if configured_package != package_name:
        raise ValueError(
            "TYPED_PLAN_PACKAGE_MISMATCH: "
            f"{configured_package!r} != {package_name!r}"
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

    raw_capabilities = config.get("typed_plan_capabilities")
    capabilities = (
        dict(raw_capabilities)
        if isinstance(raw_capabilities, Mapping)
        else {}
    )
    files: dict[str, str] = {
        expected_path: render_typed_plan_java(
            raw_plan,
            package=package_name,
            capabilities=capabilities,
        )
    }

    capability_ids = typed_plan_capability_ids(raw_plan)
    missing_capabilities = [
        capability_id
        for capability_id in capability_ids
        if capability_id not in capabilities
    ]
    if missing_capabilities:
        raise ValueError(
            "TYPED_PLAN_CAPABILITY_BINDING_REQUIRED: "
            + ", ".join(missing_capabilities)
        )
    if capability_ids:
        files[
            "src/main/java/"
            + package_name.replace(".", "/")
            + "/AuthoredHostCapabilities.java"
        ] = render_typed_host_capabilities_java(
            package_name,
            minecraft_version=str(config.get("minecraft_version") or ""),
        )

    (
        raw_state,
        raw_state_store,
        raw_network_sync,
        raw_resource_policy,
    ) = _normalized_typed_host_configs(config)
    state_authority_present = isinstance(raw_state, Mapping) and bool(raw_state)

    if raw_state_store is not None:
        if not isinstance(raw_state_store, Mapping):
            raise ValueError("TYPED_STATE_STORE_CONFIG_INVALID")
        if not state_authority_present:
            raise ValueError("TYPED_STATE_STORE_STATE_AUTHORITY_REQUIRED")
        files.update(
            _persistence_files(
                package_name=package_name,
                mod_id=mod_id,
                section=raw_state,
                config=raw_state_store,
                minecraft_version=str(config.get("minecraft_version") or ""),
            )
        )

    network_sync_needs_state = False
    if raw_network_sync is not None:
        if not isinstance(raw_network_sync, Mapping):
            raise ValueError("TYPED_NETWORK_SYNC_CONFIG_INVALID")
        network_sync_needs_state = network_sync_requires_state(
            tuple(raw_network_sync.get("__covers", ()))
        )
        if network_sync_needs_state and not state_authority_present:
            raise ValueError("TYPED_NETWORK_STATE_AUTHORITY_REQUIRED")
        files.update(
            _network_policy_files(
                package_name=package_name,
                mod_id=mod_id,
                state_section=(
                    raw_state
                    if isinstance(raw_state, Mapping)
                    else {}
                ),
                config=raw_network_sync,
            )
        )

    if raw_resource_policy is not None and not isinstance(
        raw_resource_policy, Mapping
    ):
        raise ValueError("TYPED_RESOURCE_POLICY_CONFIG_INVALID")

    state_required = (
        typed_plan_uses_state(raw_plan)
        or raw_state_store is not None
        or network_sync_needs_state
        or state_authority_present
    )
    if state_required:
        if not state_authority_present:
            raise ValueError("TYPED_PLAN_STATE_AUTHORITY_REQUIRED")
        files[
            "src/main/java/"
            + package_name.replace(".", "/")
            + "/AuthoredStateModel.java"
        ] = render_production_state_java(
            raw_state,
            package_name=package_name,
        )

    return {
        "source_count": len(files),
        "paths": tuple(sorted(files)),
        "state_required": state_required,
        "network_sync_needs_state": network_sync_needs_state,
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
            package_name,
            minecraft_version=str(config.get("minecraft_version") or ""),
        )

    raw_structured = config.get("typed_plan_structured_sections")
    structured = (
        dict(raw_structured)
        if isinstance(raw_structured, Mapping)
        else {}
    )
    (
        raw_state,
        raw_state_store,
        raw_network_sync,
        raw_resource_policy,
    ) = _normalized_typed_host_configs(config)
    state_authority_present = isinstance(raw_state, Mapping) and bool(raw_state)

    if raw_state_store is not None:
        if not isinstance(raw_state_store, Mapping):
            raise ValueError("TYPED_STATE_STORE_CONFIG_INVALID")
        if not state_authority_present:
            raise ValueError("TYPED_STATE_STORE_STATE_AUTHORITY_REQUIRED")
        files.update(
            _persistence_files(
                package_name=package_name,
                mod_id=info.mod_id,
                section=raw_state,
                config=raw_state_store,
                minecraft_version=str(config.get("minecraft_version") or ""),
            )
        )

    network_sync_needs_state = False
    if raw_network_sync is not None:
        if not isinstance(raw_network_sync, Mapping):
            raise ValueError("TYPED_NETWORK_SYNC_CONFIG_INVALID")
        network_sync_needs_state = network_sync_requires_state(
            tuple(raw_network_sync.get("__covers", ()))
        )
        if network_sync_needs_state and not state_authority_present:
            raise ValueError("TYPED_NETWORK_STATE_AUTHORITY_REQUIRED")
        files.update(
            _network_policy_files(
                package_name=package_name,
                mod_id=info.mod_id,
                state_section=(
                    raw_state
                    if isinstance(raw_state, Mapping)
                    else {}
                ),
                config=raw_network_sync,
            )
        )

    if raw_resource_policy is not None:
        if not isinstance(raw_resource_policy, Mapping):
            raise ValueError("TYPED_RESOURCE_POLICY_CONFIG_INVALID")
        files.update(
            _resource_policy_files(
                root=root,
                package_name=package_name,
                mod_id=info.mod_id,
                structured=structured,
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

    if raw_network_sync is not None:
        package_path = package_name.replace(".", "/")
        _assert_host_owned_or_absent(
            root,
            f"src/main/java/{package_path}/AuthoredNetworkSync.java",
            marker="// MMM:TYPED_NETWORK_SYNC_OWNER",
        )
        _assert_host_owned_or_absent(
            root,
            f"src/main/java/{package_path}/AuthoredNetworkClient.java",
            marker="// MMM:TYPED_NETWORK_CLIENT_OWNER",
        )

    if raw_resource_policy is not None:
        package_path = package_name.replace(".", "/")
        accessibility_path = (
            f"src/main/java/{package_path}/AuthoredAccessibility.java"
        )
        if accessibility_path in files:
            _assert_host_owned_or_absent(
                root,
                accessibility_path,
                marker="// MMM:TYPED_ACCESSIBILITY_OWNER",
            )

    if (
        typed_plan_uses_state(raw_plan)
        or raw_state_store is not None
        or network_sync_needs_state
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
    if raw_network_sync is not None:
        ensure_main_initializer_call(
            info,
            import_line=f"import {package_name}.AuthoredNetworkSync",
            call_line="AuthoredNetworkSync.register()",
            marker="typed-network-sync",
        )
        package_path = package_name.replace(".", "/")
        if (
            f"src/main/java/{package_path}/AuthoredNetworkClient.java"
            in files
        ):
            ensure_client_entrypoint(
                info,
                entrypoint=f"{package_name}.AuthoredNetworkClient",
            )
    if raw_resource_policy is not None:
        package_path = package_name.replace(".", "/")
        accessibility_path = (
            f"src/main/java/{package_path}/AuthoredAccessibility.java"
        )
        if accessibility_path in files:
            ensure_main_initializer_call(
                info,
                import_line=f"import {package_name}.AuthoredAccessibility",
                call_line="AuthoredAccessibility.register()",
                marker="typed-accessibility",
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


__all__ = [
    "generate_typed_plan_module",
    "validate_typed_plan_generation_contract",
]
