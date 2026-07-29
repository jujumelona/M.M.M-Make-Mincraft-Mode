from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_PACKS = frozenset(
    {
        "quest-system",
        "class-skill-system",
        "economy-shop",
        "gui-networking",
        "party-guild",
    }
)


def generate_system_pack(
    *,
    project_root: str | Path,
    pack_id: str,
    mod_id: str,
    package_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate Java-17 domain foundations and a versioned data contract.

    These foundations are real compileable Java types but intentionally do not
    claim Fabric event/packet registration until MinecraftCoder binds them and the
    build/runtime gates pass.
    """

    if pack_id not in _PACKS:
        raise ValueError(f"Unknown system pack: {pack_id}")
    if not _ID.fullmatch(mod_id) or not _PACKAGE.fullmatch(package_name):
        raise ValueError("Invalid mod id or Java package.")
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    class_prefix = "".join(part.capitalize() for part in pack_id.split("-"))
    java_root = root / "src" / "main" / "java" / Path(*package_name.split(".")) / "system"
    data_root = root / "src" / "main" / "resources" / "data" / mod_id / "mmm_systems"
    java_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    contract_path = data_root / f"{pack_id}.json"
    contract = {
        "schema_version": f"mmm/{pack_id}-v1",
        "pack_id": pack_id,
        "config": config,
        "server_authoritative": True,
        "minecraft_version": "1.20.1",
        "loader": "fabric",
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    java_path = java_root / f"{class_prefix}Contract.java"
    java_path.write_text(
        _java_contract(package_name, class_prefix, pack_id),
        encoding="utf-8",
    )
    validator_path = java_root / f"{class_prefix}Validator.java"
    validator_path.write_text(
        _java_validator(package_name, class_prefix, pack_id),
        encoding="utf-8",
    )
    return {
        "schema_version": "mmm/system-pack-generation-v1",
        "pack_id": pack_id,
        "files": [str(contract_path), str(java_path), str(validator_path)],
        "status": "fabric_binding_and_runtime_tests_required",
        "required_gates": [
            "server-side Fabric registration",
            "JDT diagnostics",
            "Gradle clean build",
            "GameTest",
            "restart persistence test",
            "multiplayer authority and replay test",
        ],
    }


def _java_contract(package_name: str, class_prefix: str, pack_id: str) -> str:
    return f"""package {package_name}.system;

import java.util.Map;
import java.util.Objects;

/**
 * Immutable server-authoritative contract generated for {pack_id}.
 */
public record {class_prefix}Contract(
        String id,
        int schemaVersion,
        Map<String, String> attributes
) {{
    public {class_prefix}Contract {{
        Objects.requireNonNull(id, \"id\");
        Objects.requireNonNull(attributes, \"attributes\");
        if (!id.matches(\"[a-z][a-z0-9_]{{1,63}}\")) {{
            throw new IllegalArgumentException(\"Invalid contract id: \" + id);
        }}
        if (schemaVersion < 1) {{
            throw new IllegalArgumentException(\"schemaVersion must be positive\");
        }}
        attributes = Map.copyOf(attributes);
    }}
}}
"""


def _java_validator(package_name: str, class_prefix: str, pack_id: str) -> str:
    return f"""package {package_name}.system;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Deterministic validation shared by generated server bindings and GameTests.
 */
public final class {class_prefix}Validator {{
    private {class_prefix}Validator() {{}}

    public static void validate(List<{class_prefix}Contract> contracts) {{
        Set<String> ids = new HashSet<>();
        for ({class_prefix}Contract contract : contracts) {{
            if (!ids.add(contract.id())) {{
                throw new IllegalArgumentException(\"Duplicate {pack_id} id: \" + contract.id());
            }}
        }}
    }}
}}
"""


def supported_system_packs() -> tuple[str, ...]:
    return tuple(sorted(_PACKS))
