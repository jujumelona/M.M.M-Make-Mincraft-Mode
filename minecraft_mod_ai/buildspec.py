from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILD_SPEC_VERSION = "buildspec_v2"

_TOP_KEYS = {
    "schema_version", "world", "zones", "components", "parts", "relations",
    "ports", "patterns", "operators", "task", "constraints",
}
_WORLD_KEYS = {
    "origin", "bbox", "context_blocks_ref", "terrain_ref", "protected_mask_ref",
}
_TASK_KEYS = {
    "type", "target_component_ids", "completed_component_ids", "open_port_ids",
}
_RESULT_KEYS = {
    "add_blocks_ref", "remove_blocks_ref", "replace_blocks_ref",
    "resolved_ports", "remaining_open_ports", "validation_predictions",
}
_PREDICTION_KEYS = {"supported", "connected", "constraint_violations"}
_TASK_TYPES = {
    "generate", "continue", "repair", "partial_complete",
    "extend_boundary", "connect_ports",
}
_FORBIDDEN_KEYS = {
    "brief", "caption", "description", "free_text", "image_caption",
    "function", "instruction", "label", "meaning", "name", "natural_language",
    "notes", "prompt", "scene_meaning", "style", "style_description", "text",
    "visual_identity",
}
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_./:#@+\-]{1,256}$")


class BuildSpecValidationError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def payload_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def buildspec_contract() -> dict[str, Any]:
    return {
        "schema_version": BUILD_SPEC_VERSION,
        "input_required_keys": sorted(_TOP_KEYS),
        "world_required_keys": sorted(_WORLD_KEYS),
        "task_required_keys": sorted(_TASK_KEYS),
        "output_required_keys": sorted(_RESULT_KEYS),
        "natural_language_fields_forbidden": sorted(_FORBIDDEN_KEYS),
        "boundary": "CENTRAL_AGENT_INTERPRETS_BUILDER_EXECUTES",
    }


def validate_world(world: Mapping[str, Any]) -> dict[str, Any]:
    value = _object(world, "world")
    _exact_keys(value, _WORLD_KEYS, "world")
    origin = _int_vector(value["origin"], 3, "world.origin")
    bbox = _int_vector(value["bbox"], 6, "world.bbox")
    if not (bbox[0] < bbox[3] and bbox[1] < bbox[4] and bbox[2] < bbox[5]):
        raise BuildSpecValidationError(
            "world.bbox must be [min_x,min_y,min_z,max_x,max_y,max_z]."
        )
    return {
        "origin": origin,
        "bbox": bbox,
        "context_blocks_ref": _npz(value["context_blocks_ref"], "world.context_blocks_ref"),
        "terrain_ref": _npz(value["terrain_ref"], "world.terrain_ref"),
        "protected_mask_ref": _npz(value["protected_mask_ref"], "world.protected_mask_ref"),
    }


def validate_buildspec(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = _object(spec, "buildspec")
    _exact_keys(value, _TOP_KEYS, "buildspec")
    if value["schema_version"] != BUILD_SPEC_VERSION:
        raise BuildSpecValidationError(
            f"schema_version must be {BUILD_SPEC_VERSION!r}."
        )
    normalized: dict[str, Any] = {
        "schema_version": BUILD_SPEC_VERSION,
        "world": validate_world(value["world"]),
    }
    id_fields = {
        "zones": ("zone_id", False),
        "components": ("component_id", True),
        "parts": ("part_id", True),
        "relations": ("relation_id", False),
        "ports": ("port_id", True),
        "patterns": ("pattern_id", False),
        "operators": ("operator_id", False),
    }
    for section, (id_field, required) in id_fields.items():
        normalized[section] = _records(
            value[section], section, id_field, required
        )

    component_ids = _record_ids(normalized["components"], "component_id")
    part_ids = _record_ids(normalized["parts"], "part_id")
    port_ids = _record_ids(normalized["ports"], "port_id")
    for index, part in enumerate(normalized["parts"]):
        if (
            "component_id" in part
            and part["component_id"] not in component_ids
        ):
            raise BuildSpecValidationError(
                f"parts[{index}].component_id is unknown."
            )
    for index, port in enumerate(normalized["ports"]):
        if (
            "component_id" in port
            and port["component_id"] not in component_ids
        ):
            raise BuildSpecValidationError(
                f"ports[{index}].component_id is unknown."
            )
        if "part_id" in port and port["part_id"] not in part_ids:
            raise BuildSpecValidationError(
                f"ports[{index}].part_id is unknown."
            )

    task = _object(value["task"], "task")
    _exact_keys(task, _TASK_KEYS, "task")
    if task["type"] not in _TASK_TYPES:
        raise BuildSpecValidationError(
            f"task.type must be one of {sorted(_TASK_TYPES)}."
        )
    target = _id_list(task["target_component_ids"], "task.target_component_ids")
    completed = _id_list(
        task["completed_component_ids"], "task.completed_component_ids"
    )
    open_ports = _id_list(task["open_port_ids"], "task.open_port_ids")
    if not set(target).issubset(component_ids):
        raise BuildSpecValidationError(
            "task.target_component_ids contains an unknown component."
        )
    if not set(completed).issubset(set(target)):
        raise BuildSpecValidationError(
            "task.completed_component_ids must be a subset of target_component_ids."
        )
    if not set(open_ports).issubset(port_ids):
        raise BuildSpecValidationError(
            "task.open_port_ids contains an unknown port."
        )
    normalized["task"] = {
        "type": task["type"],
        "target_component_ids": target,
        "completed_component_ids": completed,
        "open_port_ids": open_ports,
    }

    constraints = _object(value["constraints"], "constraints")
    _exact_keys(constraints, {"hard", "soft"}, "constraints")
    normalized["constraints"] = {
        "hard": _records(
            constraints["hard"], "constraints.hard", "constraint_id", False
        ),
        "soft": _records(
            constraints["soft"], "constraints.soft", "constraint_id", False
        ),
    }
    _machine_json(normalized, "buildspec")
    return normalized


def validate_builder_result(
    buildspec: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    spec = validate_buildspec(buildspec)
    value = _object(result, "builder_result")
    _exact_keys(value, _RESULT_KEYS, "builder_result")
    predictions = _object(
        value["validation_predictions"],
        "builder_result.validation_predictions",
    )
    _exact_keys(
        predictions,
        _PREDICTION_KEYS,
        "builder_result.validation_predictions",
    )
    if type(predictions["supported"]) is not bool:
        raise BuildSpecValidationError(
            "validation_predictions.supported must be boolean."
        )
    if type(predictions["connected"]) is not bool:
        raise BuildSpecValidationError(
            "validation_predictions.connected must be boolean."
        )
    resolved = _id_list(value["resolved_ports"], "resolved_ports")
    remaining = _id_list(
        value["remaining_open_ports"], "remaining_open_ports"
    )
    if set(resolved) & set(remaining):
        raise BuildSpecValidationError(
            "resolved_ports and remaining_open_ports must be disjoint."
        )
    if set(resolved) | set(remaining) != set(spec["task"]["open_port_ids"]):
        raise BuildSpecValidationError(
            "Builder output ports must exactly partition task.open_port_ids."
        )
    normalized = {
        "add_blocks_ref": _npz(value["add_blocks_ref"], "add_blocks_ref"),
        "remove_blocks_ref": _npz(value["remove_blocks_ref"], "remove_blocks_ref"),
        "replace_blocks_ref": _npz(value["replace_blocks_ref"], "replace_blocks_ref"),
        "resolved_ports": resolved,
        "remaining_open_ports": remaining,
        "validation_predictions": {
            "supported": predictions["supported"],
            "connected": predictions["connected"],
            "constraint_violations": _records(
                predictions["constraint_violations"],
                "validation_predictions.constraint_violations",
                "violation_id",
                False,
            ),
        },
    }
    _machine_json(normalized, "builder_result")
    return normalized


def referenced_world_npz(buildspec: Mapping[str, Any]) -> tuple[str, ...]:
    world = validate_buildspec(buildspec)["world"]
    return (
        world["context_blocks_ref"],
        world["terrain_ref"],
        world["protected_mask_ref"],
    )


def referenced_result_npz(result: Mapping[str, Any]) -> tuple[str, ...]:
    value = _object(result, "builder_result")
    return tuple(
        _npz(value[key], key)
        for key in ("add_blocks_ref", "remove_blocks_ref", "replace_blocks_ref")
    )


def verify_artifacts(
    root: str | Path,
    refs: Iterable[str],
) -> list[dict[str, str]]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir() or base.is_symlink():
        raise BuildSpecValidationError(f"Invalid artifact root: {base}")
    receipts: list[dict[str, str]] = []
    for ref in refs:
        target = (base / ref).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise BuildSpecValidationError(
                f"Artifact reference escaped root: {ref}"
            ) from exc
        if not target.is_file() or target.is_symlink():
            raise BuildSpecValidationError(
                f"Referenced artifact is missing: {ref}"
            )
        digest = hashlib.sha256()
        with target.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        receipts.append(
            {"ref": ref, "sha256": "sha256:" + digest.hexdigest()}
        )
    return receipts


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BuildSpecValidationError(f"{path} must be an object.")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise BuildSpecValidationError(
            f"{path} keys invalid; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def _int_vector(value: Any, length: int, path: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != length
        or any(type(item) is not int for item in value)
    ):
        raise BuildSpecValidationError(
            f"{path} must contain exactly {length} integers."
        )
    return list(value)


def _npz(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or ".." in Path(value).parts
        or not value.lower().endswith(".npz")
        or any(character.isspace() for character in value)
    ):
        raise BuildSpecValidationError(
            f"{path} must be a safe relative .npz reference."
        )
    return value


def _records(
    value: Any,
    path: str,
    id_field: str,
    id_required: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BuildSpecValidationError(f"{path} must be a list.")
    result: list[dict[str, Any]] = []
    seen: set[int | str] = set()
    for index, item in enumerate(value):
        record = _object(item, f"{path}[{index}]")
        if id_required and id_field not in record:
            raise BuildSpecValidationError(
                f"{path}[{index}] requires {id_field}."
            )
        if id_field in record:
            identifier = _machine_id(
                record[id_field], f"{path}[{index}].{id_field}"
            )
            if identifier in seen:
                raise BuildSpecValidationError(
                    f"{path} contains duplicate {id_field}: {identifier!r}."
                )
            record[id_field] = identifier
            seen.add(identifier)
        _machine_json(record, f"{path}[{index}]")
        result.append(record)
    return result


def _machine_id(value: Any, path: str) -> int | str:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and _TOKEN.fullmatch(value):
        return value
    raise BuildSpecValidationError(
        f"{path} must be a non-negative integer or machine token."
    )


def _id_list(value: Any, path: str) -> list[int | str]:
    if not isinstance(value, list):
        raise BuildSpecValidationError(f"{path} must be a list.")
    result = [
        _machine_id(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise BuildSpecValidationError(f"{path} contains duplicates.")
    return result


def _record_ids(
    records: Sequence[Mapping[str, Any]],
    id_field: str,
) -> set[int | str]:
    return {
        record[id_field]
        for record in records
        if id_field in record
    }


def _machine_json(value: Any, path: str) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BuildSpecValidationError(f"{path} contains non-finite number.")
        return
    if isinstance(value, str):
        if not _TOKEN.fullmatch(value):
            raise BuildSpecValidationError(
                f"{path} contains free-form text: {value!r}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _machine_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not _KEY.fullmatch(key):
                raise BuildSpecValidationError(
                    f"{path} contains invalid key: {key!r}"
                )
            if key in _FORBIDDEN_KEYS:
                raise BuildSpecValidationError(
                    f"{path}.{key} is a forbidden natural-language field."
                )
            _machine_json(item, f"{path}.{key}")
        return
    raise BuildSpecValidationError(
        f"{path} contains unsupported type {type(value).__name__}."
    )


__all__ = [
    "BUILD_SPEC_VERSION",
    "BuildSpecValidationError",
    "buildspec_contract",
    "canonical_json",
    "payload_sha256",
    "referenced_result_npz",
    "referenced_world_npz",
    "validate_builder_result",
    "validate_buildspec",
    "validate_world",
    "verify_artifacts",
]
