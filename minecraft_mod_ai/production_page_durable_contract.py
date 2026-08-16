from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence


_VERSION = 1
_ITEM_VERSION = 2
_FIELD_PATCH_KEYS = frozenset({"target_fingerprint", "set_fields", "delete_fields"})
_REPLACE_PATCH_KEYS = frozenset({"target_fingerprint", "replacement"})
_ID_SAFE = re.compile(r"[^a-z0-9_]+")

_SPECS = {
    "module": {
        "parser": "_module",
        "id_attr": "module_id",
        "fields": frozenset(
            {
                "module_id",
                "id",
                "name",
                "kind",
                "type",
                "config",
                "depends_on",
                "required_gates",
                "implements_deliverables",
                "implements",
            }
        ),
    },
    "asset": {
        "parser": "_asset",
        "id_attr": "asset_id",
        "fields": frozenset(
            {
                "asset_id",
                "id",
                "kind",
                "prompt",
                "description",
                "target_path",
                "width",
                "height",
                "implements_deliverables",
                "implements",
            }
        ),
    },
    "audio": {
        "parser": "_audio",
        "id_attr": "sound_id",
        "fields": frozenset(
            {
                "sound_id",
                "id",
                "kind",
                "duration_seconds",
                "frequency_hz",
                "volume",
                "loop",
                "subtitle_en",
                "subtitle_ko",
                "implements_deliverables",
                "implements",
            }
        ),
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _root() -> Path:
    explicit = os.environ.get("MMM_PLANNER_CHECKPOINT_DIR", "").strip()
    if explicit:
        base = Path(explicit).expanduser()
    elif Path("/content").is_dir():
        base = Path("/content/mmm_planner_checkpoints")
    else:
        base = Path.home() / ".cache" / "mmm" / "planner_checkpoints"
    path = base / "production_pages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(_canonical(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def page_checkpoint_path(stage: str, request: dict[str, Any]) -> Path:
    digest = _fingerprint(
        {
            "version": _VERSION,
            "stage": stage,
            "request": request,
        }
    )
    return _root() / f"page-{digest}.json"


def load_or_generate_page(
    *,
    stage: str,
    request: dict[str, Any],
    generate: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    """Persist a successful structured page before semantic item parsing begins."""

    path = page_checkpoint_path(stage, request)
    saved = _read(path)
    if saved.get("version") == _VERSION and isinstance(saved.get("page"), dict):
        return dict(saved["page"]), path

    page = generate()
    _atomic_write(
        path,
        {
            "version": _VERSION,
            "status": "raw_page_saved",
            "page": page,
        },
    )
    return page, path


def _item_state_path(page_path: Path, kind: str, index: int, raw: Any) -> Path:
    digest = _fingerprint(
        {
            "version": _ITEM_VERSION,
            "page": page_path.name,
            "kind": kind,
            "index": index,
            "raw": raw,
        }
    )
    return page_path.with_name(page_path.name + f".{kind}.{index}.{digest[:20]}.json")


def _patch_schema(*, fields: Sequence[str], replacement: bool) -> dict[str, Any]:
    if replacement:
        return {
            "type": "object",
            "properties": {
                "target_fingerprint": {"type": "string"},
                "replacement": {"type": "object", "additionalProperties": True},
            },
            "required": sorted(_REPLACE_PATCH_KEYS),
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "set_fields": {
                "type": "object",
                "properties": {field: {} for field in fields},
                "additionalProperties": False,
            },
            "delete_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": sorted(_FIELD_PATCH_KEYS),
        "additionalProperties": False,
    }


def _safe_identifier(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()
    text = _ID_SAFE.sub("_", text).strip("_")
    if not text:
        text = fallback
    if not text[0].isalpha():
        text = f"x_{text}"
    if len(text) == 1:
        text = f"{text}_1"
    return text[:64].rstrip("_") or fallback


def _unique_identifier(base: str, catalog: Any) -> str:
    if base not in catalog:
        return base
    # Host IDs are structural. Allocate deterministically, but never use an unbounded
    # search even though realistic catalogs are tiny compared with this ceiling.
    for suffix_index in range(2, 10_002):
        suffix = f"_{suffix_index}"
        candidate = f"{base[: 64 - len(suffix)].rstrip('_')}{suffix}"
        if candidate not in catalog:
            return candidate
    raise RuntimeError(f"Unable to allocate a unique production id for {base!r}")


def _deliverable_hint(raw: dict[str, Any]) -> str:
    for key in ("implements_deliverables", "implements"):
        values = raw.get(key)
        if isinstance(values, (list, tuple)):
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value
        elif isinstance(values, str) and values.strip():
            return values
    return ""


def _infer_asset_kind(raw: dict[str, Any]) -> str:
    kind = str(raw.get("kind") or "").strip().lower()
    aliases = {
        "texture": "",
        "textures": "",
    }
    kind = aliases.get(kind, kind)
    if kind:
        return kind
    path = str(raw.get("target_path") or "").replace("\\", "/").lower()
    for candidate in ("block", "entity", "gui", "environment", "icon", "item"):
        if f"/{candidate}/" in path or path.endswith(f"/{candidate}.png"):
            return candidate
    return "item"


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off", ""}:
            return False
    return value


def _deterministic_normalize(
    *,
    kind: str,
    index: int,
    raw: Any,
    catalog: Any,
) -> tuple[Any, list[str]]:
    """Repair host-owned structural facts without spending an LLM call."""

    if not isinstance(raw, dict):
        return raw, []

    value = dict(raw)
    changes: list[str] = []
    spec = _SPECS[kind]
    id_attr = str(spec["id_attr"])

    if kind == "module":
        identity_source = (
            value.get("module_id")
            or value.get("id")
            or value.get("name")
            or _deliverable_hint(value)
            or f"module_{index + 1}"
        )
        fallback = f"module_{index + 1}"
    elif kind == "asset":
        identity_source = (
            value.get("asset_id")
            or value.get("id")
            or _deliverable_hint(value)
            or f"asset_{index + 1}"
        )
        fallback = f"asset_{index + 1}"
    else:
        identity_source = (
            value.get("sound_id")
            or value.get("id")
            or _deliverable_hint(value)
            or f"sound_{index + 1}"
        )
        fallback = f"sound_{index + 1}"

    normalized_id = _safe_identifier(identity_source, fallback=fallback)
    unique_id = _unique_identifier(normalized_id, catalog)
    if value.get(id_attr) != unique_id:
        value[id_attr] = unique_id
        changes.append(f"{id_attr}:normalized_unique")

    if kind == "module":
        # An omitted config has one unambiguous host representation. Do not waste a
        # model call asking it to return an empty object. A present non-object config,
        # however, may carry intended semantics and is left for semantic repair.
        if "config" not in value:
            value["config"] = {}
            changes.append("config:defaulted_empty")
    elif kind == "asset":
        inferred_kind = _infer_asset_kind(value)
        if value.get("kind") != inferred_kind:
            value["kind"] = inferred_kind
            changes.append("kind:asset_normalized")
        if not value.get("prompt"):
            description = value.get("description")
            value["prompt"] = (
                str(description).strip()
                if description
                else f"Asset for {unique_id}"
            )
            changes.append("prompt:defaulted")
        if not value.get("target_path"):
            value["target_path"] = f"assets/mod/textures/{unique_id}.png"
            changes.append("target_path:defaulted")
    elif kind == "audio":
        raw_kind = str(value.get("kind") or "").strip().lower()
        normalized_kind = {"sfx": "effect", "sound": "effect"}.get(
            raw_kind,
            raw_kind or "effect",
        )
        if value.get("kind") != normalized_kind:
            value["kind"] = normalized_kind
            changes.append("kind:audio_normalized")
        if "loop" in value:
            normalized_loop = _normalize_bool(value["loop"])
            if normalized_loop != value["loop"]:
                value["loop"] = normalized_loop
                changes.append("loop:boolean_normalized")

    return value, changes


def _parse_error(
    module: Any,
    *,
    kind: str,
    raw: Any,
    catalog: Any,
) -> tuple[Any | None, str]:
    """Run the real parser and the real item validator, then check host identity."""

    spec = _SPECS[kind]
    parser = getattr(module, str(spec["parser"]))
    try:
        parsed = parser(raw)
        validate = getattr(parsed, "validate", None)
        if callable(validate):
            validate()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    identity = str(getattr(parsed, str(spec["id_attr"]), "")).strip()
    if not identity:
        return None, f"empty {spec['id_attr']} after parsing"
    if identity in catalog:
        return None, f"duplicate {spec['id_attr']} {identity!r}"
    return parsed, ""


def _write_resolved(
    state_path: Path,
    *,
    kind: str,
    index: int,
    original_fingerprint: str,
    round_index: int,
    resolved: dict[str, Any],
    repair_method: str,
    deterministic_changes: Sequence[str] = (),
) -> None:
    _atomic_write(
        state_path,
        {
            "version": _ITEM_VERSION,
            "status": "resolved",
            "kind": kind,
            "index": index,
            "target_fingerprint": original_fingerprint,
            "round": round_index,
            "repair_method": repair_method,
            "deterministic_changes": list(deterministic_changes),
            "resolved": resolved,
        },
    )


def _raise_repair_failure(
    module: Any,
    *,
    kind: str,
    index: int,
    state_path: Path,
    original_fingerprint: str,
    current: Any,
    error: str,
    round_index: int,
    reason: str,
    last_patch_sha256: str = "",
) -> None:
    _atomic_write(
        state_path,
        {
            "version": _ITEM_VERSION,
            "status": "failed",
            "kind": kind,
            "index": index,
            "target_fingerprint": original_fingerprint,
            "round": round_index,
            "reason": reason,
            "current": current,
            "validation_error": error,
            "last_patch_sha256": last_patch_sha256,
        },
    )
    error_type = getattr(module, "SpecValidationError", ValueError)
    raise error_type(
        f"Production {kind}[{index}] repair failed safely ({reason}): {error}"
    )


def _patch_one_item(
    module: Any,
    router: Any,
    *,
    kind: str,
    index: int,
    raw: Any,
    catalog: Any,
    page_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Resolve one production item with progress-driven semantic repair."""

    from . import planner_json_runtime_contract as runtime
    from .planner_strict_json_contract import _extract_one_complete_object

    spec = _SPECS[kind]
    allowed = frozenset(spec["fields"])
    original_fingerprint = _fingerprint(raw)
    state_path = _item_state_path(page_path, kind, index, raw)
    saved = _read(state_path)

    if saved.get("version") == _ITEM_VERSION and isinstance(saved.get("resolved"), dict):
        resolved_raw, deterministic_changes = _deterministic_normalize(
            kind=kind,
            index=index,
            raw=dict(saved["resolved"]),
            catalog=catalog,
        )
        parsed, error = _parse_error(
            module,
            kind=kind,
            raw=resolved_raw,
            catalog=catalog,
        )
        if not error and parsed is not None and isinstance(resolved_raw, dict):
            if deterministic_changes:
                _write_resolved(
                    state_path,
                    kind=kind,
                    index=index,
                    original_fingerprint=original_fingerprint,
                    round_index=int(saved.get("round", 0)),
                    resolved=resolved_raw,
                    repair_method="resume_deterministic",
                    deterministic_changes=deterministic_changes,
                )
            return parsed, dict(resolved_raw)

    current_source = (
        saved.get("current", raw)
        if saved.get("version") == _ITEM_VERSION
        else raw
    )
    current, deterministic_changes = _deterministic_normalize(
        kind=kind,
        index=index,
        raw=current_source,
        catalog=catalog,
    )
    parsed, error = _parse_error(module, kind=kind, raw=current, catalog=catalog)
    if not error and parsed is not None and isinstance(current, dict):
        _write_resolved(
            state_path,
            kind=kind,
            index=index,
            original_fingerprint=original_fingerprint,
            round_index=0,
            resolved=dict(current),
            repair_method="deterministic" if deterministic_changes else "none",
            deterministic_changes=deterministic_changes,
        )
        return parsed, dict(current)

    seen_states: set[str] = set()
    seen_patch_hashes: set[str] = set()
    round_index = 0
    last_patch_sha256 = ""

    _MAX_ITEM_REPAIR_ROUNDS = 3
    while True:
        round_index += 1
        if round_index > _MAX_ITEM_REPAIR_ROUNDS:
            # Exhausted model repair rounds. Attempt aggressive deterministic normalization
            candidate, _ = _deterministic_normalize(
                kind=kind,
                index=index,
                raw=current if isinstance(current, dict) else {},
                catalog=catalog,
            )
            parsed, fallback_err = _parse_error(module, kind=kind, raw=candidate, catalog=catalog)
            if not fallback_err and parsed is not None:
                _write_resolved(
                    state_path,
                    kind=kind,
                    index=index,
                    original_fingerprint=original_fingerprint,
                    round_index=round_index - 1,
                    resolved=candidate,
                    repair_method="deterministic_fallback",
                    deterministic_changes=["exhausted_repair_fallback"],
                )
                return parsed, candidate
            _raise_repair_failure(
                module,
                kind=kind,
                index=index,
                state_path=state_path,
                original_fingerprint=original_fingerprint,
                current=current,
                error=error,
                round_index=round_index - 1,
                reason="exhausted_repair_rounds",
                last_patch_sha256=last_patch_sha256,
            )

        # Mapping-shaped items can always be repaired field-by-field. Whole-object
        # regeneration is reserved for non-object values that cannot be field patched.
        replacement_mode = not isinstance(current, dict)
        repair_mode = "replacement" if replacement_mode else "field_patch"
        state_fingerprint = _fingerprint(
            {"current": current, "error": error, "repair_mode": repair_mode}
        )
        if state_fingerprint in seen_states:
            _raise_repair_failure(
                module,
                kind=kind,
                index=index,
                state_path=state_path,
                original_fingerprint=original_fingerprint,
                current=current,
                error=error,
                round_index=max(0, round_index - 1),
                reason="repeated_validation_state",
                last_patch_sha256=last_patch_sha256,
            )
        seen_states.add(state_fingerprint)

        _atomic_write(
            state_path,
            {
                "version": _ITEM_VERSION,
                "status": "repairing",
                "kind": kind,
                "index": index,
                "target_fingerprint": original_fingerprint,
                "round": round_index,
                "repair_mode": repair_mode,
                "current": current,
                "validation_error": error,
            },
        )

        if not replacement_mode:
            prompt = (
                f"You repair exactly one Minecraft {kind} production object. "
                "Return only target_fingerprint, set_fields, delete_fields. Change only "
                "fields required by validation_error and preserve every other field exactly. "
                "Do not output Markdown, explanation, siblings, or a production page."
            )
            output_contract: dict[str, Any] = {
                "target_fingerprint": original_fingerprint,
                "set_fields": {"only_invalid_or_missing_fields": "corrected value"},
                "delete_fields": ["only_invalid_extra_fields"],
            }
            schema = _patch_schema(fields=sorted(allowed), replacement=False)
        else:
            prompt = (
                f"You regenerate exactly one invalid Minecraft {kind} production object. "
                "Return only target_fingerprint and replacement. Preserve the current item's "
                "purpose and implements_deliverables, but produce a complete object that fixes "
                "validation_error. Do not output Markdown, explanation, siblings, or a page."
            )
            output_contract = {
                "target_fingerprint": original_fingerprint,
                "replacement": {"complete_valid_object": "for this exact production item"},
            }
            schema = _patch_schema(fields=sorted(allowed), replacement=True)

        request = {
            "item_kind": kind,
            "target_fingerprint": original_fingerprint,
            "repair_mode": repair_mode,
            "current_value": current,
            "validation_error": error,
            "allowed_fields": sorted(allowed),
            "instruction": (
                "Repair only this item. Sibling production items are already persisted "
                "and must not be regenerated or mentioned."
            ),
            "output_contract": output_contract,
        }

        token = runtime._JSON_SCHEMA.set(schema)
        try:
            text = router.generate_text(
                "planner",
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": _canonical(request)},
                ],
                media_paths=(),
                response_format="json",
            )
        finally:
            runtime._JSON_SCHEMA.reset(token)

        last_patch_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if last_patch_sha256 in seen_patch_hashes:
            _raise_repair_failure(
                module,
                kind=kind,
                index=index,
                state_path=state_path,
                original_fingerprint=original_fingerprint,
                current=current,
                error=error,
                round_index=round_index,
                reason="repeated_model_output",
                last_patch_sha256=last_patch_sha256,
            )
        seen_patch_hashes.add(last_patch_sha256)

        patch: dict[str, Any] | None = None
        candidate: Any = current
        try:
            patch = _extract_one_complete_object(text)
            if patch.get("target_fingerprint") != original_fingerprint:
                raise ValueError("target_fingerprint mismatch")

            if not replacement_mode:
                if frozenset(map(str, patch)) != _FIELD_PATCH_KEYS:
                    raise ValueError("field patch has invalid top-level keys")
                set_fields = patch.get("set_fields")
                delete_fields = patch.get("delete_fields")
                if not isinstance(set_fields, dict):
                    raise ValueError("set_fields must be an object")
                if not isinstance(delete_fields, list) or any(
                    not isinstance(value, str) for value in delete_fields
                ):
                    raise ValueError("delete_fields must be an array of strings")
                if any(field not in allowed for field in set_fields):
                    raise ValueError("set_fields contains a field outside this item contract")
                candidate = dict(current)
                for field in delete_fields:
                    if field in allowed:
                        candidate.pop(field, None)
                candidate.update(set_fields)
            else:
                if frozenset(map(str, patch)) != _REPLACE_PATCH_KEYS:
                    raise ValueError("replacement patch has invalid top-level keys")
                replacement = patch.get("replacement")
                if not isinstance(replacement, dict):
                    raise ValueError("replacement must be an object")
                candidate = dict(replacement)

            candidate, deterministic_after_model = _deterministic_normalize(
                kind=kind,
                index=index,
                raw=candidate,
                catalog=catalog,
            )
            parsed, next_error = _parse_error(
                module,
                kind=kind,
                raw=candidate,
                catalog=catalog,
            )
            if next_error or parsed is None:
                error = next_error or "item validator rejected candidate"
                current = candidate
                continue

            if not isinstance(candidate, dict):
                raise ValueError("validated candidate is not an object")
            _write_resolved(
                state_path,
                kind=kind,
                index=index,
                original_fingerprint=original_fingerprint,
                round_index=round_index,
                resolved=candidate,
                repair_method=(
                    "model_replacement" if replacement_mode else "model_field_patch"
                ),
                deterministic_changes=deterministic_after_model,
            )
            return parsed, candidate
        except Exception as exc:
            if isinstance(candidate, dict):
                current = candidate
            error = f"{type(exc).__name__}: {exc}"



def resolve_page_items(
    module: Any,
    router: Any,
    *,
    page: dict[str, Any],
    page_path: Path,
    module_catalog: Any,
    asset_catalog: Any,
    audio_catalog: Any,
    test_catalog: set[str],
) -> tuple[list[Any], list[Any], list[Any], list[str]]:
    """Validate/persist every production item independently; never drop bad siblings."""

    outputs: dict[str, list[Any]] = {"module": [], "asset": [], "audio": []}
    corrected_raw: dict[str, list[dict[str, Any]]] = {
        "module": [],
        "asset": [],
        "audio": [],
    }
    catalogs = {
        "module": module_catalog,
        "asset": asset_catalog,
        "audio": audio_catalog,
    }
    source_fields = {"module": "modules", "asset": "assets", "audio": "audio"}

    for kind in ("module", "asset", "audio"):
        raw_values = page.get(source_fields[kind], [])
        if not isinstance(raw_values, list):
            raw_values = []
        for index, raw in enumerate(raw_values):
            parsed, repaired_raw = _patch_one_item(
                module,
                router,
                kind=kind,
                index=index,
                raw=raw,
                catalog=catalogs[kind],
                page_path=page_path,
            )
            identity = str(getattr(parsed, str(_SPECS[kind]["id_attr"]), "")).strip()
            catalogs[kind].add(identity)
            outputs[kind].append(parsed)
            corrected_raw[kind].append(repaired_raw)

    tests_raw = page.get("acceptance_tests", [])
    tests = [
        value.strip()
        for value in tests_raw
        if isinstance(value, str) and value.strip() and value.strip() not in test_catalog
    ] if isinstance(tests_raw, list) else []

    receipt_path = page_path.with_name(page_path.name + ".resolved.json")
    _atomic_write(
        receipt_path,
        {
            "version": _VERSION,
            "item_contract_version": _ITEM_VERSION,
            "status": "resolved",
            "module_ids": [value.module_id for value in outputs["module"]],
            "asset_ids": [value.asset_id for value in outputs["asset"]],
            "audio_ids": [value.sound_id for value in outputs["audio"]],
            "acceptance_tests": tests,
            "corrected_raw_sha256": _fingerprint(corrected_raw),
        },
    )
    return outputs["module"], outputs["asset"], outputs["audio"], tests


__all__ = ["load_or_generate_page", "page_checkpoint_path", "resolve_page_items"]
