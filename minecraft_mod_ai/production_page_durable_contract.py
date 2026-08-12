from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence


_VERSION = 1
_FIELD_PATCH_KEYS = frozenset({"target_fingerprint", "set_fields", "delete_fields"})
_REPLACE_PATCH_KEYS = frozenset({"target_fingerprint", "replacement"})

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
            "version": _VERSION,
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


def _parse_error(
    module: Any,
    *,
    kind: str,
    raw: Any,
    catalog: Any,
) -> tuple[Any | None, str]:
    spec = _SPECS[kind]
    parser = getattr(module, str(spec["parser"]))
    try:
        parsed = parser(raw)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    identity = str(getattr(parsed, str(spec["id_attr"]), "")).strip()
    if identity and identity in catalog:
        return None, (
            f"duplicate {spec['id_attr']} {identity!r}; change only the identity field "
            "to a new compatible unique id and preserve the rest"
        )
    return parsed, ""


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
    """Patch only invalid fields until the real production parser accepts the item."""

    from . import planner_json_runtime_contract as runtime
    from .planner_strict_json_contract import _extract_one_complete_object

    spec = _SPECS[kind]
    allowed = frozenset(spec["fields"])
    original_fingerprint = _fingerprint(raw)
    state_path = _item_state_path(page_path, kind, index, raw)
    saved = _read(state_path)

    if saved.get("version") == _VERSION and isinstance(saved.get("resolved"), dict):
        resolved_raw = dict(saved["resolved"])
        parsed, error = _parse_error(
            module,
            kind=kind,
            raw=resolved_raw,
            catalog=catalog,
        )
        if not error and parsed is not None:
            return parsed, resolved_raw

    current = saved.get("current", raw) if saved.get("version") == _VERSION else raw
    parsed, error = _parse_error(module, kind=kind, raw=current, catalog=catalog)
    if not error and parsed is not None and isinstance(current, dict):
        return parsed, dict(current)

    round_index = int(saved.get("round", 0)) if saved.get("version") == _VERSION else 0
    while True:
        round_index += 1
        field_patch = isinstance(current, dict)
        _atomic_write(
            state_path,
            {
                "version": _VERSION,
                "status": "patching",
                "kind": kind,
                "index": index,
                "target_fingerprint": original_fingerprint,
                "round": round_index,
                "current": current,
                "validation_error": error,
            },
        )

        if field_patch:
            prompt = (
                f"You are a deterministic field-level JSON patcher for one Minecraft {kind}. "
                "Return only target_fingerprint, set_fields, delete_fields. DO NOT rewrite "
                "the whole object. Change only fields required by validation_error; preserve "
                "every correct field exactly. Do not output Markdown, explanation, siblings, "
                "or a production page."
            )
            output_contract: dict[str, Any] = {
                "target_fingerprint": original_fingerprint,
                "set_fields": {"only_invalid_or_missing_fields": "corrected value"},
                "delete_fields": ["only_invalid_extra_fields"],
            }
            schema = _patch_schema(fields=sorted(allowed), replacement=False)
        else:
            prompt = (
                f"You are a deterministic JSON repairer for one Minecraft {kind}. The saved "
                "value is not an object, so return one replacement object only. Do not output "
                "a page, explanation, Markdown, or any sibling object."
            )
            output_contract = {
                "target_fingerprint": original_fingerprint,
                "replacement": {"valid_object": "for this exact production item"},
            }
            schema = _patch_schema(fields=sorted(allowed), replacement=True)

        request = {
            "item_kind": kind,
            "target_fingerprint": original_fingerprint,
            "current_value": current,
            "validation_error": error,
            "allowed_fields": sorted(allowed),
            "instruction": (
                "Repair only this saved item. Existing sibling production items are already "
                "persisted and must not be regenerated or mentioned."
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

        patch: dict[str, Any] | None = None
        try:
            patch = _extract_one_complete_object(text)
            if patch.get("target_fingerprint") != original_fingerprint:
                raise ValueError("target_fingerprint mismatch")

            if field_patch:
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
                    if field not in allowed:
                        candidate.pop(field, None)
                candidate.update(set_fields)
            else:
                if frozenset(map(str, patch)) != _REPLACE_PATCH_KEYS:
                    raise ValueError("replacement patch has invalid top-level keys")
                replacement = patch.get("replacement")
                if not isinstance(replacement, dict):
                    raise ValueError("replacement must be an object")
                candidate = dict(replacement)

            parsed, next_error = _parse_error(
                module,
                kind=kind,
                raw=candidate,
                catalog=catalog,
            )
            if next_error or parsed is None:
                raise ValueError(next_error or "item parser rejected replacement")

            _atomic_write(
                state_path,
                {
                    "version": _VERSION,
                    "status": "resolved",
                    "kind": kind,
                    "index": index,
                    "target_fingerprint": original_fingerprint,
                    "round": round_index,
                    "resolved": candidate,
                },
            )
            return parsed, candidate
        except Exception as exc:
            if field_patch and isinstance(patch, dict):
                set_fields = patch.get("set_fields")
                delete_fields = patch.get("delete_fields")
                if isinstance(set_fields, dict) and isinstance(delete_fields, list):
                    trial = dict(current)
                    for field in delete_fields:
                        if isinstance(field, str) and field not in allowed:
                            trial.pop(field, None)
                    trial.update(
                        {field: value for field, value in set_fields.items() if field in allowed}
                    )
                    current = trial
            elif not field_patch and isinstance(patch, dict):
                replacement = patch.get("replacement")
                if isinstance(replacement, dict):
                    current = dict(replacement)
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

    # Record a compact final semantic receipt. The large raw page remains immutable.
    receipt_path = page_path.with_name(page_path.name + ".resolved.json")
    _atomic_write(
        receipt_path,
        {
            "version": _VERSION,
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
