from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Sequence

from .planner_outline_prompt_contract import _SCALABLE_OUTLINE_PROMPT
from .planner_strict_json_contract import (
    _extract_one_complete_object,
    _extract_outline_sequence,
    _outermost_complete_json_containers,
)


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})
_BATCH_FIELDS = frozenset(
    {"batch_id", "scope", "depends_on_batches", "deliverables", "exports"}
)
_FIELD_PATCH_FIELDS = frozenset({"target_fingerprint", "set_fields", "delete_fields"})
_REPLACE_PATCH_FIELDS = frozenset({"target_fingerprint", "replacement_batch"})
_CHECKPOINT_VERSION = 2


def _outline_allowed(expected_contracts: Sequence[frozenset[str]]) -> bool:
    return _OUTLINE_FIELDS in tuple(expected_contracts)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _batch_identity(value: Any) -> str:
    if isinstance(value, dict):
        batch_id = str(value.get("batch_id", "")).strip()
        if batch_id:
            return "id:" + batch_id
    return "sha256:" + _fingerprint(value)


def _checkpoint_root() -> Path:
    explicit = os.environ.get("MMM_PLANNER_CHECKPOINT_DIR", "").strip()
    if explicit:
        root = Path(explicit).expanduser()
    elif Path("/content").is_dir():
        root = Path("/content/mmm_planner_checkpoints")
    else:
        root = Path.home() / ".cache" / "mmm" / "planner_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _checkpoint_path(stage: str, request: dict[str, Any] | str) -> Path:
    digest = _fingerprint({"stage": stage, "request": request})
    safe_stage = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in stage
    ).strip("_")[:60] or "planner"
    return _checkpoint_root() / f"{safe_stage}-{digest[:20]}.json"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(value, dict) or value.get("version") != _CHECKPOINT_VERSION:
        return {}
    return value


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    payload = {"version": _CHECKPOINT_VERSION, **state}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _merge_saved_batches(saved: list[Any], incoming: Sequence[Any]) -> None:
    known = {_batch_identity(value) for value in saved}
    for value in incoming:
        identity = _batch_identity(value)
        if identity in known:
            continue
        saved.append(value)
        known.add(identity)


def _all_complete_dicts(text: str) -> list[dict[str, Any]]:
    """Return every complete dict embedded in text, including children of a cut page."""

    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, relative_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        span = (start, start + int(relative_end))
        if span in seen:
            continue
        seen.add(span)
        values.append(dict(value))
    return values


def _batch_candidates_from_text(text: str) -> list[dict[str, Any]]:
    """Salvage complete batch objects even when the surrounding page was truncated."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in _all_complete_dicts(text):
        fields = frozenset(str(key) for key in value)
        if not fields.intersection(_BATCH_FIELDS):
            continue
        # Avoid treating page envelopes or arbitrary config objects as batches.
        if "batch_id" not in value and not fields.issubset(_BATCH_FIELDS):
            continue
        identity = _fingerprint(value)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(value)
    return result


def _field_patch_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "set_fields": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string"},
                    "scope": {"type": "string"},
                    "depends_on_batches": string_array,
                    "deliverables": string_array,
                    "exports": string_array,
                },
                "additionalProperties": False,
            },
            "delete_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": sorted(_FIELD_PATCH_FIELDS),
        "additionalProperties": False,
    }


def _replacement_patch_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "replacement_batch": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string"},
                    "scope": {"type": "string"},
                    "depends_on_batches": string_array,
                    "deliverables": string_array,
                    "exports": string_array,
                },
                "required": sorted(_BATCH_FIELDS),
                "additionalProperties": False,
            },
        },
        "required": sorted(_REPLACE_PATCH_FIELDS),
        "additionalProperties": False,
    }


def _batch_validation_error(module: Any, raw: Any) -> str:
    try:
        module._production_batch(raw)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def _accepted_batch_ids(saved_batches: Sequence[Any]) -> list[str]:
    return [
        str(value.get("batch_id", "")).strip()
        for value in saved_batches
        if isinstance(value, dict) and str(value.get("batch_id", "")).strip()
    ]


def _patch_one_invalid_batch(
    runtime_module: Any,
    module: Any,
    router: Any,
    *,
    raw_batch: Any,
    validation_error: str,
    accepted_batch_ids: Sequence[str],
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    """Keep patching only the invalid object/fields until the real validator accepts it."""

    original_fingerprint = _fingerprint(raw_batch)
    pending = checkpoint_state.get("pending_patch")
    if (
        isinstance(pending, dict)
        and pending.get("target_fingerprint") == original_fingerprint
    ):
        current_value = pending.get("current_value", raw_batch)
        current_error = str(pending.get("validation_error", validation_error))
        patch_round = int(pending.get("round", 0))
    else:
        current_value = raw_batch
        current_error = validation_error
        patch_round = 0

    while True:
        patch_round += 1
        field_patch = isinstance(current_value, dict)
        checkpoint_state["status"] = "patching"
        checkpoint_state["pending_patch"] = {
            "target_fingerprint": original_fingerprint,
            "round": patch_round,
            "current_value": current_value,
            "validation_error": current_error,
        }
        _save_checkpoint(checkpoint_path, checkpoint_state)

        if field_patch:
            output_contract = {
                "target_fingerprint": original_fingerprint,
                "set_fields": {"only_fields_that_must_change": "corrected value"},
                "delete_fields": ["only_invalid_extra_field_names"],
            }
            prompt = (
                "You are a deterministic field-level JSON patcher. Fix exactly ONE "
                "invalid production batch. Return only target_fingerprint, set_fields, "
                "and delete_fields. DO NOT rewrite the whole batch. Put in set_fields "
                "only fields that must change or are missing. Put in delete_fields only "
                "invalid extra fields. Preserve every other field byte-for-byte in host "
                "state. Do not output Markdown, explanation, a page, or another batch."
            )
            schema = _field_patch_schema()
        else:
            output_contract = {
                "target_fingerprint": original_fingerprint,
                "replacement_batch": {
                    "batch_id": "string",
                    "scope": "string",
                    "depends_on_batches": ["string"],
                    "deliverables": ["string"],
                    "exports": ["string"],
                },
            }
            prompt = (
                "You are a deterministic JSON object repairer. The invalid value is not "
                "a batch object, so return exactly one replacement batch object. Do not "
                "generate a page, plan, explanation, Markdown, or any other batch."
            )
            schema = _replacement_patch_schema()

        patch_request = {
            "target_fingerprint": original_fingerprint,
            "current_batch": current_value,
            "validation_error": current_error,
            "accepted_batch_ids": list(accepted_batch_ids),
            "required_fields": sorted(_BATCH_FIELDS),
            "rules": {
                "batch_id": "non-empty descriptive snake_case string",
                "scope": "non-empty string",
                "depends_on_batches": "unique non-empty strings; no self dependency",
                "deliverables": "NON-EMPTY array of unique non-empty strings",
                "exports": "array of unique non-empty snake_case strings",
            },
            "output_contract": output_contract,
        }

        token = runtime_module._JSON_SCHEMA.set(schema)
        try:
            text = router.generate_text(
                "planner",
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(patch_request, ensure_ascii=False),
                    },
                ],
                media_paths=(),
                response_format="json",
            )
        finally:
            runtime_module._JSON_SCHEMA.reset(token)

        patch: dict[str, Any] | None = None
        try:
            patch = _extract_one_complete_object(text)
            if patch.get("target_fingerprint") != original_fingerprint:
                raise ValueError("target_fingerprint does not match the saved invalid batch")

            if field_patch:
                if frozenset(str(key) for key in patch) != _FIELD_PATCH_FIELDS:
                    raise ValueError(
                        f"field patch keys must be {sorted(_FIELD_PATCH_FIELDS)}"
                    )
                set_fields = patch.get("set_fields")
                delete_fields = patch.get("delete_fields")
                if not isinstance(set_fields, dict):
                    raise ValueError("set_fields must be an object")
                if not isinstance(delete_fields, list) or any(
                    not isinstance(value, str) for value in delete_fields
                ):
                    raise ValueError("delete_fields must be an array of field names")
                if any(key not in _BATCH_FIELDS for key in set_fields):
                    raise ValueError("set_fields contains a non-batch field")
                if any(key in _BATCH_FIELDS for key in delete_fields):
                    raise ValueError("required batch fields may not be deleted")
                if not set_fields and not delete_fields:
                    raise ValueError("patch must change at least one invalid field")

                candidate = dict(current_value)
                for field in delete_fields:
                    candidate.pop(field, None)
                candidate.update(set_fields)
            else:
                if frozenset(str(key) for key in patch) != _REPLACE_PATCH_FIELDS:
                    raise ValueError(
                        f"replacement patch keys must be {sorted(_REPLACE_PATCH_FIELDS)}"
                    )
                replacement = patch.get("replacement_batch")
                if not isinstance(replacement, dict):
                    raise ValueError("replacement_batch must be an object")
                candidate = dict(replacement)

            error = _batch_validation_error(module, candidate)
            if error:
                raise ValueError(error)

            checkpoint_state["pending_patch"] = None
            checkpoint_state["status"] = "collecting"
            _save_checkpoint(checkpoint_path, checkpoint_state)
            return candidate
        except Exception as exc:
            # Patch the patch next round. Accepted plan data is never touched.
            if field_patch and isinstance(patch, dict):
                set_fields = patch.get("set_fields")
                delete_fields = patch.get("delete_fields")
                if isinstance(set_fields, dict) and isinstance(delete_fields, list):
                    trial = dict(current_value)
                    for field in delete_fields:
                        if isinstance(field, str) and field not in _BATCH_FIELDS:
                            trial.pop(field, None)
                    trial.update(
                        {
                            key: value
                            for key, value in set_fields.items()
                            if key in _BATCH_FIELDS
                        }
                    )
                    current_value = trial
            elif not field_patch and isinstance(patch, dict):
                replacement = patch.get("replacement_batch")
                if isinstance(replacement, dict):
                    current_value = replacement
            current_error = f"{type(exc).__name__}: {exc}"


def _process_pending_batches(
    runtime_module: Any,
    module: Any,
    router: Any,
    *,
    saved_batches: list[Any],
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> None:
    pending = list(checkpoint_state.get("pending_batches", []))
    while pending:
        raw = pending[0]
        error = _batch_validation_error(module, raw)
        if error:
            resolved = _patch_one_invalid_batch(
                runtime_module,
                module,
                router,
                raw_batch=raw,
                validation_error=error,
                accepted_batch_ids=_accepted_batch_ids(saved_batches),
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )
        elif isinstance(raw, dict):
            resolved = dict(raw)
        else:
            resolved = _patch_one_invalid_batch(
                runtime_module,
                module,
                router,
                raw_batch=raw,
                validation_error="batch must be a JSON object",
                accepted_batch_ids=_accepted_batch_ids(saved_batches),
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )

        _merge_saved_batches(saved_batches, [resolved])
        pending.pop(0)
        checkpoint_state.update(
            {
                "saved_batches": saved_batches,
                "pending_batches": pending,
                "pending_patch": None,
                "status": "collecting",
            }
        )
        _save_checkpoint(checkpoint_path, checkpoint_state)


def _resume_result(checkpoint_state: dict[str, Any], saved_batches: list[Any]) -> dict[str, Any]:
    return {
        "production_batches": saved_batches,
        "complete": bool(checkpoint_state.get("page_complete", False)),
        "next_cursor": str(checkpoint_state.get("page_next_cursor", "")),
    }


def install(runtime_module: Any) -> None:
    """Persist valid planner work and patch only invalid fields/objects in place."""

    from . import complete_planner as complete_planner_module

    # Normal page size belongs to AI; host must not impose a 2-target/1-target width.
    def preserve_requested_production_width(
        request: dict[str, Any] | str,
        attempt: int,
    ) -> dict[str, Any] | str:
        del attempt
        return request

    preserve_requested_production_width._mmm_ai_chosen_page_width = True  # type: ignore[attr-defined]
    runtime_module._narrow_production_repair_request = preserve_requested_production_width

    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_persistent_field_patch", False):
        return

    @wraps(current)
    def generate_with_persistent_field_patch(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        if not _outline_allowed(expected_contracts):
            return current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )

        checkpoint_path = _checkpoint_path(stage, request)
        checkpoint_state = _load_checkpoint(checkpoint_path)
        saved_batches = list(checkpoint_state.get("saved_batches", []))

        # Resume exactly where a previous run stopped before asking for any new plan.
        if checkpoint_state.get("pending_batches"):
            _process_pending_batches(
                runtime_module,
                complete_planner_module,
                router,
                saved_batches=saved_batches,
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )
            checkpoint_state = _load_checkpoint(checkpoint_path)
            if checkpoint_state.get("status") == "page_complete":
                return _resume_result(checkpoint_state, saved_batches)

        if checkpoint_state.get("status") in {"page_complete", "complete"}:
            return _resume_result(checkpoint_state, saved_batches)

        generation_round = int(checkpoint_state.get("generation_round", 0))
        first_generation = generation_round == 0

        while True:
            generation_round += 1
            accepted_ids = _accepted_batch_ids(saved_batches)
            attempt_request: dict[str, Any] | str = request
            if isinstance(request, dict):
                attempt_request = dict(request)
                resume_cursor = str(checkpoint_state.get("resume_cursor", ""))
                if resume_cursor:
                    attempt_request["cursor"] = resume_cursor
                if accepted_ids:
                    attempt_request["accepted_outline_prefix"] = {
                        "saved_batch_ids": accepted_ids,
                        "saved_batch_count": len(accepted_ids),
                        "saved_batches_sha256": _fingerprint(saved_batches),
                        "instruction": (
                            "These batches are already valid and persisted. Continue only "
                            "missing work; never regenerate or modify them."
                        ),
                    }
                last_error = checkpoint_state.get("last_envelope_error")
                if last_error:
                    attempt_request["missing_fragment_reason"] = last_error

            request_text = (
                attempt_request
                if isinstance(attempt_request, str)
                else json.dumps(attempt_request, ensure_ascii=False)
            )
            schema_view = (
                attempt_request.get("contract")
                if isinstance(attempt_request, dict)
                and isinstance(attempt_request.get("contract"), dict)
                else None
            )
            schema = (
                runtime_module._schema_for_contract(schema_view)
                if isinstance(schema_view, dict)
                and frozenset(schema_view) == _OUTLINE_FIELDS
                else None
            )

            checkpoint_state.update(
                {
                    "stage": stage,
                    "request_sha256": _fingerprint(request),
                    "saved_batches": saved_batches,
                    "generation_round": generation_round,
                    "status": "generating_missing_work",
                }
            )
            _save_checkpoint(checkpoint_path, checkpoint_state)

            token = runtime_module._JSON_SCHEMA.set(schema)
            try:
                text = router.generate_text(
                    "planner",
                    [
                        {"role": "system", "content": _SCALABLE_OUTLINE_PROMPT},
                        {"role": "user", "content": request_text},
                    ],
                    media_paths=media_paths if first_generation else (),
                    response_format="json",
                )
            finally:
                runtime_module._JSON_SCHEMA.reset(token)
            first_generation = False

            # Preserve compatibility with the legacy one-shot initial contract.
            if not saved_batches:
                containers = _outermost_complete_json_containers(text)
                if len(containers) == 1 and isinstance(containers[0].value, dict):
                    fields = frozenset(str(key) for key in containers[0].value)
                    if fields in tuple(expected_contracts) and fields != _OUTLINE_FIELDS:
                        return runtime_module._extract_with_safe_empty_defaults(
                            complete_planner_module,
                            text,
                            expected_contracts=expected_contracts,
                        )

            try:
                page = _extract_outline_sequence(text)
                raw_batches = list(page["production_batches"])
                page_complete = bool(page["complete"])
                page_next_cursor = str(page["next_cursor"])
                envelope_error = ""
            except Exception as exc:
                # A broken envelope does not erase complete batch objects already emitted.
                raw_batches = _batch_candidates_from_text(text)
                page_complete = False
                page_next_cursor = ""
                envelope_error = f"{type(exc).__name__}: {exc}"

            if not raw_batches:
                # A valid terminal outline page may legitimately contain zero batches.
                # Treat it as successful no-op work instead of regenerating forever.
                if page_complete and not page_next_cursor and not envelope_error:
                    checkpoint_state.update(
                        {
                            "saved_batches": saved_batches,
                            "pending_batches": [],
                            "pending_patch": None,
                            "page_complete": True,
                            "page_next_cursor": "",
                            "resume_cursor": "",
                            "last_envelope_error": "",
                            "status": "complete",
                        }
                    )
                    _save_checkpoint(checkpoint_path, checkpoint_state)
                    return _resume_result(checkpoint_state, saved_batches)

                checkpoint_state.update(
                    {
                        "saved_batches": saved_batches,
                        "last_envelope_error": envelope_error or "no complete batch object",
                        "status": "awaiting_missing_fragment",
                    }
                )
                _save_checkpoint(checkpoint_path, checkpoint_state)
                continue

            if not page_complete and not page_next_cursor:
                page_next_cursor = "host_resume_" + _fingerprint(
                    {"saved": saved_batches, "new": raw_batches}
                )[:16]

            # Persist the whole queue BEFORE validating the first item. From here on a
            # crash/restart resumes the exact queue rather than regenerating the page.
            checkpoint_state.update(
                {
                    "saved_batches": saved_batches,
                    "pending_batches": raw_batches,
                    "pending_patch": None,
                    "page_complete": page_complete,
                    "page_next_cursor": page_next_cursor,
                    "resume_cursor": page_next_cursor,
                    "last_envelope_error": envelope_error,
                    "status": "collecting",
                }
            )
            _save_checkpoint(checkpoint_path, checkpoint_state)

            _process_pending_batches(
                runtime_module,
                complete_planner_module,
                router,
                saved_batches=saved_batches,
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )

            checkpoint_state.update(
                {
                    "saved_batches": saved_batches,
                    "pending_batches": [],
                    "pending_patch": None,
                    "page_complete": page_complete,
                    "page_next_cursor": page_next_cursor,
                    "status": "complete" if page_complete else "page_complete",
                }
            )
            _save_checkpoint(checkpoint_path, checkpoint_state)
            return _resume_result(checkpoint_state, saved_batches)

    generate_with_persistent_field_patch._mmm_persistent_field_patch = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_persistent_field_patch


__all__ = ["install"]
