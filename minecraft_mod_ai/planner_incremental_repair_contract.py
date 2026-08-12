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
_PATCH_FIELDS = frozenset({"target_fingerprint", "replacement_batch"})
_CHECKPOINT_VERSION = 1


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
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in _all_complete_dicts(text):
        fields = frozenset(str(key) for key in value)
        if fields != _BATCH_FIELDS:
            continue
        identity = _fingerprint(value)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(value)
    return result


def _patch_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    batch = {
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
    }
    return {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "replacement_batch": batch,
        },
        "required": sorted(_PATCH_FIELDS),
        "additionalProperties": False,
    }


def _batch_validation_error(module: Any, raw: Any) -> str:
    try:
        module._production_batch(raw)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def _repair_one_batch(
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
    """Patch one invalid batch forever-until-valid; never regenerate its page."""

    target_fingerprint = _fingerprint(raw_batch)
    current_value = raw_batch
    current_error = validation_error
    patch_round = 0

    while True:
        patch_round += 1
        request = {
            "batch_patch_request": {
                "target_fingerprint": target_fingerprint,
                "invalid_batch": current_value,
                "validation_error": current_error,
                "accepted_batch_ids": list(accepted_batch_ids),
                "required_batch_fields": sorted(_BATCH_FIELDS),
                "rules": {
                    "batch_id": "non-empty descriptive snake_case string, unique",
                    "scope": "non-empty string",
                    "depends_on_batches": (
                        "unique non-empty strings; no self dependency; only accepted/known ids"
                    ),
                    "deliverables": "non-empty array of unique non-empty strings",
                    "exports": "array of unique non-empty snake_case strings",
                },
                "output_contract": {
                    "target_fingerprint": target_fingerprint,
                    "replacement_batch": {
                        "batch_id": "string",
                        "scope": "string",
                        "depends_on_batches": ["string"],
                        "deliverables": ["string"],
                        "exports": ["string"],
                    },
                },
            }
        }
        prompt = (
            "You are a deterministic JSON patcher. Repair exactly ONE invalid "
            "production batch. Do not generate a page, plan, explanation, Markdown, "
            "or any other batch. Preserve every semantically correct value from "
            "invalid_batch and change only what validation_error requires. Return "
            "exactly one JSON object with keys target_fingerprint and replacement_batch. "
            "target_fingerprint must exactly echo the supplied fingerprint."
        )

        token = runtime_module._JSON_SCHEMA.set(_patch_schema())
        try:
            text = router.generate_text(
                "planner",
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                media_paths=(),
                response_format="json",
            )
        finally:
            runtime_module._JSON_SCHEMA.reset(token)

        try:
            patch = _extract_one_complete_object(text)
            if frozenset(str(key) for key in patch) != _PATCH_FIELDS:
                raise ValueError(
                    f"patch fields must be {sorted(_PATCH_FIELDS)}"
                )
            if patch.get("target_fingerprint") != target_fingerprint:
                raise ValueError("patch target_fingerprint does not match requested batch")
            replacement = patch.get("replacement_batch")
            if not isinstance(replacement, dict):
                raise ValueError("replacement_batch must be an object")
            if frozenset(str(key) for key in replacement) != _BATCH_FIELDS:
                raise ValueError(
                    f"replacement_batch fields must be {sorted(_BATCH_FIELDS)}"
                )
            error = _batch_validation_error(module, replacement)
            if error:
                raise ValueError(error)
            return dict(replacement)
        except Exception as exc:
            # This is a patch-of-the-patch, not regeneration of accepted plan work.
            current_value = (
                patch.get("replacement_batch")
                if "patch" in locals() and isinstance(patch, dict)
                else current_value
            )
            current_error = f"{type(exc).__name__}: {exc}"
            checkpoint_state["status"] = "patching"
            checkpoint_state["pending_patch"] = {
                "target_fingerprint": target_fingerprint,
                "round": patch_round,
                "current_value": current_value,
                "validation_error": current_error,
            }
            _save_checkpoint(checkpoint_path, checkpoint_state)


def _validated_batches_with_patches(
    runtime_module: Any,
    module: Any,
    router: Any,
    *,
    raw_batches: Sequence[Any],
    saved_batches: list[Any],
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    accepted_ids = [
        str(value.get("batch_id", "")).strip()
        for value in saved_batches
        if isinstance(value, dict) and str(value.get("batch_id", "")).strip()
    ]

    for raw in raw_batches:
        error = _batch_validation_error(module, raw)
        if not error and isinstance(raw, dict):
            resolved_batch = dict(raw)
        else:
            resolved_batch = _repair_one_batch(
                runtime_module,
                module,
                router,
                raw_batch=raw,
                validation_error=error or "batch must be a JSON object",
                accepted_batch_ids=accepted_ids,
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )
        resolved.append(resolved_batch)
        batch_id = str(resolved_batch.get("batch_id", "")).strip()
        if batch_id and batch_id not in accepted_ids:
            accepted_ids.append(batch_id)
        _merge_saved_batches(saved_batches, [resolved_batch])
        checkpoint_state["saved_batches"] = saved_batches
        checkpoint_state["pending_patch"] = None
        checkpoint_state["status"] = "collecting"
        _save_checkpoint(checkpoint_path, checkpoint_state)
    return resolved


def install(runtime_module: Any) -> None:
    """Persist valid planner work and patch only invalid batch objects in place."""

    from . import complete_planner as complete_planner_module

    # Host no longer invents a 2-target/1-target page width. AI owns normal page size.
    def preserve_requested_production_width(
        request: dict[str, Any] | str,
        attempt: int,
    ) -> dict[str, Any] | str:
        del attempt
        return request

    preserve_requested_production_width._mmm_ai_chosen_page_width = True  # type: ignore[attr-defined]
    runtime_module._narrow_production_repair_request = preserve_requested_production_width

    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_persistent_batch_patch", False):
        return

    @wraps(current)
    def generate_with_persistent_batch_patch(
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
        resume_cursor = str(checkpoint_state.get("resume_cursor", ""))
        if checkpoint_state.get("status") == "complete":
            return {
                "production_batches": saved_batches,
                "complete": bool(checkpoint_state.get("complete", True)),
                "next_cursor": str(checkpoint_state.get("next_cursor", "")),
            }

        attempt_request = request
        if isinstance(request, dict) and resume_cursor:
            attempt_request = {**request, "cursor": resume_cursor}
        request_text = (
            attempt_request
            if isinstance(attempt_request, str)
            else json.dumps(attempt_request, ensure_ascii=False)
        )

        # One normal generation for new work. Semantic defects inside that output are
        # patched object-by-object below; this call is never repeated merely because a
        # batch descriptor is invalid.
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
        token = runtime_module._JSON_SCHEMA.set(schema)
        try:
            text = router.generate_text(
                "planner",
                [
                    {"role": "system", "content": _SCALABLE_OUTLINE_PROMPT},
                    {"role": "user", "content": request_text},
                ],
                media_paths=media_paths,
                response_format="json",
            )
        finally:
            runtime_module._JSON_SCHEMA.reset(token)

        # Legacy single-object contracts remain compatible on the first call.
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

        # Prefer complete outline pages. Their batch members are independently patched
        # in place; a single bad member never invalidates its siblings.
        try:
            page = _extract_outline_sequence(text)
            raw_batches = page["production_batches"]
            complete = bool(page["complete"])
            next_cursor = str(page["next_cursor"])
        except Exception as envelope_error:
            # If the outer page was truncated, preserve every complete batch object that
            # survived in the stream. Then continue from that checkpoint; do not ask the
            # model to recreate those batches.
            raw_batches = _batch_candidates_from_text(text)
            complete = False
            next_cursor = ""
            checkpoint_state["last_envelope_error"] = (
                f"{type(envelope_error).__name__}: {envelope_error}"
            )
            if not raw_batches:
                # There is no accepted semantic object to patch. Preserve state and ask
                # only for the missing continuation fragment on the next loop iteration.
                checkpoint_state.update(
                    {
                        "saved_batches": saved_batches,
                        "resume_cursor": resume_cursor,
                        "status": "awaiting_continuation",
                    }
                )
                _save_checkpoint(checkpoint_path, checkpoint_state)
                continuation_request = (
                    {**request, "accepted_outline_prefix": {
                        "saved_batch_ids": [
                            value.get("batch_id") for value in saved_batches
                            if isinstance(value, dict)
                        ],
                        "instruction": "Continue only missing outline work; do not regenerate saved batches.",
                    }}
                    if isinstance(request, dict)
                    else request
                )
                return generate_with_persistent_batch_patch(
                    router,
                    system_prompt=system_prompt,
                    request=continuation_request,
                    media_paths=(),
                    expected_contracts=expected_contracts,
                    stage=stage,
                )

        checkpoint_state.update(
            {
                "stage": stage,
                "request_sha256": _fingerprint(request),
                "saved_batches": saved_batches,
                "resume_cursor": next_cursor or resume_cursor,
                "status": "collecting",
            }
        )
        _save_checkpoint(checkpoint_path, checkpoint_state)

        _validated_batches_with_patches(
            runtime_module,
            complete_planner_module,
            router,
            raw_batches=raw_batches,
            saved_batches=saved_batches,
            checkpoint_path=checkpoint_path,
            checkpoint_state=checkpoint_state,
        )

        checkpoint_state["saved_batches"] = saved_batches
        checkpoint_state["resume_cursor"] = next_cursor
        checkpoint_state["complete"] = complete
        checkpoint_state["next_cursor"] = next_cursor
        checkpoint_state["status"] = "complete" if complete else "page_complete"
        _save_checkpoint(checkpoint_path, checkpoint_state)

        return {
            "production_batches": saved_batches,
            "complete": complete,
            "next_cursor": next_cursor,
        }

    generate_with_persistent_batch_patch._mmm_persistent_batch_patch = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_persistent_batch_patch


__all__ = ["install"]
