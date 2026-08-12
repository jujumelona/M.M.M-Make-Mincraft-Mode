from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from typing import Any, Sequence

from .planner_outline_prompt_contract import _SCALABLE_OUTLINE_PROMPT
from .planner_strict_json_contract import (
    _extract_outline_prefix,
    _extract_outline_sequence,
    _outermost_complete_json_containers,
)


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})


def _outline_allowed(expected_contracts: Sequence[frozenset[str]]) -> bool:
    return _OUTLINE_FIELDS in tuple(expected_contracts)


def _repair_attempts() -> int:
    raw = os.environ.get("MMM_OUTLINE_REPAIR_ATTEMPTS", "6").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 6
    return max(2, min(value, 12))


def _batch_identity(value: Any) -> str:
    if isinstance(value, dict):
        batch_id = str(value.get("batch_id", "")).strip()
        if batch_id:
            return "id:" + batch_id
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _merge_saved_batches(saved: list[Any], incoming: Sequence[Any]) -> None:
    known = {_batch_identity(value) for value in saved}
    for value in incoming:
        identity = _batch_identity(value)
        if identity in known:
            continue
        saved.append(value)
        known.add(identity)


def _partition_batch_descriptors(
    module: Any,
    raw_batches: Sequence[Any],
    *,
    page_label: str,
) -> tuple[list[Any], list[str]]:
    valid: list[Any] = []
    errors: list[str] = []
    for batch_index, raw in enumerate(raw_batches, start=1):
        try:
            module._production_batch(raw)
        except Exception as exc:
            errors.append(
                f"{page_label} batch {batch_index}: {type(exc).__name__}: {exc}"
            )
            continue
        valid.append(raw)
    return valid, errors


def _valid_batch_descriptors(module: Any, text: str) -> tuple[list[Any], list[str]]:
    """Salvage individually valid batch descriptors from an otherwise bad response."""

    valid: list[Any] = []
    errors: list[str] = []
    for container_index, container in enumerate(
        _outermost_complete_json_containers(text),
        start=1,
    ):
        value = container.value
        if not isinstance(value, dict):
            continue
        raw_batches = value.get("production_batches")
        if not isinstance(raw_batches, list):
            continue
        accepted, rejected = _partition_batch_descriptors(
            module,
            raw_batches,
            page_label=f"page {container_index}",
        )
        valid.extend(accepted)
        errors.extend(rejected)
    return valid, errors


def _saved_receipt(saved: Sequence[Any], cursor: str, repair_error: str) -> dict[str, Any]:
    batch_ids = [
        str(value.get("batch_id", "")).strip()
        for value in saved
        if isinstance(value, dict) and str(value.get("batch_id", "")).strip()
    ]
    encoded = json.dumps(
        list(saved),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "saved_batch_count": len(saved),
        "saved_batch_ids": batch_ids,
        "saved_batches_sha256": hashlib.sha256(encoded).hexdigest(),
        "resume_cursor": cursor,
        "repair_error": repair_error,
        "instruction": (
            "These batches are already host-validated and saved. Do not regenerate "
            "or modify them; emit only the rejected/missing continuation."
        ),
    }


def _repair_request(
    original: dict[str, Any] | str,
    *,
    saved: Sequence[Any],
    cursor: str,
    repair_error: str,
) -> dict[str, Any] | str:
    if not isinstance(original, dict):
        return original
    repaired = dict(original)
    if cursor:
        repaired["cursor"] = cursor
    repaired["accepted_outline_prefix"] = _saved_receipt(
        saved,
        cursor,
        repair_error,
    )
    repaired["repair_error"] = repair_error
    return repaired


def _terminal_page(saved: Sequence[Any], page: dict[str, Any]) -> dict[str, Any]:
    combined = list(saved)
    _merge_saved_batches(combined, page["production_batches"])
    return {
        "production_batches": combined,
        "complete": page["complete"],
        "next_cursor": page["next_cursor"],
    }


def install(runtime_module: Any) -> None:
    """Checkpoint valid planner work and repair only rejected outline fragments."""

    from . import complete_planner as complete_planner_module

    # Remove the earlier arbitrary 2-target/1-target production width override. Repair
    # may narrow to actually missing work, but normal generation size belongs to AI.
    def preserve_requested_production_width(
        request: dict[str, Any] | str,
        attempt: int,
    ) -> dict[str, Any] | str:
        del attempt
        return request

    preserve_requested_production_width._mmm_ai_chosen_page_width = True  # type: ignore[attr-defined]
    runtime_module._narrow_production_repair_request = preserve_requested_production_width

    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_incremental_outline_repair", False):
        return

    @wraps(current)
    def generate_with_incremental_outline_repair(
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

        saved_batches: list[Any] = []
        resume_cursor = ""
        previous_error = ""
        last_error: BaseException | None = None

        for attempt in range(_repair_attempts()):
            attempt_request = _repair_request(
                request,
                saved=saved_batches,
                cursor=resume_cursor,
                repair_error=previous_error,
            )
            request_text = (
                attempt_request
                if isinstance(attempt_request, str)
                else json.dumps(attempt_request, ensure_ascii=False)
            )
            prompt = _SCALABLE_OUTLINE_PROMPT
            if attempt:
                prompt += (
                    "\nINCREMENTAL REPAIR: Correct only the rejected or missing portion. "
                    "The host has already saved every valid batch listed in "
                    "accepted_outline_prefix. Do not reproduce them. If repair_error "
                    "names an invalid batch descriptor, emit a corrected replacement "
                    "for that descriptor and then continue the outline if needed. "
                    "Host diagnostic: "
                    + previous_error
                )

            view = (
                attempt_request.get("contract")
                if isinstance(attempt_request, dict)
                and isinstance(attempt_request.get("contract"), dict)
                else None
            )
            schema = (
                runtime_module._schema_for_contract(view)
                if isinstance(view, dict)
                and frozenset(view) == _OUTLINE_FIELDS
                else None
            )
            token = runtime_module._JSON_SCHEMA.set(schema)
            try:
                text = router.generate_text(
                    "planner",
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": request_text},
                    ],
                    media_paths=media_paths if attempt == 0 else (),
                    response_format="json",
                )
            finally:
                runtime_module._JSON_SCHEMA.reset(token)

            # Compatibility: a one-shot legacy contract is still accepted on the
            # initial call when no outline work has yet been checkpointed.
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
                valid_batches, batch_errors = _partition_batch_descriptors(
                    complete_planner_module,
                    page["production_batches"],
                    page_label="outline",
                )
                if batch_errors:
                    _merge_saved_batches(saved_batches, valid_batches)
                    cursor_value = page.get("next_cursor")
                    if isinstance(cursor_value, str) and cursor_value:
                        resume_cursor = cursor_value
                    previous_error = (
                        "invalid batch descriptors: " + " | ".join(batch_errors[:4])
                    )
                    last_error = complete_planner_module.SpecValidationError(previous_error)
                    continue
                return _terminal_page(saved_batches, page)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                prefix, prefix_error = _extract_outline_prefix(text)
                if prefix is not None:
                    prefix_valid, prefix_errors = _partition_batch_descriptors(
                        complete_planner_module,
                        prefix.get("production_batches", []),
                        page_label="accepted prefix",
                    )
                    _merge_saved_batches(saved_batches, prefix_valid)
                    cursor_value = prefix.get("next_cursor")
                    if isinstance(cursor_value, str) and cursor_value:
                        resume_cursor = cursor_value
                else:
                    prefix_errors = []

                # Even if the page envelope/bookkeeping is bad, preserve every batch
                # descriptor that already passes the real host batch parser.
                valid_batches, batch_errors = _valid_batch_descriptors(
                    complete_planner_module,
                    text,
                )
                _merge_saved_batches(saved_batches, valid_batches)
                detail = prefix_error or str(exc)
                all_batch_errors = [*prefix_errors, *batch_errors]
                if all_batch_errors:
                    detail += "; invalid batch descriptors: " + " | ".join(
                        all_batch_errors[:4]
                    )
                previous_error = detail

        assert last_error is not None
        saved_ids = [
            str(value.get("batch_id", "")).strip()
            for value in saved_batches
            if isinstance(value, dict) and str(value.get("batch_id", "")).strip()
        ]
        raise complete_planner_module.SpecValidationError(
            f"{stage} incremental repair exhausted after preserving "
            f"{len(saved_batches)} valid batches; saved_batch_ids={saved_ids[:12]}; "
            f"last_error={previous_error}"
        ) from last_error

    generate_with_incremental_outline_repair._mmm_incremental_outline_repair = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_incremental_outline_repair


__all__ = ["install"]
