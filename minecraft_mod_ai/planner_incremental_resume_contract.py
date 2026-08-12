from __future__ import annotations

import json
import os
from functools import wraps
from typing import Any, Sequence


_FIELD_PATCH_FIELDS = frozenset({"target_fingerprint", "set_fields", "delete_fields"})
_REPLACE_PATCH_FIELDS = frozenset({"target_fingerprint", "replacement_batch"})
_DEFAULT_OUTLINE_GENERATION_ATTEMPTS = 2
_DEFAULT_BATCH_REPAIR_ATTEMPTS = 2


def _bounded_env(name: str, default: int, *, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(1, min(value, maximum))


def _field_patch_schema(batch_fields: Sequence[str]) -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    properties: dict[str, Any] = {
        "batch_id": {"type": "string"},
        "scope": {"type": "string"},
        "depends_on_batches": string_array,
        "deliverables": string_array,
        "exports": string_array,
    }
    return {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "set_fields": {
                "type": "object",
                "properties": {
                    field: properties.get(field, {}) for field in batch_fields
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


def _replacement_patch_schema(batch_fields: Sequence[str]) -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    properties: dict[str, Any] = {
        "batch_id": {"type": "string"},
        "scope": {"type": "string"},
        "depends_on_batches": string_array,
        "deliverables": string_array,
        "exports": string_array,
    }
    return {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "replacement_batch": {
                "type": "object",
                "properties": {
                    field: properties.get(field, {}) for field in batch_fields
                },
                "required": sorted(batch_fields),
                "additionalProperties": False,
            },
        },
        "required": sorted(_REPLACE_PATCH_FIELDS),
        "additionalProperties": False,
    }


def _contextual_batch_error(
    incremental_module: Any,
    module: Any,
    candidate: Any,
    accepted_batch_ids: Sequence[str],
) -> str:
    error = incremental_module._batch_validation_error(module, candidate)
    if error:
        return error
    if not isinstance(candidate, dict):
        return "batch must be a JSON object"
    candidate_id = str(candidate.get("batch_id", "")).strip()
    if candidate_id and candidate_id in set(accepted_batch_ids):
        return (
            f"duplicate batch_id {candidate_id!r}; this id is already saved. "
            "Choose a new descriptive snake_case batch_id while preserving the batch purpose."
        )
    return ""


def _install_bounded_batch_repair(incremental_module: Any) -> None:
    current = incremental_module._patch_one_invalid_batch
    if getattr(current, "_mmm_bounded_semantic_batch_repair", False):
        return

    def patch_one_invalid_batch(
        runtime_module: Any,
        module: Any,
        router: Any,
        *,
        raw_batch: Any,
        validation_error: str,
        accepted_batch_ids: Sequence[str],
        checkpoint_path: Any,
        checkpoint_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Repair one batch with one field patch then one full-item regeneration.

        A retry is useful only when the action changes. Repeating the same deterministic
        patch request cannot add information, so semantic escalation is explicit and
        bounded. Backend exceptions still propagate with the saved pending state so a
        later process can resume without regenerating accepted outline work.
        """

        batch_fields = frozenset(incremental_module._BATCH_FIELDS)
        original_fingerprint = incremental_module._fingerprint(raw_batch)
        pending = checkpoint_state.get("pending_patch")
        if (
            isinstance(pending, dict)
            and pending.get("target_fingerprint") == original_fingerprint
        ):
            current_value = pending.get("current_value", raw_batch)
            current_error = str(pending.get("validation_error", validation_error))
        else:
            current_value = raw_batch
            current_error = validation_error

        max_attempts = _bounded_env(
            "MMM_PLANNER_BATCH_REPAIR_ATTEMPTS",
            _DEFAULT_BATCH_REPAIR_ATTEMPTS,
            maximum=2,
        )
        seen_states: set[str] = set()
        seen_outputs: set[str] = set()
        last_output_sha256 = ""

        for attempt in range(1, max_attempts + 1):
            replacement_mode = not isinstance(current_value, dict) or attempt > 1
            repair_mode = "replacement" if replacement_mode else "field_patch"
            state_sha256 = incremental_module._fingerprint(
                {
                    "mode": repair_mode,
                    "current": current_value,
                    "validation_error": current_error,
                    "accepted_batch_ids": list(accepted_batch_ids),
                }
            )
            if state_sha256 in seen_states:
                module.SpecValidationError(
                    "Production batch repair repeated an identical semantic state."
                )
            seen_states.add(state_sha256)

            checkpoint_state["status"] = "patching"
            checkpoint_state["pending_patch"] = {
                "target_fingerprint": original_fingerprint,
                "round": attempt,
                "repair_mode": repair_mode,
                "current_value": current_value,
                "validation_error": current_error,
                "state_sha256": state_sha256,
            }
            incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)

            if not replacement_mode:
                prompt = (
                    "You are a deterministic field-level JSON patcher. Fix exactly ONE "
                    "invalid production batch. Return only target_fingerprint, set_fields, "
                    "and delete_fields. DO NOT rewrite the whole batch. Change only fields "
                    "required by validation_error; preserve every other field exactly."
                )
                output_contract: dict[str, Any] = {
                    "target_fingerprint": original_fingerprint,
                    "set_fields": {"only_invalid_or_missing_fields": "corrected value"},
                    "delete_fields": ["only_invalid_extra_field_names"],
                }
                schema = _field_patch_schema(sorted(batch_fields))
            else:
                prompt = (
                    "You regenerate exactly ONE invalid production batch. Return only "
                    "target_fingerprint and replacement_batch. Preserve the batch purpose, "
                    "dependencies, deliverables and exports whenever they are valid, but emit "
                    "one complete batch object that fixes validation_error. Do not emit a page, "
                    "sibling batch, explanation, or Markdown."
                )
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
                schema = _replacement_patch_schema(sorted(batch_fields))

            patch_request = {
                "target_fingerprint": original_fingerprint,
                "repair_mode": repair_mode,
                "current_batch": current_value,
                "validation_error": current_error,
                "accepted_batch_ids": list(accepted_batch_ids),
                "required_fields": sorted(batch_fields),
                "rules": {
                    "batch_id": "unique non-empty descriptive snake_case string",
                    "scope": "non-empty string",
                    "depends_on_batches": "unique non-empty strings; no self dependency",
                    "deliverables": "NON-EMPTY unique strings",
                    "exports": "unique non-empty snake_case strings",
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

            last_output_sha256 = incremental_module._fingerprint(text)
            if last_output_sha256 in seen_outputs:
                checkpoint_state.update(
                    {
                        "status": "failed",
                        "pending_patch": {
                            **checkpoint_state["pending_patch"],
                            "reason": "repeated_model_output",
                            "last_output_sha256": last_output_sha256,
                        },
                    }
                )
                incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)
                raise module.SpecValidationError(
                    "Production batch repair repeated identical model output."
                )
            seen_outputs.add(last_output_sha256)

            patch: dict[str, Any] | None = None
            candidate: Any = current_value
            try:
                patch = incremental_module._extract_one_complete_object(text)
                if patch.get("target_fingerprint") != original_fingerprint:
                    raise ValueError("target_fingerprint does not match saved invalid batch")

                if not replacement_mode:
                    if frozenset(str(key) for key in patch) != _FIELD_PATCH_FIELDS:
                        raise ValueError("field patch has invalid top-level keys")
                    set_fields = patch.get("set_fields")
                    delete_fields = patch.get("delete_fields")
                    if not isinstance(set_fields, dict):
                        raise ValueError("set_fields must be an object")
                    if not isinstance(delete_fields, list) or any(
                        not isinstance(value, str) for value in delete_fields
                    ):
                        raise ValueError("delete_fields must be an array of field names")
                    if any(str(key) not in batch_fields for key in set_fields):
                        raise ValueError("set_fields contains a non-batch field")
                    if any(field in batch_fields for field in delete_fields):
                        raise ValueError("required batch fields may not be deleted")
                    if not set_fields and not delete_fields:
                        raise ValueError("field patch must change at least one field")
                    candidate = dict(current_value)
                    for field in delete_fields:
                        candidate.pop(field, None)
                    candidate.update(set_fields)
                else:
                    if frozenset(str(key) for key in patch) != _REPLACE_PATCH_FIELDS:
                        raise ValueError("replacement patch has invalid top-level keys")
                    replacement = patch.get("replacement_batch")
                    if not isinstance(replacement, dict):
                        raise ValueError("replacement_batch must be an object")
                    candidate = dict(replacement)

                next_error = _contextual_batch_error(
                    incremental_module,
                    module,
                    candidate,
                    accepted_batch_ids,
                )
                if next_error:
                    current_value = candidate
                    current_error = next_error
                    continue

                checkpoint_state["pending_patch"] = None
                checkpoint_state["status"] = "collecting"
                incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)
                return dict(candidate)
            except Exception as exc:
                if isinstance(candidate, dict):
                    current_value = candidate
                current_error = f"{type(exc).__name__}: {exc}"

        checkpoint_state.update(
            {
                "status": "failed",
                "pending_patch": {
                    "target_fingerprint": original_fingerprint,
                    "round": max_attempts,
                    "current_value": current_value,
                    "validation_error": current_error,
                    "reason": "repair_budget_exhausted",
                    "last_output_sha256": last_output_sha256,
                },
            }
        )
        incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)
        raise module.SpecValidationError(
            "Production batch could not satisfy the host contract after field repair and "
            f"single-item regeneration: {current_error}"
        )

    patch_one_invalid_batch._mmm_bounded_semantic_batch_repair = True  # type: ignore[attr-defined]
    patch_one_invalid_batch.__wrapped__ = current  # type: ignore[attr-defined]
    incremental_module._patch_one_invalid_batch = patch_one_invalid_batch


def _install_contextual_pending_queue(incremental_module: Any) -> None:
    current = incremental_module._process_pending_batches
    if getattr(current, "_mmm_contextual_pending_queue", False):
        return

    @wraps(current)
    def process_with_context(*args: Any, **kwargs: Any) -> None:
        runtime_module = args[0] if len(args) > 0 else kwargs.get("runtime_module")
        module = args[1] if len(args) > 1 else kwargs.get("module")
        router = args[2] if len(args) > 2 else kwargs.get("router")
        saved_batches = kwargs.get("saved_batches")
        checkpoint_path = kwargs.get("checkpoint_path")
        checkpoint_state = kwargs.get("checkpoint_state")

        if (
            runtime_module is None
            or module is None
            or router is None
            or not isinstance(saved_batches, list)
            or checkpoint_path is None
            or not isinstance(checkpoint_state, dict)
        ):
            current(*args, **kwargs)
            return

        pending = list(checkpoint_state.get("pending_batches", []))
        while pending:
            candidate = pending[0]
            accepted_ids = incremental_module._accepted_batch_ids(saved_batches)
            error = _contextual_batch_error(
                incremental_module,
                module,
                candidate,
                accepted_ids,
            )
            if error:
                resolved = incremental_module._patch_one_invalid_batch(
                    runtime_module,
                    module,
                    router,
                    raw_batch=candidate,
                    validation_error=error,
                    accepted_batch_ids=accepted_ids,
                    checkpoint_path=checkpoint_path,
                    checkpoint_state=checkpoint_state,
                )
            elif isinstance(candidate, dict):
                resolved = dict(candidate)
            else:
                raise module.SpecValidationError("Production batch must be a JSON object.")

            # The bounded repair function guarantees parser validity and contextual
            # uniqueness before returning; never loop a repaired candidate again.
            post_error = _contextual_batch_error(
                incremental_module,
                module,
                resolved,
                accepted_ids,
            )
            if post_error:
                raise module.SpecValidationError(
                    f"Production batch repair returned invalid state: {post_error}"
                )

            incremental_module._merge_saved_batches(saved_batches, [resolved])
            pending.pop(0)
            checkpoint_state.update(
                {
                    "saved_batches": saved_batches,
                    "pending_batches": pending,
                    "pending_patch": None,
                    "status": "collecting",
                }
            )
            incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)

        checkpoint_state["saved_batches"] = saved_batches
        checkpoint_state["pending_patch"] = None
        checkpoint_state["status"] = (
            "complete" if checkpoint_state.get("page_complete") else "page_complete"
        )
        incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)

    process_with_context._mmm_contextual_pending_queue = True  # type: ignore[attr-defined]
    incremental_module._process_pending_batches = process_with_context


def _install_outline_cycle_guard(incremental_module: Any) -> None:
    from . import complete_planner

    current = complete_planner._generate_json_page_with_repair
    if getattr(current, "_mmm_outline_cycle_guard", False):
        return

    class _GuardedRouter:
        def __init__(self, router: Any, error_type: type[Exception]) -> None:
            self._router = router
            self._error_type = error_type
            self._seen_requests: set[str] = set()
            self._outline_calls = 0
            self._outline_limit = _bounded_env(
                "MMM_PLANNER_OUTLINE_GENERATION_ATTEMPTS",
                _DEFAULT_OUTLINE_GENERATION_ATTEMPTS,
                maximum=4,
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._router, name)

        def generate_text(
            self,
            role: str,
            messages: Any,
            *,
            media_paths=(),
            response_format="text",
        ) -> str:
            system_content = ""
            user_content = ""
            if isinstance(messages, (list, tuple)) and messages:
                first = messages[0]
                last = messages[-1]
                if isinstance(first, dict):
                    system_content = str(first.get("content", ""))
                if isinstance(last, dict):
                    user_content = str(last.get("content", ""))

            is_batch_repair = (
                "field-level JSON patcher" in system_content
                or "regenerate exactly ONE invalid production batch" in system_content
            )
            if not is_batch_repair:
                self._outline_calls += 1
                if self._outline_calls > self._outline_limit:
                    raise self._error_type(
                        "Production outline generation made no valid progress after the "
                        "diagnostic regeneration path."
                    )

            request_sha256 = incremental_module._fingerprint(
                {
                    "role": role,
                    "system": system_content,
                    "user": user_content,
                    "response_format": response_format,
                    "media_paths": [str(path) for path in media_paths],
                }
            )
            if request_sha256 in self._seen_requests:
                raise self._error_type(
                    "Planner repair cycle detected before repeating an identical model request."
                )
            self._seen_requests.add(request_sha256)

            return self._router.generate_text(
                role,
                messages,
                media_paths=media_paths,
                response_format=response_format,
            )

    @wraps(current)
    def generate_cycle_safe(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        if not incremental_module._outline_allowed(expected_contracts):
            return current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )
        guarded = _GuardedRouter(router, complete_planner.SpecValidationError)
        return current(
            guarded,
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )

    generate_cycle_safe._mmm_outline_cycle_guard = True  # type: ignore[attr-defined]
    complete_planner._generate_json_page_with_repair = generate_cycle_safe


def install(incremental_module: Any) -> None:
    """Install crash-safe resume plus terminating planner repair semantics.

    The older repair implementation remains only as an implementation detail below this
    final runtime layer. All active outline batch repair is bounded and cycle-aware: a
    bad object gets a field patch, then one complete-item regeneration, and an outline
    envelope gets one diagnostic regeneration. Identical model requests are never sent
    twice in one planner call.
    """

    from . import planner_json_runtime_contract as planner_runtime_module

    def no_production_width_narrowing(request: Any, attempt: int) -> Any:
        del attempt
        return request

    no_production_width_narrowing._mmm_no_fixed_production_width = True  # type: ignore[attr-defined]
    planner_runtime_module._narrow_production_repair_request = no_production_width_narrowing

    _install_bounded_batch_repair(incremental_module)
    _install_contextual_pending_queue(incremental_module)
    _install_outline_cycle_guard(incremental_module)


__all__ = ["install"]
