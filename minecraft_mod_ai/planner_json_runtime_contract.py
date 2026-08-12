from __future__ import annotations

import json
import os
from contextvars import ContextVar
from functools import wraps
from typing import Any, Sequence

from .planner_strict_json_contract import _extract_one_complete_object


_JSON_SCHEMA: ContextVar[dict[str, Any] | None] = ContextVar(
    "mmm_planner_json_schema",
    default=None,
)

_PRODUCTION_FIELDS = frozenset(
    {
        "modules",
        "assets",
        "audio",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
)

_GENERIC_PAGE_ATTEMPTS = 3
_PRODUCTION_PAGE_ATTEMPTS = 5


def _schema_for_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {
            "type": "array",
            "items": _schema_for_value(value[0]) if value else {},
        }
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {
                str(key): _schema_for_value(item)
                for key, item in value.items()
            },
            "required": [str(key) for key in value],
            "additionalProperties": True,
        }
    if value is None:
        return {"type": "null"}
    return {}


def _schema_for_contract(view: dict[str, Any]) -> dict[str, Any]:
    if frozenset(view) != _PRODUCTION_FIELDS:
        return _schema_for_value(view)
    string_array = {"type": "array", "items": {"type": "string"}}
    module_item = {
        "type": "object",
        "properties": {
            "module_id": {"type": "string"},
            "kind": {"type": "string"},
            "config": {"type": "object", "additionalProperties": True},
            "depends_on": string_array,
            "required_gates": string_array,
            "implements_deliverables": string_array,
        },
        "required": [
            "module_id",
            "kind",
            "config",
            "depends_on",
            "required_gates",
        ],
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "properties": {
            "completed_deliverables": string_array,
            "complete": {"type": "boolean"},
            "next_cursor": {"type": "string"},
            "modules": {"type": "array", "items": module_item},
            "assets": {"type": "array", "items": {"type": "object"}},
            "audio": {"type": "array", "items": {"type": "object"}},
            "acceptance_tests": string_array,
        },
        "required": list(view),
        "additionalProperties": False,
    }


def _contract_view(
    request: dict[str, Any] | str,
    expected_contracts: Sequence[frozenset[str]],
) -> dict[str, Any] | None:
    if len(expected_contracts) != 1 or not isinstance(request, dict):
        return None
    raw = request.get("contract")
    if not isinstance(raw, dict):
        return None
    expected = expected_contracts[0]
    if frozenset(raw) != expected:
        return None
    return raw


def _is_production_page(
    module: Any,
    expected_contracts: Sequence[frozenset[str]],
) -> bool:
    return (
        len(expected_contracts) == 1
        and expected_contracts[0] == frozenset(module._PRODUCTION_PAGE_CONTRACT)
    )


def _candidate_key_summary(module: Any, text: str) -> str:
    try:
        objects = module._json_objects(text)
    except Exception:
        return "no parseable JSON object"
    if not objects:
        return "no parseable JSON object"
    summaries: list[str] = []
    for value in objects[-3:]:
        if isinstance(value, dict):
            summaries.append("[" + ", ".join(sorted(map(str, value))) + "]")
    return "top-level fields=" + " | ".join(summaries)


def _alias_only(module: Any, raw: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(raw)
    for key, value in list(candidate.items()):
        alias = module._FIELD_ALIASES.get(key)
        if alias and alias not in candidate:
            candidate[alias] = value
    return candidate


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if isinstance(item, str) and str(item).strip()
    ]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _target_names(request: dict[str, Any] | str) -> list[str]:
    if not isinstance(request, dict):
        return []
    return _string_list(request.get("current_target_deliverables", []))


def _remaining_names(request: dict[str, Any] | str) -> list[str]:
    if not isinstance(request, dict):
        return []
    remaining = _string_list(request.get("remaining_deliverables", []))
    return remaining or _target_names(request)


def _output_ids(
    modules: Sequence[dict[str, Any]],
    assets: Sequence[dict[str, Any]],
    audio: Sequence[dict[str, Any]],
) -> set[str]:
    result: set[str] = set()
    for item, key in (
        *((item, "module_id") for item in modules),
        *((item, "asset_id") for item in assets),
        *((item, "sound_id") for item in audio),
    ):
        value = str(item.get(key, "")).strip()
        if value:
            result.add(value)
    return result


def _derive_completed_deliverables(
    candidate: dict[str, Any],
    *,
    targets: Sequence[str],
    modules: Sequence[dict[str, Any]],
    assets: Sequence[dict[str, Any]],
    audio: Sequence[dict[str, Any]],
    acceptance_tests: Sequence[str],
) -> list[str]:
    target_set = set(targets)
    completed: set[str] = {
        value
        for value in _string_list(candidate.get("completed_deliverables", []))
        if value in target_set
    }

    ids = _output_ids(modules, assets, audio)
    tests = set(acceptance_tests)
    completed.update(value for value in targets if value in ids or value in tests)

    for item in [*modules, *assets, *audio]:
        claims = item.get("implements_deliverables")
        if not isinstance(claims, (list, tuple)):
            claims = item.get("implements")
        if isinstance(claims, (list, tuple)):
            completed.update(
                str(value).strip()
                for value in claims
                if str(value).strip() in target_set
            )

    evidence = candidate.get("deliverable_evidence")
    if isinstance(evidence, dict):
        for target, raw_evidence in evidence.items():
            target_name = str(target).strip()
            if target_name not in target_set or not isinstance(raw_evidence, dict):
                continue
            referenced = set(
                _string_list(raw_evidence.get("module_ids", []))
                + _string_list(raw_evidence.get("asset_ids", []))
                + _string_list(raw_evidence.get("audio_ids", []))
                + _string_list(raw_evidence.get("acceptance_tests", []))
            )
            if referenced & (ids | tests):
                completed.add(target_name)

    # Once recovery has narrowed the host request to one deliverable, every emitted
    # implementation/test artifact belongs to that sole target.  This is host-owned
    # attribution, not a model-invented completion claim.
    if (
        len(targets) == 1
        and targets[0] not in completed
        and (modules or assets or audio or acceptance_tests)
    ):
        completed.add(targets[0])

    return [target for target in targets if target in completed]


def _extract_production_page_with_host_bookkeeping(
    module: Any,
    text: str,
    request: dict[str, Any] | str,
) -> dict[str, Any]:
    try:
        raw = _extract_one_complete_object(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise module.SpecValidationError(
            f"Production page did not contain one complete strict JSON object: {exc}"
        ) from exc

    candidate = _alias_only(module, raw)
    modules = _dict_list(candidate.get("modules", []))
    assets = _dict_list(candidate.get("assets", []))
    audio = _dict_list(candidate.get("audio", []))
    acceptance_tests = _string_list(candidate.get("acceptance_tests", []))
    targets = _target_names(request)
    completed = _derive_completed_deliverables(
        candidate,
        targets=targets,
        modules=modules,
        assets=assets,
        audio=audio,
        acceptance_tests=acceptance_tests,
    )

    remaining = _remaining_names(request)
    completed_set = set(completed)
    still_remaining = [value for value in remaining if value not in completed_set]
    complete = not still_remaining
    next_cursor = "" if complete else f"host_remaining_{len(still_remaining)}"

    return {
        "modules": modules,
        "assets": assets,
        "audio": audio,
        "acceptance_tests": acceptance_tests,
        "completed_deliverables": completed,
        "complete": complete,
        "next_cursor": next_cursor,
    }


def _extract_with_safe_empty_defaults(
    module: Any,
    text: str,
    *,
    expected_contracts: Sequence[frozenset[str]],
) -> dict[str, Any]:
    production_expected = frozenset(module._PRODUCTION_PAGE_CONTRACT)
    if len(expected_contracts) != 1 or expected_contracts[0] != production_expected:
        return module._extract_json(text, expected_contracts=expected_contracts)

    # Kept for compatibility with direct callers. The installed production path below
    # uses _extract_production_page_with_host_bookkeeping because request context is
    # required to compute progress safely.
    for raw in reversed(module._json_objects(text)):
        if not isinstance(raw, dict):
            continue
        candidate = _alias_only(module, raw)
        candidate.setdefault("assets", [])
        candidate.setdefault("audio", [])
        candidate.setdefault("acceptance_tests", [])
        required_semantic = {
            "modules",
            "completed_deliverables",
            "complete",
            "next_cursor",
        }
        if required_semantic <= frozenset(candidate):
            return {field: candidate[field] for field in production_expected}
    raise module.SpecValidationError(
        "Production page did not contain all host-required semantic fields."
    )


def _validate_production_progress(
    module: Any,
    page: dict[str, Any],
    request: dict[str, Any] | str,
    expected_contracts: Sequence[frozenset[str]],
) -> None:
    if not _is_production_page(module, expected_contracts):
        return
    if not isinstance(request, dict):
        raise module.SpecValidationError(
            "Production page validation requires its host request object."
        )

    for field in (
        "modules",
        "assets",
        "audio",
        "acceptance_tests",
        "completed_deliverables",
    ):
        if not isinstance(page.get(field), list):
            raise module.SpecValidationError(
                f"Production page field {field} must be a list."
            )
    if type(page.get("complete")) is not bool:
        raise module.SpecValidationError("Production page complete must be boolean.")
    if not isinstance(page.get("next_cursor"), str):
        raise module.SpecValidationError("Production page next_cursor must be a string.")

    targets = _target_names(request)
    completed = _string_list(page["completed_deliverables"])
    target_set = set(targets)
    invalid = [value for value in completed if value not in target_set]
    if invalid:
        raise module.SpecValidationError(
            "Production page completed_deliverables contains names outside the "
            f"current host target: {invalid[:4]}"
        )
    if targets and not completed:
        raise module.SpecValidationError(
            "Production page made no host-verifiable deliverable progress."
        )
    if not (
        page["modules"]
        or page["assets"]
        or page["audio"]
        or page["acceptance_tests"]
    ):
        raise module.SpecValidationError(
            "Production page declared completion without any implementation or test output."
        )

    remaining = _remaining_names(request)
    expected_complete = not [
        value for value in remaining if value not in set(completed)
    ]
    if page["complete"] != expected_complete:
        raise module.SpecValidationError(
            "Production page host completion bookkeeping is inconsistent."
        )
    if page["complete"] and page["next_cursor"]:
        raise module.SpecValidationError(
            "Complete production page must not carry a continuation cursor."
        )
    if not page["complete"] and not page["next_cursor"]:
        raise module.SpecValidationError(
            "Incomplete production page requires a host continuation cursor."
        )


def _narrow_production_repair_request(
    request: dict[str, Any] | str,
    attempt: int,
) -> dict[str, Any] | str:
    if not isinstance(request, dict):
        return request
    targets = _target_names(request)
    if not targets:
        return request

    # Keep normal production pages comfortably below an 8k-output ceiling.  The
    # caller may group four deliverables for throughput, but two implementation
    # targets per decode avoids spending an entire long generation on an object that
    # is likely to be truncated near max_new_tokens.
    width = 2 if attempt == 0 else 1
    selected = targets[:width]
    if selected == targets:
        return request

    # A rejected page is never regenerated at the same width.  Recovery collapses
    # to one outstanding deliverable; the outer host loop returns for the rest after
    # this page is verified.
    narrowed = dict(request)
    narrowed["current_target_deliverable"] = selected[0]
    narrowed["current_target_deliverables"] = selected
    narrowed["remaining_deliverables"] = selected
    narrowed["total_remaining"] = len(selected)
    narrowed["cursor"] = ""
    return narrowed


def _attempt_budget(production_page: bool) -> int:
    env_name = (
        "MMM_PRODUCTION_JSON_ATTEMPTS"
        if production_page
        else "MMM_PLANNER_JSON_ATTEMPTS"
    )
    default = _PRODUCTION_PAGE_ATTEMPTS if production_page else _GENERIC_PAGE_ATTEMPTS
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(2, min(value, 12))


def _install_server_schema_and_strict_mtp(
    hardware_policy_module: Any,
) -> None:
    original_payload = hardware_policy_module._server_payload
    if not getattr(original_payload, "_mmm_json_schema_payload", False):

        @wraps(original_payload)
        def payload_with_schema(adapter: Any, request: Any) -> dict[str, Any]:
            payload = original_payload(adapter, request)
            schema = _JSON_SCHEMA.get()
            if schema is not None and getattr(request, "response_format", None) == "json":
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": schema,
                }
            return payload

        payload_with_schema._mmm_json_schema_payload = True
        hardware_policy_module._server_payload = payload_with_schema

    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    current_generate = LlamaCppAdapter.generate
    if getattr(current_generate, "_mmm_final_strict_mtp", False):
        return

    @wraps(current_generate)
    def final_strict_mtp_generate(self: Any, request: Any) -> str:
        from .colab_mtp_server import (
            SERVER_API_URL,
            colab_mtp_server_enabled,
            colab_mtp_server_running,
            start_colab_mtp_server,
        )

        explicit = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
        if colab_mtp_server_enabled():
            if colab_mtp_server_running():
                explicit = SERVER_API_URL
                os.environ["LLAMA_SERVER_URL"] = explicit
            else:
                explicit = start_colab_mtp_server(self.config).strip().rstrip("/")
            if not explicit:
                raise RuntimeError("MTP server is enabled but produced no server URL.")
            # Never fall back to an in-process second GGUF when the user explicitly
            # enabled the managed MTP server. Surface the actual server failure.
            return hardware_policy_module._strict_server_generate(
                self,
                request,
                explicit,
            )

        if explicit:
            return hardware_policy_module._strict_server_generate(
                self,
                request,
                explicit,
            )
        return current_generate(self, request)

    final_strict_mtp_generate._mmm_final_strict_mtp = True
    LlamaCppAdapter.generate = final_strict_mtp_generate


def _install_planner_page_contract(complete_planner_module: Any) -> None:
    original = complete_planner_module._generate_json_page_with_repair
    if getattr(original, "_mmm_schema_constrained_pages", False):
        return

    @wraps(original)
    def generate_json_page_with_schema(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        view = _contract_view(request, expected_contracts)
        schema = _schema_for_contract(view) if view is not None else None
        required_fields = (
            list(view)
            if view is not None
            else sorted(set().union(*(set(item) for item in expected_contracts)))
        )
        contract_text = (
            json.dumps(view, ensure_ascii=False, separators=(",", ":"))
            if view is not None
            else "required top-level fields: " + ", ".join(required_fields)
        )
        production_page = _is_production_page(
            complete_planner_module,
            expected_contracts,
        )
        attempts = _attempt_budget(production_page)

        previous_diagnostic = ""
        last_error: BaseException | None = None
        for attempt in range(attempts):
            attempt_request = (
                _narrow_production_repair_request(request, attempt)
                if production_page
                else request
            )
            request_text = (
                attempt_request
                if isinstance(attempt_request, str)
                else json.dumps(attempt_request, ensure_ascii=False)
            )
            prompt = (
                system_prompt
                + "\n\nHOST JSON CONTRACT: Return one JSON object with these required "
                + "top-level fields and compatible value types: "
                + contract_text
                + ". Do not omit empty arrays; return them as []."
            )
            if production_page:
                original_targets = _target_names(request)
                active_targets = _target_names(attempt_request)
                if active_targets != original_targets:
                    prompt += (
                        "\nACTIVE HOST PAGE WIDTH OVERRIDE: Earlier batching text may name "
                        + "more deliverables, but this decode must implement ONLY "
                        + json.dumps(active_targets, ensure_ascii=False)
                        + ". The outer host loop will schedule the remaining deliverables "
                        + "after this page succeeds."
                    )
            if attempt:
                prompt += (
                    "\nREPAIR THIS PAGE. The previous page was rejected by the host: "
                    + previous_diagnostic
                    + ". Do not repeat the previous oversized/invalid response. Return "
                    + "one corrected JSON object only."
                )
                if production_page:
                    repair_targets = _target_names(attempt_request)
                    prompt += (
                        " RECOVERY MODE is host-narrowed to exactly these deliverables: "
                        + json.dumps(repair_targets, ensure_ascii=False)
                        + ". Implement only those targets in this response. Keep config "
                        + "structured and concise; do not duplicate request prose. Put the "
                        + "exact target names in completed_deliverables. The host owns "
                        + "pagination bookkeeping, but still emit complete and next_cursor "
                        + "with valid JSON types."
                    )
                else:
                    prompt += (
                        " Preserve the exact host-required top-level field names and types."
                    )

            token = _JSON_SCHEMA.set(schema)
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
                _JSON_SCHEMA.reset(token)

            try:
                if production_page:
                    page = _extract_production_page_with_host_bookkeeping(
                        complete_planner_module,
                        text,
                        attempt_request,
                    )
                else:
                    page = _extract_with_safe_empty_defaults(
                        complete_planner_module,
                        text,
                        expected_contracts=expected_contracts,
                    )
                _validate_production_progress(
                    complete_planner_module,
                    page,
                    attempt_request,
                    expected_contracts,
                )
                return page
            except complete_planner_module.SpecValidationError as exc:
                last_error = exc
                previous_diagnostic = (
                    f"{exc}; {_candidate_key_summary(complete_planner_module, text)}"
                )

        assert last_error is not None
        repair_count = attempts - 1
        raise complete_planner_module.SpecValidationError(
            f"{stage} failed after {repair_count} page-local repairs: {last_error}; "
            + previous_diagnostic
        ) from last_error

    generate_json_page_with_schema._mmm_schema_constrained_pages = True
    complete_planner_module._generate_json_page_with_repair = generate_json_page_with_schema


def install(complete_planner_module: Any) -> None:
    """Constrain planner JSON at decode time and keep explicit MTP fail-closed."""

    if getattr(complete_planner_module, "_mmm_planner_json_runtime_contract", False):
        return
    from . import llama_server_hardware_policy as hardware_policy_module

    _install_server_schema_and_strict_mtp(hardware_policy_module)
    _install_planner_page_contract(complete_planner_module)
    complete_planner_module._mmm_planner_json_runtime_contract = True


__all__ = ["install"]
