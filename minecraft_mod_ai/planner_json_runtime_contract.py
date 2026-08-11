from __future__ import annotations

import json
import os
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping, Sequence


_JSON_SCHEMA: ContextVar[dict[str, Any] | None] = ContextVar(
    "mmm_planner_json_schema",
    default=None,
)


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


def _extract_with_safe_empty_defaults(
    module: Any,
    text: str,
    *,
    expected_contracts: Sequence[frozenset[str]],
) -> dict[str, Any]:
    try:
        return module._extract_json(text, expected_contracts=expected_contracts)
    except module.SpecValidationError as original_error:
        pass

    if len(expected_contracts) != 1:
        raise original_error
    expected = expected_contracts[0]
    production_expected = frozenset(module._PRODUCTION_PAGE_CONTRACT)
    if expected != production_expected:
        raise original_error

    # Empty asset/audio/test collections carry no semantics. A model that omitted
    # one of those empty lists still has a recoverable production page. Completion
    # fields are deliberately NOT invented here because they drive host progress.
    for raw in reversed(module._json_objects(text)):
        if not isinstance(raw, dict):
            continue
        candidate = module._normalize_json_candidate(raw)
        if "modules" not in candidate:
            continue
        candidate.setdefault("assets", [])
        candidate.setdefault("audio", [])
        candidate.setdefault("acceptance_tests", [])
        if expected <= frozenset(candidate):
            return {field: candidate[field] for field in expected}
    raise original_error


def _validate_production_progress(
    module: Any,
    page: dict[str, Any],
    request: dict[str, Any] | str,
    expected_contracts: Sequence[frozenset[str]],
) -> None:
    if len(expected_contracts) != 1:
        return
    if expected_contracts[0] != frozenset(module._PRODUCTION_PAGE_CONTRACT):
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

    targets = [
        str(value).strip()
        for value in request.get("current_target_deliverables", [])
        if isinstance(value, str) and str(value).strip()
    ]
    completed = [
        str(value).strip()
        for value in page["completed_deliverables"]
        if isinstance(value, str) and str(value).strip()
    ]
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
        request_text = (
            request
            if isinstance(request, str)
            else json.dumps(request, ensure_ascii=False)
        )
        view = _contract_view(request, expected_contracts)
        schema = _schema_for_value(view) if view is not None else None
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

        previous_diagnostic = ""
        last_error: BaseException | None = None
        for attempt in range(2):
            prompt = (
                system_prompt
                + "\n\nHOST JSON CONTRACT: Return one JSON object with these required "
                + "top-level fields and compatible value types: "
                + contract_text
                + ". Do not omit empty arrays; return them as []."
            )
            if attempt:
                prompt += (
                    "\nREPAIR ONLY THIS PAGE. The previous page was rejected by the host: "
                    + previous_diagnostic
                    + ". Return only the corrected JSON object. Preserve the current "
                    + "target deliverable names exactly in completed_deliverables."
                )

            token = _JSON_SCHEMA.set(schema)
            try:
                text = router.generate_text(
                    "planner",
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": request_text},
                    ],
                    media_paths=media_paths,
                    response_format="json",
                )
            finally:
                _JSON_SCHEMA.reset(token)

            try:
                page = _extract_with_safe_empty_defaults(
                    complete_planner_module,
                    text,
                    expected_contracts=expected_contracts,
                )
                _validate_production_progress(
                    complete_planner_module,
                    page,
                    request,
                    expected_contracts,
                )
                return page
            except complete_planner_module.SpecValidationError as exc:
                last_error = exc
                previous_diagnostic = (
                    f"{exc}; {_candidate_key_summary(complete_planner_module, text)}"
                )

        assert last_error is not None
        raise complete_planner_module.SpecValidationError(
            f"{stage} failed after one page-local repair: {last_error}; "
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
