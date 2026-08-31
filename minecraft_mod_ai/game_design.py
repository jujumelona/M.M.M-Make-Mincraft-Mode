from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .capability_plugins import plugin_manifest
from .central_research import normalize_research_brief
from .model_router import ModelRouter
from .planner import HeuristicPlanner, _proposal_from_model_data
from .spec import Proposal, SpecValidationError

_GAME_DESIGN_FIELDS = (
    "title",
    "pitch",
    "core_loop",
    "progression",
    "combat",
    "mod_context",
    "modules",
    "assets",
    "acceptance_tests",
)
_OPTIONAL_GAME_DESIGN_FIELDS = ("art_direction",)
_REQUEST_PAGE_JSON_TEXT_BYTES = 32 * 1024
_REQUEST_INGESTION_SCHEMA = "mmm/authoritative-request-ingestion-v1"
_REQUEST_PAGE_SCHEMA = "mmm/authoritative-request-page-v1"
_ASSET_KINDS = frozenset({"item", "block", "entity", "gui", "environment"})

_MODULE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plugin_id": {"type": "string", "minLength": 1},
        "status": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
        "requirement_refs": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "implementation_obligations": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": [
        "plugin_id",
        "status",
        "reason",
        "requirement_refs",
        "implementation_obligations",
    ],
    "additionalProperties": False,
}
_ASSET_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "kind": {"type": "string", "minLength": 1},
        "brief": {"type": "string", "minLength": 1},
    },
    "required": ["id", "kind", "brief"],
    "additionalProperties": False,
}
_GAME_DESIGN_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "pitch": {"type": "string", "minLength": 1},
        "core_loop": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "progression": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "combat": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
        "mod_context": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
        "modules": {
            "type": "array",
            "minItems": 1,
            "items": _MODULE_RESPONSE_SCHEMA,
        },
        "assets": {"type": "array", "items": _ASSET_RESPONSE_SCHEMA},
        "acceptance_tests": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "art_direction": {"type": "object"},
    },
    "required": list(_GAME_DESIGN_FIELDS),
    "additionalProperties": False,
}
_GAME_DESIGN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"game_design": _GAME_DESIGN_VALUE_SCHEMA},
    "required": ["game_design"],
    "additionalProperties": False,
}


class GameDesignPlanner:
    """Create the frozen game design before retrieval and implementation planning.

    This module owns the execution path. ModelRouter uses research-first sectioned
    Markdown design; non-agentic routers use the strict host-owned JSON fallback. Runtime
    bootstrap must not replace either generator, readiness validation, or page merge.
    """

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
    ) -> tuple[dict[str, Any], Proposal]:
        if not prompt.strip():
            raise SpecValidationError("프롬프트를 입력해 주세요.")

        page_budget = _request_page_bytes(self.router)
        request_pages = _lossless_request_pages(
            prompt,
            max_json_text_bytes=page_budget,
        )
        if len(request_pages) == 1:
            design = _generate_game_design_once(
                self.router,
                authoritative_prompt=prompt,
                media_paths=media_paths,
                system_prompt=_system_prompt(),
            )
        else:
            design = self._plan_sharded_request(
                prompt,
                request_pages=request_pages,
                media_paths=media_paths,
                page_budget=page_budget,
            )
        design = _validate_ready_design(prompt, design)

        # Existing-project evidence is collected by the host before planning and
        # joined with the independently produced semantic design before any target
        # or reuse decision. It remains private host context and is never inferred
        # or rewritten by the model.
        existing_report = getattr(self.router, "_mmm_existing_project_report", None)
        if isinstance(existing_report, Mapping):
            design = {**design, "_existing_project_report": dict(existing_report)}
        existing_inventory = getattr(
            self.router, "_mmm_existing_project_inventory", None
        )
        inventory_future = getattr(
            self.router, "_mmm_existing_project_inventory_future", None
        )
        if existing_inventory is None and hasattr(inventory_future, "result"):
            inventory = inventory_future.result()
            validate = getattr(inventory, "validate", None)
            if not callable(validate):
                raise SpecValidationError(
                    "Existing-project inventory did not return a validated host object."
                )
            validate()
            to_dict = getattr(inventory, "to_dict", None)
            if not callable(to_dict):
                raise SpecValidationError(
                    "Existing-project inventory cannot be bound to planning."
                )
            existing_inventory = to_dict()
            self.router._mmm_existing_project_inventory = existing_inventory
        if isinstance(existing_inventory, Mapping):
            from .project_inventory import validate_project_inventory_payload

            inventory_payload = validate_project_inventory_payload(existing_inventory)
            self.router._mmm_existing_project_inventory = inventory_payload
            design = {
                **design,
                "_existing_project_inventory": inventory_payload,
                "_existing_snapshot": inventory_payload,
                "_component_catalog": dict(
                    inventory_payload.get("component_catalog") or {}
                ),
            }

        from .evidence_first_planning import build_request_catalog
        from .evidence_request_guard import active_authoritative_request_catalog
        from .reuse_planner import compile_pre_retrieval_plan

        request_catalog = active_authoritative_request_catalog(prompt)
        if request_catalog is None:
            request_catalog = build_request_catalog(prompt, design)
        design = {**design, "_evidence_request_catalog": request_catalog}
        pre_retrieval_plan = compile_pre_retrieval_plan(prompt, design)
        design = {**design, "_pre_retrieval_plan": pre_retrieval_plan}
        print(
            "planning: semantic plan ready before retrieval "
            f"plan_sha256={pre_retrieval_plan['plan_sha256']} "
            f"requirements={len(pre_retrieval_plan['planned_work'])}",
            flush=True,
        )

        research_brief = normalize_research_brief(prompt, design)
        design = {**design, "_research_brief": research_brief}
        build_slice = _deterministic_bootstrap(prompt, design)
        proposal = _proposal_from_model_data(prompt, build_slice)
        if proposal.requested_prompt != prompt:
            proposal = replace(
                proposal,
                requested_prompt=prompt,
                approval_hash="",
            ).with_hash()
        return design, proposal

    def _plan_sharded_request(
        self,
        prompt: str,
        *,
        request_pages: tuple[str, ...],
        media_paths: Sequence[str | Path],
        page_budget: int,
    ) -> dict[str, Any]:
        from . import agentic_research_game_design as agentic

        agentic_mode = agentic.supports_agentic_research_router(self.router)
        page_research: tuple[Mapping[str, Any], ...] = ()
        if agentic_mode:
            from .pre_design_research_pipeline import collect_design_research

            page_count = len(request_pages)
            page_research = tuple(
                collect_design_research(
                    self.router,
                    page_text,
                    trace_metadata={
                        "request_page_index": page_index,
                        "request_page_count": page_count,
                    },
                )
                for page_index, page_text in enumerate(request_pages)
            )

        prompt_bytes = prompt.encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        page_designs: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        byte_offset = 0
        char_offset = 0
        authority = _active_authority()

        for page_index, page_text in enumerate(request_pages):
            encoded_page = page_text.encode("utf-8")
            byte_start = byte_offset
            byte_offset += len(encoded_page)
            char_start = char_offset
            char_offset += len(page_text)
            receipt = {
                "page_index": page_index,
                "page_count": len(request_pages),
                "byte_start": byte_start,
                "byte_end": byte_offset,
                "byte_length": len(encoded_page),
                "content_sha256": hashlib.sha256(encoded_page).hexdigest(),
            }
            request = {
                "schema_version": _REQUEST_PAGE_SCHEMA,
                "full_request": {
                    "sha256": prompt_sha256,
                    "byte_length": len(prompt_bytes),
                    "page_count": len(request_pages),
                },
                "page": receipt,
                "authoritative_request_text": page_text,
            }
            token = _activate_page_authority(
                authority,
                page_text=page_text,
                char_start=char_start,
                char_end=char_offset,
            )
            try:
                design = _generate_game_design_once(
                    self.router,
                    authoritative_prompt=json.dumps(
                        request,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    media_paths=media_paths if page_index == 0 else (),
                    system_prompt=_sharded_design_system_prompt(),
                    fallback_prompt=page_text,
                    precollected_research=(
                        page_research[page_index] if page_research else None
                    ),
                )
            finally:
                _reset_page_authority(token)
            page_designs.append(design)
            receipts.append({**receipt, "design_sha256": _json_sha256(design)})

        if byte_offset != len(prompt_bytes) or "".join(request_pages) != prompt:
            raise SpecValidationError(
                "Authoritative request paging failed its lossless host check."
            )

        merged = _merge_game_design_pages(page_designs)
        chain = hashlib.sha256(
            f"{_REQUEST_INGESTION_SCHEMA}:{prompt_sha256}".encode()
        )
        for receipt in receipts:
            chain.update(
                json.dumps(
                    receipt,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        ingestion = {
            "schema_version": _REQUEST_INGESTION_SCHEMA,
            "prompt_sha256": prompt_sha256,
            "prompt_byte_length": len(prompt_bytes),
            "page_count": len(request_pages),
            "page_json_text_max_bytes": page_budget,
            "pages": receipts,
            "chain_sha256": chain.hexdigest(),
            "authority": {
                "semantic_input": "user_request",
                "receipts": "host_computed",
                "model_output": "descriptive_values_only",
                "execution_authority": False,
            },
        }
        result: dict[str, Any] = {**merged, "_request_ingestion": ingestion}
        if page_research:
            result["_pre_design_research"] = _sharded_research_ledger(
                prompt,
                request_pages,
                page_research,
            )
        return result


GameDesignPlanner.plan._mmm_host_owned_template = True  # type: ignore[attr-defined]


def _generate_game_design_once(
    router: Any,
    *,
    authoritative_prompt: str,
    media_paths: Sequence[str | Path],
    system_prompt: str,
    fallback_prompt: str | None = None,
    precollected_research: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single native generation entrypoint; runtime installers never replace it."""
    from . import agentic_research_game_design as agentic

    design_prompt = str(fallback_prompt or authoritative_prompt)
    if agentic.supports_agentic_research_router(router):
        from .pre_design_research_pipeline import collect_design_research

        research = (
            dict(precollected_research)
            if isinstance(precollected_research, Mapping)
            else collect_design_research(router, design_prompt)
        )
        design = agentic.generate_sectioned_game_design(
            __import__(__name__, fromlist=["*"]),
            router,
            design_prompt,
            media_paths=media_paths,
            research=research,
        )
        canonical = _canonical_game_design(design)
        result: dict[str, Any] = {
            **canonical,
            "_pre_design_research": research,
            "_research_brief": dict(research.get("research_brief") or {}),
        }
        return _validate_ready_design(design_prompt, result)

    authority = _active_authority()
    effective_system_prompt = system_prompt
    if authority is not None:
        _authority_prompt, ledger = authority
        effective_system_prompt = _augment_system_prompt(system_prompt, ledger)
    fallback = _game_design_skeleton(design_prompt)
    messages = [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": authoritative_prompt},
    ]
    try:
        raw = router.generate_text(
            "planner",
            messages,
            media_paths=media_paths,
            response_format="json",
            response_schema=_GAME_DESIGN_RESPONSE_SCHEMA,
            enable_tools=False,
        )
        candidate = _extract_model_design(str(raw))
        result = _merge_model_design(fallback, candidate)
    except (TypeError, ValueError, RuntimeError):
        result = fallback
    return _validate_ready_design(design_prompt, result)


def _generate_sharded_design_page(
    router: Any,
    *,
    request_text: str,
    media_paths: Sequence[str | Path],
    page_index: int,
    page_count: int,
) -> dict[str, Any]:
    del page_index, page_count
    return _generate_game_design_once(
        router,
        authoritative_prompt=request_text,
        media_paths=media_paths,
        system_prompt=_sharded_design_system_prompt(),
    )


def _active_authority() -> tuple[str, tuple[dict[str, Any], ...]] | None:
    from . import evidence_request_guard as request_guard

    active = request_guard._ACTIVE_REQUEST_CATALOG.get()
    if active is None:
        return None
    prompt, catalog = active
    raw_requirements = catalog.get("requirements", [])
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return None
    ledger: list[dict[str, Any]] = []
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        span = raw.get("source_span")
        authored_text = (
            str(span.get("text") or "").strip()
            if isinstance(span, Mapping)
            else str(raw.get("statement") or "").strip()
        )
        ledger.append(
            {
                "requirement_id": requirement_id,
                "capability": str(raw.get("capability") or "").strip(),
                "authored_text": authored_text,
                "semantic_statement": str(raw.get("semantic_statement") or "").strip(),
                "acceptance": list(raw.get("acceptance") or []),
                "source_span": dict(span) if isinstance(span, Mapping) else {},
            }
        )
    return (prompt, tuple(ledger)) if ledger else None


def _activate_page_authority(
    authority: tuple[str, tuple[dict[str, Any], ...]] | None,
    *,
    page_text: str,
    char_start: int,
    char_end: int,
) -> Any:
    if authority is None:
        return None
    from . import evidence_request_guard as request_guard

    _prompt, ledger = authority
    selected = []
    for item in ledger:
        span = item.get("source_span")
        start = span.get("char_start") if isinstance(span, Mapping) else None
        end = span.get("char_end") if isinstance(span, Mapping) else None
        if type(start) is int and type(end) is int and start < char_end and end > char_start:
            selected.append(dict(item))
    if not selected:
        return None
    return request_guard._ACTIVE_REQUEST_CATALOG.set(
        (page_text, {"requirements": selected})
    )


def _reset_page_authority(token: Any) -> None:
    if token is None:
        return
    from . import evidence_request_guard as request_guard

    request_guard._ACTIVE_REQUEST_CATALOG.reset(token)


def _augment_system_prompt(
    system_prompt: str,
    ledger: Sequence[Mapping[str, Any]],
) -> str:
    compact = [
        {
            "requirement_id": item["requirement_id"],
            "capability": item.get("capability", ""),
            "authored_text": item.get("authored_text", ""),
            "semantic_statement": item.get("semantic_statement", ""),
            "acceptance": item.get("acceptance", []),
        }
        for item in ledger
    ]
    return system_prompt + (
        "\n\nFROZEN REQUIREMENT AUTHORITY (host-owned; do not rewrite IDs):\n"
        + json.dumps(compact, ensure_ascii=False, sort_keys=True)
        + "\nEvery approved requirement must appear in at least one modules[].requirement_refs. "
        "Each such module must contain concrete implementation_obligations. Preserve the "
        "authored behavior; do not invent target versions, mappings, API signatures, or "
        "unrequested mechanics. All required game-design fields must be substantively filled."
    )


def _assert_minimum_design_depth(design: Mapping[str, Any]) -> None:
    for field in ("core_loop", "progression", "acceptance_tests"):
        value = design.get(field)
        if not isinstance(value, list) or not any(str(item).strip() for item in value):
            raise SpecValidationError(
                f"design readiness failed: {field} is empty after model normalization"
            )
    if _active_authority() is not None:
        modules = design.get("modules")
        if not isinstance(modules, list) or not modules:
            raise SpecValidationError(
                "design readiness failed: modules are empty after model normalization"
            )


def _validate_ready_design(prompt: str, design: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(design, Mapping):
        raise SpecValidationError("game design generation returned a non-object result")
    result = dict(design)
    _assert_minimum_design_depth(result)
    authority = _active_authority()
    if authority is None or authority[0] != prompt:
        return result
    from . import agentic_research_game_design as agentic

    return agentic._validate_requirement_coverage(result, authority[1])


def _sharded_research_ledger(
    prompt: str,
    request_pages: Sequence[str],
    page_research: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for page_index, (page_text, research) in enumerate(
        zip(request_pages, page_research, strict=True)
    ):
        pages.append(
            {
                "page_index": page_index,
                "content_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
                "research_sha256": str(research.get("research_sha256") or ""),
                "model_view_sha256": str(research.get("model_view_sha256") or ""),
                "research": dict(research),
            }
        )
    ledger: dict[str, Any] = {
        "schema_version": "mmm/agentic-pre-design-research-sharded-v1",
        "request_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "page_count": len(pages),
        "pages": pages,
        "method": {
            "ordering": "all page research completes before any page design",
            "model_context": "each page design receives its own bounded research view",
            "host_ledger": "all page research views retained for provenance",
        },
    }
    rendered = json.dumps(
        ledger,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    ledger["research_sha256"] = "sha256:" + hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest()
    return ledger


def _game_design_skeleton(prompt: str) -> dict[str, Any]:
    heuristic = HeuristicPlanner().plan(prompt)
    return {
        "title": heuristic.spec.mod_name or "Generated Mod",
        "pitch": heuristic.spec.summary or "Implement the requested Minecraft mod.",
        "core_loop": [],
        "progression": [],
        "combat": {},
        "mod_context": {},
        "modules": [],
        "assets": [],
        "acceptance_tests": ["The requested behavior is observable in Minecraft."],
    }


def _extract_model_design(text: str) -> dict[str, Any]:
    objects = _json_objects(text)
    for value in reversed(objects):
        nested = value.get("game_design")
        if isinstance(nested, dict):
            return nested
        if any(field in value for field in _GAME_DESIGN_FIELDS):
            return value
    return {}


def _merge_model_design(
    skeleton: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(skeleton)
    for field in ("title", "pitch"):
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()
    for field in ("core_loop", "progression", "acceptance_tests"):
        values = _strings(candidate.get(field))
        if values:
            result[field] = values
    for field in ("combat", "mod_context"):
        value = candidate.get(field)
        if isinstance(value, Mapping):
            result[field] = {
                str(key): values
                for key, raw in value.items()
                if (values := _strings(raw))
            }
    modules = _modules(candidate.get("modules"))
    if modules:
        result["modules"] = modules
    assets = _assets(candidate.get("assets"))
    if assets:
        result["assets"] = assets
    art = candidate.get("art_direction")
    if isinstance(art, Mapping) and art:
        result["art_direction"] = dict(art)
    _validate_design(result)
    return result


def _modules(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    catalog = {
        str(item.get("plugin_id", "")).strip(): str(item.get("status", "")).strip()
        for item in _planner_plugin_manifest()["plugins"]
        if isinstance(item, Mapping) and str(item.get("plugin_id", "")).strip()
    }
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        plugin_id = _identifier(
            raw.get("plugin_id") or raw.get("module_id") or raw.get("id")
        )
        if not plugin_id or plugin_id in seen:
            continue
        reason = _first_string(
            raw,
            "reason",
            "description",
            "purpose",
            "brief",
            "summary",
            "name",
        ) or plugin_id
        status = str(raw.get("status") or catalog.get(plugin_id) or "custom").strip()
        if status not in {"implemented", "custom"}:
            status = catalog.get(plugin_id, "custom")
        output.append(
            {
                "plugin_id": plugin_id,
                "status": status,
                "reason": reason,
                "requirement_refs": _strings(raw.get("requirement_refs")),
                "implementation_obligations": _strings(
                    raw.get("implementation_obligations")
                ),
            }
        )
        seen.add(plugin_id)
    return output


def _assets(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        asset_id = _identifier(raw.get("id") or raw.get("asset_id"))
        kind = str(raw.get("kind") or "").strip().casefold()
        brief = _first_string(raw, "brief", "prompt", "description")
        if not asset_id or asset_id in seen or kind not in _ASSET_KINDS or not brief:
            continue
        output.append({"id": asset_id, "kind": kind, "brief": brief})
        seen.add(asset_id)
    return output


def _merge_game_design_pages(pages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not pages:
        return _game_design_skeleton("requested mod")
    result = dict(pages[0])
    for page in pages[1:]:
        for field in ("core_loop", "progression", "acceptance_tests"):
            result[field] = list(
                dict.fromkeys([*_strings(result.get(field)), *_strings(page.get(field))])
            )
        for field in ("combat", "mod_context"):
            merged_map: dict[str, list[str]] = {
                key: list(values)
                for key, values in result.get(field, {}).items()
                if isinstance(key, str) and isinstance(values, list)
            }
            raw_map = page.get(field)
            if isinstance(raw_map, Mapping):
                for key, raw in raw_map.items():
                    values = _strings(raw)
                    if values:
                        merged_map[str(key)] = list(
                            dict.fromkeys([*merged_map.get(str(key), []), *values])
                        )
            result[field] = merged_map
        result["modules"] = _merge_identity_records(
            result.get("modules"), page.get("modules"), "plugin_id"
        )
        result["assets"] = _merge_identity_records(
            result.get("assets"), page.get("assets"), "id"
        )
        if "art_direction" not in result and isinstance(page.get("art_direction"), Mapping):
            result["art_direction"] = dict(page["art_direction"])
    _validate_design(result)
    return result


def _merge_identity_records(left: Any, right: Any, key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in [
        *(left if isinstance(left, list) else []),
        *(right if isinstance(right, list) else []),
    ]:
        if not isinstance(raw, Mapping):
            continue
        identity = str(raw.get(key) or "").strip()
        if not identity or identity in seen:
            continue
        output.append(dict(raw))
        seen.add(identity)
    return output


def _request_page_bytes(router: ModelRouter | None = None, role: str = "planner") -> int:
    if router is not None:
        try:
            config = router.registry.role(router.profile, role)
            context = int(getattr(config, "max_context", 0) or 0)
            if context > 0:
                available_tokens = max(1024, context - 2048)
                return max(4 * 1024, min(64 * 1024, int(available_tokens * 3.5)))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    return _REQUEST_PAGE_JSON_TEXT_BYTES


def _authoritative_request_pages(
    prompt: str,
    router: ModelRouter | None = None,
) -> tuple[str, ...]:
    return _lossless_request_pages(
        prompt,
        max_json_text_bytes=_request_page_bytes(router),
    )


def _lossless_request_pages(
    prompt: str,
    *,
    max_json_text_bytes: int,
) -> tuple[str, ...]:
    if not prompt:
        return ()
    if _json_text_bytes(prompt) <= max_json_text_bytes:
        return (prompt,)
    pages: list[str] = []
    start = 0
    while start < len(prompt):
        end = start
        encoded_size = 0
        preferred_cut: int | None = None
        while end < len(prompt):
            size = _json_character_bytes(prompt[end])
            if encoded_size + size > max_json_text_bytes:
                break
            encoded_size += size
            end += 1
            if prompt[end - 1].isspace() and encoded_size >= max_json_text_bytes // 2:
                preferred_cut = end
        if end == start:
            raise SpecValidationError(
                "One request character exceeded the bounded JSON page budget."
            )
        cut = preferred_cut if end < len(prompt) and preferred_cut is not None else end
        pages.append(prompt[start:cut])
        start = cut
    if "".join(pages) != prompt:
        raise SpecValidationError("Authoritative request paging was not lossless.")
    return tuple(pages)


def _json_character_bytes(character: str) -> int:
    if len(character) != 1:
        raise ValueError("_json_character_bytes requires exactly one character")
    codepoint = ord(character)
    if character in {'"', "\\", "\x08", "\x0c", "\n", "\r", "\t"}:
        return 2
    if codepoint < 32:
        return 6
    return len(character.encode("utf-8"))


def _json_text_bytes(value: str) -> int:
    """Return exact UTF-8 byte length of one JSON string using the C encoder."""
    if not isinstance(value, str):
        raise TypeError("_json_text_bytes requires a string")
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8")) - 2


def _system_prompt() -> str:
    manifest = json.dumps(
        _planner_plugin_manifest(),
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        "Fill values inside the host-owned game_design template only. "
        "Return one JSON object with top-level game_design. Preserve requested systems; "
        "do not invent unrelated features. User-facing text must use the user's language. "
        "Identifiers stay English snake_case. The host owns platform selection, paging, "
        "required fields, validation and fallback.\nPlugin catalog:\n" + manifest
    )


def _sharded_design_system_prompt() -> str:
    return (
        _system_prompt()
        + "\nThis input is one lossless host page of a larger request. Describe only requirements "
        "present on this page. Do not invent cross-page dependencies or completion claims."
    )


def _planner_plugin_manifest() -> dict[str, Any]:
    manifest = plugin_manifest()
    return {
        "product_scope": manifest["product_scope"],
        "standalone_map_generation": manifest["standalone_map_generation"],
        "plugins": [
            {"plugin_id": plugin["plugin_id"], "status": plugin["status"]}
            for plugin in manifest["plugins"]
        ],
    }


def _canonical_game_design(design: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: design[field] for field in _GAME_DESIGN_FIELDS if field in design}
    for field in _OPTIONAL_GAME_DESIGN_FIELDS:
        if field in design:
            result[field] = design[field]
    _validate_design(result)
    return result


def _validate_design(design: dict[str, Any]) -> None:
    for field in ("title", "pitch"):
        value = design.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SpecValidationError(f"game_design.{field} must be a non-empty string")
    for field in ("core_loop", "progression", "acceptance_tests", "modules", "assets"):
        if not isinstance(design.get(field), list):
            raise SpecValidationError(f"game_design.{field} must be a list")
    for field in ("combat", "mod_context"):
        if not isinstance(design.get(field), dict):
            raise SpecValidationError(f"game_design.{field} must be an object")


def _deterministic_bootstrap(prompt: str, design: Mapping[str, Any]) -> dict[str, Any]:
    proposal = HeuristicPlanner().plan(prompt)
    spec = proposal.spec
    title = str(design.get("title") or spec.mod_name).strip()
    pitch = str(design.get("pitch") or spec.summary).strip()
    normalized = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in title.lower()
    )
    stem = "_".join(part for part in normalized.split("_") if part)
    if not stem:
        stem = f"mmm_{hashlib.sha256(title.encode('utf-8')).hexdigest()[:10]}"
    if not stem[0].isalpha():
        stem = f"mmm_{stem}"
    mod_id = f"{stem[:55].rstrip('_')}_mod"
    return {
        "mod_id": mod_id,
        "mod_name": title,
        "package_name": f"ai.minecraft.generated.{mod_id}",
        "summary": pitch,
        "contents": [
            {
                "content_id": content.content_id,
                "kind": content.kind.value,
                "display_name_en": content.display_name_en,
                "display_name_ko": content.display_name_ko,
                "color": content.color,
                "recipe": content.recipe,
            }
            for content in spec.contents
        ],
        "deferred_capabilities": [
            deferred.capability for deferred in proposal.deferred_requests
        ],
    }


def _json_objects(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).replace("```", "").strip()
    decoder = json.JSONDecoder(strict=False)
    values: list[dict[str, Any]] = []
    try:
        value = json.loads(cleaned, strict=False)
        if isinstance(value, dict):
            return [value]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    )


def _identifier(value: Any) -> str:
    text = re.sub(
        r"[^a-z0-9_]+", "_", str(value or "").strip().lower()
    ).strip("_")
    if not text:
        return ""
    if not text[0].isalpha():
        text = f"feature_{text}"
    return text[:63]


def _first_string(value: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


__all__ = ["GameDesignPlanner"]
