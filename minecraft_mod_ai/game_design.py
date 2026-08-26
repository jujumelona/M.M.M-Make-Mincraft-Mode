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

_GAME_DESIGN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"game_design": {"type": "object"}},
    "required": ["game_design"],
    "additionalProperties": False,
}


class GameDesignPlanner:
    """Create game design with a host-owned schema and one model call per page.

    The model may fill values but never controls required fields, paging receipts, or
    validation flow. Malformed/partial output is merged into a deterministic skeleton;
    there is no model repair/retry loop.
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

        # Existing-project evidence is collected by the host before planning and
        # joined with the independently produced semantic design before any target
        # or reuse decision.  It remains private host context (underscore-prefixed)
        # and is never inferred or rewritten by the model.
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

            inventory_payload = validate_project_inventory_payload(
                existing_inventory
            )
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

        design = {
            **design,
            "_evidence_request_catalog": build_request_catalog(prompt, design),
        }

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
        prompt_bytes = prompt.encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        page_designs: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        byte_offset = 0

        for page_index, page_text in enumerate(request_pages):
            encoded_page = page_text.encode("utf-8")
            byte_start = byte_offset
            byte_offset += len(encoded_page)
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
            )
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
        return {**merged, "_request_ingestion": ingestion}


# Transitional marker: prevents the retired bind-time agentic wrapper from replacing
# this host-owned implementation while bootstrap cleanup proceeds.
GameDesignPlanner.plan._mmm_agentic_research_sectioned = True  # type: ignore[attr-defined]
GameDesignPlanner.plan._mmm_host_owned_template = True  # type: ignore[attr-defined]


def _generate_game_design_once(
    router: Any,
    *,
    authoritative_prompt: str,
    media_paths: Sequence[str | Path],
    system_prompt: str,
    fallback_prompt: str | None = None,
) -> dict[str, Any]:
    fallback = _game_design_skeleton(fallback_prompt or authoritative_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
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
    except (TypeError, ValueError, RuntimeError):
        return fallback
    candidate = _extract_model_design(str(raw))
    return _merge_model_design(fallback, candidate)


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


def _modules(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    catalog = {
        str(item.get("plugin_id", "")).strip(): str(item.get("status", "")).strip()
        for item in _planner_plugin_manifest()["plugins"]
        if isinstance(item, Mapping) and str(item.get("plugin_id", "")).strip()
    }
    output: list[dict[str, str]] = []
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
            {"plugin_id": plugin_id, "status": status, "reason": reason}
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
            result.get("modules"),
            page.get("modules"),
            "plugin_id",
        )
        result["assets"] = _merge_identity_records(
            result.get("assets"),
            page.get("assets"),
            "id",
        )
        if "art_direction" not in result and isinstance(
            page.get("art_direction"), Mapping
        ):
            result["art_direction"] = dict(page["art_direction"])
    _validate_design(result)
    return result


def _merge_identity_records(left: Any, right: Any, key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in [*(left if isinstance(left, list) else []), *(right if isinstance(right, list) else [])]:
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
            {
                "plugin_id": plugin["plugin_id"],
                "status": plugin["status"],
            }
            for plugin in manifest["plugins"]
        ],
    }


def _canonical_game_design(design: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        field: design[field]
        for field in _GAME_DESIGN_FIELDS
        if field in design
    }
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
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
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
