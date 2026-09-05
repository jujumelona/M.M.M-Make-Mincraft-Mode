from __future__ import annotations

"""Host-owned evidence-first Minecraft implementation planning.

This compiler owns implementation architecture.  Authored behavior is represented by a
canonical capability catalog upstream; this layer maps that capability to a researched
Minecraft template, binds verified reuse and a resolved target, materializes a concrete
DAG, and validates the DAG by deterministic recompilation with tracing disabled.

No language model chooses registry/persistence/network/worldgen/UI architecture here.
"""

import hashlib
import heapq
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .canonical_capability_ontology import resolve_capabilities_from_phrase_structured
from .minecraft_template_catalog import (
    FEATURE_CLIENT,
    FEATURE_DATAGEN,
    FEATURE_MIXIN,
    FEATURE_NETWORK,
    FEATURE_PERSISTENCE,
    FEATURE_REGISTRY,
    FEATURE_WORLDGEN,
    RESEARCH_BASIS,
    TEMPLATE_CATALOG_SCHEMA,
    profile_for_capability,
    requirement_branch_features,
)
from .minecraft_template_steps import ROOT_PROVIDE, TemplateStep, steps_for_profile
from .root_cause_trace import emit_root_cause

SCHEMA = "mmm/evidence-first-implementation-plan-v1"
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_COMPONENT_ID_RE = re.compile(
    r"^(?:[a-z][a-z0-9_]{1,63}|component:[a-z0-9_-]+:[0-9a-f]{64})$"
)
_SEMANTIC_BOUNDARY = re.compile(r"[^.!?\n\r]+(?:[.!?]+|$)", re.UNICODE)
_CLAUSE_SEPARATOR = re.compile(
    r"\s*(?:,|;|→|->|=>|/|\||•|\u2022|\u25b6|\u25cf|\u2013|\u2014)\s*",
    re.UNICODE,
)
_BRANCHES = (
    "needs_registry",
    "needs_datagen",
    "needs_persistence",
    "needs_network",
    "needs_client_render",
    "needs_worldgen",
    "needs_mixin",
    "needs_loader_leaf",
)


class EvidencePlanError(ValueError):
    """Raised when host-owned planning evidence is incomplete or inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    encoded = (
        value.encode("utf-8")
        if isinstance(value, str)
        else _canonical(value).encode("utf-8")
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = ""
    return _sha(payload)


def _slug(value: Any, fallback: str = "item") -> str:
    raw = str(value or "")
    text = re.sub(r"[^a-z0-9_]+", "_", raw.casefold()).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = f"{fallback}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10]}"
    if not text[0].isalpha():
        text = f"{fallback}_{text}"
    return text[:36]


def _stable_id(prefix: str, semantic: str, discriminator: Any) -> str:
    digest = _sha({"semantic": semantic, "discriminator": discriminator})[7:17]
    return f"{prefix}_{_slug(semantic)}_{digest}"[:63]


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(
        dict.fromkeys(
            text
            for item in values
            if (text := str(item).strip())
        )
    )


def _canonical_capability(value: Any) -> str:
    text = str(value or "").strip().casefold().removeprefix("capability:")
    return "capability:" + text if text else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _semantic_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Split only at authored sentence/list boundaries, never by token size."""

    spans: list[tuple[int, int]] = []
    line_offset = 0
    for raw_line in re.split(r"\r?\n|\r", prompt):
        line_start = line_offset
        line_end = line_offset + len(raw_line)
        line_offset = line_end + len(prompt[line_end : line_end + 1])
        if not raw_line.strip():
            continue
        matched = False
        for match in _SEMANTIC_BOUNDARY.finditer(raw_line):
            start = line_start + match.start()
            end = line_start + match.end()
            inner = raw_line[match.start() : match.end()]
            bullet = re.match(r"^[\s\-\*•▶●]*(?:\d+\.\s*)?", inner)
            if bullet:
                start += bullet.end()
            while start < end and prompt[start].isspace():
                start += 1
            while end > start and prompt[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
                matched = True
        if not matched:
            start = line_start
            end = line_end
            while start < end and prompt[start].isspace():
                start += 1
            while end > start and prompt[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
    if not spans and prompt.strip():
        spans.append((len(prompt) - len(prompt.lstrip()), len(prompt.rstrip())))
    return tuple(spans)


def _semantic_clause_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for sentence_start, sentence_end in _semantic_spans(prompt):
        cursor = sentence_start
        for separator in _CLAUSE_SEPARATOR.finditer(
            prompt, sentence_start, sentence_end
        ):
            left, right = cursor, separator.start()
            while left < right and prompt[left].isspace():
                left += 1
            while right > left and prompt[right - 1].isspace():
                right -= 1
            if left < right:
                result.append((left, right))
            cursor = separator.end()
        left, right = cursor, sentence_end
        while left < right and prompt[left].isspace():
            left += 1
        while right > left and prompt[right - 1].isspace():
            right -= 1
        if left < right:
            result.append((left, right))
    return tuple(result)


def _matched_source_span(prompt: str, statement: str) -> tuple[int, int] | None:
    folded = prompt.casefold()
    candidates = tuple(
        dict.fromkeys(
            candidate.strip()
            for candidate in (
                statement,
                statement.strip().rstrip(".?!;:"),
                statement.replace("_", " "),
                statement.strip().rstrip(".?!;:").replace("_", " "),
            )
            if candidate.strip()
        )
    )
    for candidate in candidates:
        start = folded.find(candidate.casefold())
        if start < 0:
            continue
        end = start + len(candidate)
        return next(
            (
                (left, right)
                for left, right in _semantic_clause_spans(prompt)
                if left <= start and end <= right
            ),
            (start, end),
        )
    return None


def _source_span(prompt: str, statement: str) -> dict[str, Any]:
    matched = _matched_source_span(prompt, statement)
    if matched is None:
        spans = _semantic_spans(prompt)
        start, end = spans[0] if spans else (0, len(prompt))
    else:
        start, end = matched
    text = prompt[start:end]
    return {
        "source_id": "requested_prompt",
        "char_start": start,
        "char_end": end,
        "text": text,
        "text_sha256": _sha(text),
    }


def _fallback_capability(statement: str) -> str:
    resolution = resolve_capabilities_from_phrase_structured(statement)
    explicit = [
        node.capability_id
        for node in resolution.nodes
        if node.origin == "explicit" and not node.capability_id.startswith("unresolved:")
    ]
    if explicit:
        return str(explicit[0]).casefold()
    return "custom.semantic_" + _sha(statement)[7:23]


def _is_public_acceptance(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    folded = text.casefold()
    internal = (
        "owned anchors",
        "owned_anchor",
        "declared provides",
        "declared_provides",
        "required gates",
        "required_gates",
        "task_sha256",
        "done_predicate",
    )
    return not re.match(r"^task_[a-z0-9_]+\s*:", folded) and not any(
        marker in folded for marker in internal
    )


def _word_overlap(left: str, right: str) -> bool:
    token = re.compile(r"[\w]{2,}", re.UNICODE)
    return bool(
        {item.casefold() for item in token.findall(left)}
        & {item.casefold() for item in token.findall(right)}
    )


def build_request_catalog(
    prompt: str,
    game_design: Mapping[str, Any],
    router: Any | None = None,
) -> dict[str, Any]:
    """Return the frozen authoritative catalog or a deterministic host-only fallback."""

    del router
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvidencePlanError("Evidence-first planning requires a non-empty request.")
    existing = game_design.get("_evidence_request_catalog")
    if isinstance(existing, Mapping):
        catalog = dict(existing)
        _validate_request_catalog(catalog, prompt=prompt)
        return catalog

    acceptance_source = _strings(game_design.get("acceptance_tests"))
    requirements: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(_semantic_clause_spans(prompt)):
        statement = prompt[start:end]
        capability = _fallback_capability(statement)
        requirement_id = _stable_id(
            "req",
            capability,
            {"prompt_sha256": _sha(prompt), "index": index, "span": [start, end]},
        )
        profile = profile_for_capability(capability)
        acceptance = [
            item for item in acceptance_source if _word_overlap(item, statement)
        ] or [f"Verify the observable player-facing behavior for capability {capability}."]
        span = _source_span(prompt, statement)
        requirements.append(
            {
                "requirement_id": requirement_id,
                "capability": capability,
                "statement": statement,
                "semantic_statement": statement,
                "mandatory": True,
                "provenance_role": "explicit",
                "source_span": span,
                "derived_from": [],
                "depends_on": [],
                "provides": [_canonical_capability(capability)],
                "gameplay_capabilities": [capability],
                "implementation_capabilities": list(
                    profile.implementation_capabilities
                ),
                "artifact_task_ids": [
                    _stable_id(
                        "task",
                        implementation,
                        {"requirement_id": requirement_id, "layer": "implementation"},
                    )
                    for implementation in profile.implementation_capabilities
                ],
                "semantic_type": "gameplay_mechanic",
                "unlock_policy": {
                    "required_capabilities": [],
                    "required_requirement_refs": [],
                    "optional_capabilities": [],
                    "optional_requirement_refs": [],
                    "policy": "host_feature_model_and_authored_state_only",
                },
                "artifact_obligations": [
                    {"kind": kind, "status": "REQUIRED_DESIGN_AND_GENERATION"}
                    for kind in profile.artifact_kinds
                ],
                "design_resolution_obligations": list(
                    profile.design_resolution_obligations
                ),
                "runtime_acceptance": [
                    f"Exercise and independently observe the authored runtime behavior for {capability}."
                ],
                "semantic_status": "RESOLVED",
                "unresolved_spans": [],
                "acceptance": list(dict.fromkeys(acceptance)),
                "observable_behavior": {
                    "given": "the authored preconditions are established",
                    "when": statement,
                    "then": "the authored observable outcome occurs",
                },
                "template_profile": {
                    "template_id": profile.template_id,
                    "architecture_owner": "host",
                },
            }
        )
    if not requirements:
        raise EvidencePlanError("The request did not yield any semantic requirement.")
    catalog: dict[str, Any] = {
        "prompt_sha256": _sha(prompt),
        "prompt_char_length": len(prompt),
        "purpose": str(
            game_design.get("pitch") or game_design.get("description") or prompt
        ).strip(),
        "requirements": requirements,
        "constraints": list(_strings(game_design.get("constraints"))),
        "non_goals": list(_strings(game_design.get("non_goals"))),
        "deployment_expectations": list(
            _strings(game_design.get("deployment_expectations"))
        ),
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = _hash_without(catalog, "catalog_sha256")
    return catalog


def _validate_request_catalog(catalog: Mapping[str, Any], *, prompt: str) -> None:
    if catalog.get("catalog_sha256") != _hash_without(catalog, "catalog_sha256"):
        raise EvidencePlanError("Pre-target request catalog hash mismatch.")
    if (
        catalog.get("prompt_sha256") != _sha(prompt)
        or catalog.get("prompt_char_length") != len(prompt)
    ):
        raise EvidencePlanError("Pre-target request catalog is stale for the supplied prompt.")
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise EvidencePlanError("Pre-target request catalog has no requirements.")
    ids: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise EvidencePlanError("Pre-target request requirement must be an object.")
        requirement_id = str(requirement.get("requirement_id") or "")
        if not _ID_RE.fullmatch(requirement_id) or requirement_id in ids:
            raise EvidencePlanError("Pre-target requirement IDs are invalid or duplicated.")
        ids.add(requirement_id)
        span = _mapping(requirement.get("source_span"))
        start, end = span.get("char_start"), span.get("char_end")
        text = str(span.get("text") or "")
        if (
            type(start) is not int
            or type(end) is not int
            or not (0 <= start < end <= len(prompt))
            or prompt[start:end] != text
            or span.get("text_sha256") != _sha(text)
        ):
            raise EvidencePlanError(
                f"Pre-target request source receipt is stale for {requirement_id}."
            )
        if bool(requirement.get("mandatory", True)) and requirement.get(
            "semantic_status", "RESOLVED"
        ) == "UNRESOLVED":
            raise EvidencePlanError("Mandatory request text is unresolved.")
        if not _strings(requirement.get("provides")):
            raise EvidencePlanError("Mandatory request requirement has no capability.")
    for requirement in requirements:
        if any(dep not in ids for dep in _strings(requirement.get("depends_on"))):
            raise EvidencePlanError("Request dependency references an unknown requirement.")


def _normalize_sha(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", text):
        return "sha256:" + text
    return text


def _component_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        nested = value.get("components")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, Mapping)]
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def normalize_component_catalog(
    game_design: Mapping[str, Any],
    component_catalog: Any = None,
) -> tuple[dict[str, Any], ...]:
    raw = component_catalog
    inventory_attested = False
    if raw is None:
        inventory = game_design.get("_existing_project_inventory")
        if isinstance(inventory, Mapping):
            try:
                from .project_inventory import validate_project_inventory_payload
            except (ImportError, AttributeError):
                validate_project_inventory_payload = None
            if callable(validate_project_inventory_payload):
                validated = validate_project_inventory_payload(inventory)
                inventory_payload = (
                    dict(validated)
                    if isinstance(validated, Mapping)
                    else dict(inventory)
                )
                raw = inventory_payload.get("component_catalog")
                inventory_attested = True
        if raw is None:
            raw = game_design.get("_component_catalog")

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(_component_items(raw)):
        locator = str(
            item.get("locator") or item.get("path") or item.get("symbol") or ""
        ).strip()
        provides = _strings(item.get("provides"))
        identifier = str(item.get("component_id") or "").strip()
        if not identifier:
            identifier = _stable_id(
                "component", locator or "receipt", {"index": index, "provides": provides}
            )
        if not _COMPONENT_ID_RE.fullmatch(identifier):
            identifier = _stable_id(
                "component", identifier, {"locator": locator, "provides": provides}
            )
        if identifier in seen:
            raise EvidencePlanError(f"Duplicate component id: {identifier}")
        seen.add(identifier)
        raw_provenance = item.get("provenance")
        provenance = _mapping(raw_provenance)
        origin = str(
            provenance.get("origin")
            or (raw_provenance if isinstance(raw_provenance, str) else "")
            or item.get("origin")
            or "unknown"
        ).strip().casefold()
        content_sha256 = _normalize_sha(
            item.get("content_sha256")
            or item.get("sha256")
            or item.get("content_hash")
        )
        bound_to_project = bool(
            inventory_attested
            and origin in {"same_project", "existing_project", "workspace"}
        )
        evidence_refs = _strings(item.get("evidence_refs"))
        raw_evidence = item.get("evidence")
        if not evidence_refs and isinstance(raw_evidence, list):
            evidence_refs = tuple(
                str(evidence.get("locator") or evidence.get("locator_id") or "").strip()
                for evidence in raw_evidence
                if isinstance(evidence, Mapping)
                and str(
                    evidence.get("locator") or evidence.get("locator_id") or ""
                ).strip()
            )
        if locator and content_sha256 and not evidence_refs:
            evidence_refs = (f"locator:{locator}@{content_sha256}",)
        same_project_complete = bool(locator and _SHA_RE.fullmatch(content_sha256))
        external_complete = bool(
            locator
            and _SHA_RE.fullmatch(content_sha256)
            and (provenance.get("repository") or provenance.get("artifact_coordinates"))
            and (provenance.get("revision") or provenance.get("version"))
            and provenance.get("license")
            and isinstance(item.get("compatibility"), Mapping)
            and provenance.get("dependency_closure_verified") is True
        )
        evidence_complete = (
            same_project_complete
            if origin in {"same_project", "existing_project", "workspace"}
            else external_complete
        )
        verified = bool(
            inventory_attested
            and origin in {"same_project", "existing_project", "workspace"}
            and same_project_complete
            and evidence_refs
        )
        component: dict[str, Any] = {
            "component_id": identifier,
            "kind": str(item.get("kind") or "symbol").strip().casefold(),
            "locator": locator,
            "content_sha256": content_sha256,
            "provides": list(provides),
            "requires": list(_strings(item.get("requires"))),
            "target": _mapping(item.get("target"))
            or {
                "minecraft_versions": list(_strings(item.get("minecraft_versions"))),
                "loaders": list(_strings(item.get("loaders"))),
            },
            "side": str(item.get("side") or "common").strip().casefold(),
            "provenance": provenance or {"origin": origin},
            "license_refs": list(_strings(item.get("license_refs"))),
            "evidence_refs": list(evidence_refs),
            "verification_status": (
                "verified"
                if verified
                else "external_candidate" if external_complete else "unverified"
            ),
            "evidence_complete": evidence_complete,
            "bound_to_project": bound_to_project,
            "receipt_sha256": "",
        }
        component["receipt_sha256"] = _hash_without(component, "receipt_sha256")
        output.append(component)
    return tuple(output)


def _reuse_payload(game_design: Mapping[str, Any]) -> dict[str, Any]:
    direct = game_design.get("_reuse_plan")
    if isinstance(direct, Mapping):
        return dict(direct)
    selection = game_design.get("_platform_selection")
    if isinstance(selection, Mapping) and isinstance(selection.get("reuse_plan"), Mapping):
        return dict(selection["reuse_plan"])
    return {}


def _target_decision(
    game_design: Mapping[str, Any], target_decision: Any = None
) -> dict[str, Any]:
    raw = _mapping(target_decision)
    if not raw:
        raw = _mapping(game_design.get("_platform_selection"))
    target = _mapping(raw.get("target"))
    if not target:
        target = {
            "minecraft_version": "unresolved",
            "loader": "unresolved",
            "source_api_family": "unresolved",
        }
    policy = (
        "preserve"
        if raw.get("preserved_existing_target")
        else "migrate" if raw.get("migration_requested") else "new"
    )
    optimizer = _mapping(raw.get("optimizer"))
    inventory = _mapping(
        game_design.get("_existing_project_inventory")
        or game_design.get("_existing_snapshot")
    )
    inventory_target = _mapping(inventory.get("target"))
    inventory_modules = (
        inventory.get("modules") if isinstance(inventory.get("modules"), list) else []
    )
    topology_modules = [
        item
        for item in inventory_modules
        if isinstance(item, Mapping)
        and not (
            len(inventory_modules) > 1
            and str(item.get("module_id") or "") == ":"
            and not _strings(item.get("source_sets"))
        )
    ]
    project_topology = {
        "module_ids": [
            str(item.get("module_id") or "")
            for item in topology_modules
            if str(item.get("module_id") or "")
        ],
        "loaders": list(_strings(inventory_target.get("loaders"))),
        "source_sets": sorted(
            {
                str(source_set)
                for item in topology_modules
                for source_set in _strings(item.get("source_sets"))
            }
        ),
    }
    supplied_topology = _mapping(raw.get("project_topology"))
    if supplied_topology:
        project_topology = {
            "module_ids": list(_strings(supplied_topology.get("module_ids"))),
            "loaders": list(_strings(supplied_topology.get("loaders"))),
            "source_sets": list(_strings(supplied_topology.get("source_sets"))),
        }
    rejected: list[dict[str, Any]] = []
    candidates = optimizer.get("candidates")
    if isinstance(candidates, list):
        selected_key = (
            str(target.get("minecraft_version") or ""),
            str(target.get("loader") or "").casefold(),
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            candidate_target = _mapping(candidate.get("target"))
            key = (
                str(candidate_target.get("minecraft_version") or ""),
                str(candidate_target.get("loader") or "").casefold(),
            )
            if key != selected_key:
                rejected.append(
                    {
                        "target": candidate_target,
                        "total_expected_cost": candidate.get("total_expected_cost"),
                        "reason": "ranked_below_selected_after_hard_gates_and_verified_reuse",
                    }
                )
    resolved = bool(
        str(target.get("minecraft_version") or "").strip().casefold()
        not in {"", "unresolved"}
        and str(target.get("loader") or "").strip().casefold()
        not in {"", "unresolved"}
    )
    result: dict[str, Any] = {
        "policy": policy,
        "coordinates": target,
        "hard_gate_status": "passed" if resolved else "deferred",
        "preserved_existing_target": bool(raw.get("preserved_existing_target")),
        "migration_requested": bool(raw.get("migration_requested")),
        "decision_reason": str(
            raw.get("reason") or optimizer.get("selection_basis") or "host target input"
        ),
        "rejected_alternatives": rejected,
        "project_topology": project_topology,
        "evidence_refs": [f"platform-selection:{_sha(raw)}"] if raw else [],
        "decision_sha256": "",
    }
    result["decision_sha256"] = _hash_without(result, "decision_sha256")
    return result


def _verified_project_provides(components: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        _canonical_capability(capability)
        for component in components
        if component.get("verification_status") == "verified"
        and component.get("bound_to_project") is True
        for capability in _strings(component.get("provides"))
        if capability.casefold().startswith("capability:")
    }


def _validated_external_reuse(
    raw: Mapping[str, Any], *, capability: str, target: Mapping[str, Any]
) -> bool:
    mode = str(raw.get("mode") or "").strip().casefold()
    source_id = str(raw.get("source_id") or "").strip()
    if mode == "library":
        return source_id.startswith("host-api:")
    donor = raw.get("donor")
    if not isinstance(donor, Mapping):
        return False
    if mode == "mmm_verified":
        registry = donor.get("registry_component")
        if not isinstance(registry, Mapping):
            return False
        try:
            from .component_registry import VerifiedComponent

            parsed = VerifiedComponent.from_dict(registry)
        except (ImportError, TypeError, ValueError):
            return False
        if parsed is None:
            return False
        coordinates = _mapping(target.get("coordinates"))
        return (
            _canonical_capability(capability)
            in {_canonical_capability(item) for item in parsed.capabilities}
            and parsed.minecraft_version
            == str(coordinates.get("minecraft_version") or "")
            and parsed.loader == str(coordinates.get("loader") or "").casefold()
        )
    if mode not in {"source_transplant", "adapt"}:
        return False
    if _canonical_capability(donor.get("capability")) != _canonical_capability(capability):
        return False
    from .source_transplant import SourceTransplantError, validated_reuse_donor

    try:
        validated_reuse_donor(raw)
    except (SourceTransplantError, ValueError):
        return False
    return True


def _reuse_decisions(
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    reuse_plan: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_by_capability = (
        {
            str(item.get("capability") or "").strip(): item
            for item in reuse_plan.get("capabilities", ())
            if isinstance(item, Mapping) and str(item.get("capability") or "").strip()
        }
        if isinstance(reuse_plan.get("capabilities"), list)
        else {}
    )
    result: list[dict[str, Any]] = []
    for requirement in requirements:
        capability = str(requirement["capability"])
        raw = _mapping(raw_by_capability.get(capability))
        requested_component_refs = set(_strings(raw.get("component_refs")))
        canonical = _canonical_capability(capability)
        matches = [
            str(item["component_id"])
            for item in components
            if canonical
            in {
                _canonical_capability(value)
                for value in _strings(item.get("provides"))
                if value.casefold().startswith("capability:")
            }
            and item.get("verification_status") == "verified"
        ]
        project_matches = [
            str(item["component_id"])
            for item in components
            if str(item["component_id"]) in matches
            and item.get("bound_to_project") is True
        ]
        mode = str(raw.get("mode") or "fresh").strip().casefold()
        if project_matches:
            action = "retain"
            evidence_status = "verified"
            refs = project_matches
            source_refs: list[str] = []
        elif _validated_external_reuse(raw, capability=capability, target=target):
            action = "adapt"
            evidence_status = "verified_external"
            refs = [
                str(item["component_id"])
                for item in components
                if str(item["component_id"]) in requested_component_refs
                and item.get("evidence_complete") is True
            ]
            source_refs = [
                f"external-reuse:{_sha(raw)}",
                *list(_strings(raw.get("source_id"))),
            ]
        else:
            action = "fresh"
            evidence_status = "missing" if mode == "same_project" else "not_applicable"
            refs = []
            source_refs = []
        decision: dict[str, Any] = {
            "decision_id": _stable_id(
                "reuse", capability, {"action": action, "refs": refs}
            ),
            "requirement_ref": requirement["requirement_id"],
            "capability": capability,
            "action": action,
            "component_refs": refs,
            "source_refs": source_refs,
            "external_receipt": dict(raw) if action == "adapt" else {},
            "evidence_status": evidence_status,
            "residual_work": (
                "none; verified project-bound component is retained"
                if action == "retain"
                else str(
                    raw.get("rationale")
                    or "implement and independently verify the missing capability"
                )
            ),
            "source_mode": mode,
            "decision_sha256": "",
        }
        decision["decision_sha256"] = _hash_without(decision, "decision_sha256")
        result.append(decision)
    return tuple(result)


def _branch_predicates(
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    refs_by_branch: dict[str, list[str]] = {branch: [] for branch in _BRANCHES}
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        for branch in requirement_branch_features(requirement):
            if branch in refs_by_branch and requirement_id:
                refs_by_branch[branch].append(requirement_id)
    if any(
        str(component.get("kind") or "").casefold() == "generated_resource"
        for component in components
    ):
        refs_by_branch["needs_datagen"].append("component:generated_resource")
    topology = _mapping(target.get("project_topology"))
    module_ids = _strings(topology.get("module_ids"))
    loaders = _strings(topology.get("loaders"))
    if len(module_ids) > 1 or len(loaders) > 1:
        refs_by_branch["needs_loader_leaf"].append("target:multi_loader_topology")

    result: dict[str, dict[str, Any]] = {}
    for branch in _BRANCHES:
        evidence_refs = list(dict.fromkeys(refs_by_branch[branch]))
        active = bool(evidence_refs)
        result[branch] = {
            "predicate": branch,
            "status": "ACTIVE" if active else "NOT_APPLICABLE",
            "evidence_refs": (
                evidence_refs
                if active
                else ["host-template-catalog:no-matching-feature"]
            ),
            "reason": (
                "activated by host Minecraft template feature model"
                if active
                else "no selected template or target topology activates this branch"
            ),
        }
    return result


def _active(branches: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    value = branches.get(name)
    return isinstance(value, Mapping) and value.get("status") == "ACTIVE"


def _required_gates(
    capability: str,
    branches: Mapping[str, Mapping[str, Any]],
    *,
    semantic_type: str = "gameplay_mechanic",
    step: TemplateStep | None = None,
) -> tuple[str, ...]:
    profile = profile_for_capability(capability, semantic_type=semantic_type)
    features = set(step.branch_features if step is not None else profile.branch_features)
    gates = ["source_static_validation", "target_compile"]
    if FEATURE_DATAGEN in features:
        gates.append("generated_resource_validation")
    if FEATURE_NETWORK in features:
        gates.append("network_protocol_validation")
    if FEATURE_WORLDGEN in features:
        gates.append("worldgen_runtime_validation")
    if FEATURE_MIXIN in features or (
        semantic_type == "software_quality" and _active(branches, "needs_mixin")
    ):
        gates.extend(("behavior_equivalence", "performance_regression"))
    return tuple(dict.fromkeys(gates))


def _semantic_steps(
    capability: str,
    branches: Mapping[str, Mapping[str, Any]],
    *,
    semantic_type: str = "gameplay_mechanic",
) -> tuple[TemplateStep, ...]:
    del branches
    return steps_for_profile(
        profile_for_capability(capability, semantic_type=semantic_type)
    )


def _ownership_context(game_design: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _mapping(
        game_design.get("_existing_project_inventory")
        or game_design.get("_existing_snapshot")
    )
    modules = inventory.get("modules") if isinstance(inventory.get("modules"), list) else []
    topology_modules = [
        item
        for item in modules
        if isinstance(item, Mapping)
        and not (
            len(modules) > 1
            and str(item.get("module_id") or "") == ":"
            and not _strings(item.get("source_sets"))
        )
    ]
    roots = [
        dict(root)
        for module in modules
        if isinstance(module, Mapping)
        for root in module.get("source_roots", ())
        if isinstance(root, Mapping)
    ]
    source_candidates = [
        item
        for item in roots
        if item.get("language") in {"java", "kotlin"}
        and not item.get("test")
        and "client" not in str(item.get("source_set") or "").casefold()
    ]
    source = next(
        (
            item
            for item in source_candidates
            if "common" in str(item.get("module_id") or "").casefold()
        ),
        source_candidates[0] if source_candidates else {},
    )
    resource = next(
        (
            item
            for item in roots
            if item.get("language") == "resources" and not item.get("test")
        ),
        {},
    )
    test = next((item for item in roots if item.get("test")), {})
    metadata = inventory.get("metadata") if isinstance(inventory.get("metadata"), list) else []
    mod_id = str(game_design.get("mod_id") or "").strip()
    if not mod_id:
        mod_id = next(
            (
                str(item.get("mod_id") or "").strip()
                for item in metadata
                if isinstance(item, Mapping) and str(item.get("mod_id") or "").strip()
            ),
            "generated_mod",
        )
    namespaces = _strings(inventory.get("namespaces"))
    namespace = namespaces[0] if namespaces else f"generated.{_slug(mod_id)}"
    language = str(source.get("language") or "java")
    extension = "kt" if language == "kotlin" else "java"
    return {
        "module_id": str(source.get("module_id") or resource.get("module_id") or ":"),
        "source_set": str(source.get("source_set") or "main"),
        "source_root": str(source.get("path") or f"src/main/{language}"),
        "resource_root": str(resource.get("path") or "src/main/resources"),
        "test_root": str(test.get("path") or f"src/test/{language}"),
        "namespace": namespace,
        "mod_id": _slug(mod_id),
        "extension": extension,
        "topology_module_ids": [
            str(item.get("module_id") or "")
            for item in topology_modules
            if str(item.get("module_id") or "")
        ],
        "topology_source_sets": sorted(
            {
                str(source_set)
                for item in topology_modules
                for source_set in _strings(item.get("source_sets"))
            }
        ),
    }


def _class_name(value: str) -> str:
    words = [item for item in re.split(r"[^A-Za-z0-9]+", value) if item]
    result = "".join(item[:1].upper() + item[1:] for item in words) or "SemanticTask"
    if not result[0].isalpha():
        result = "Task" + result
    return result[:96]


def _anchors(
    capability: str,
    step: TemplateStep,
    task_id: str,
    ownership: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base = _slug(capability)
    class_name = _class_name(task_id)
    namespace_path = str(ownership["namespace"]).replace(".", "/")
    locators = {
        "symbol": (
            f"{ownership['source_root']}/{namespace_path}/mmmplan/{class_name}.{ownership['extension']}"
            f"#{class_name}"
        ),
        "resource": f"resource:{ownership['mod_id']}:{base}/{step.name}",
        "registry_id": f"registry:{ownership['mod_id']}:{base}/{step.name}",
        "test": (
            f"{ownership['test_root']}/{namespace_path}/mmmplan/{class_name}Test.{ownership['extension']}"
            f"#{class_name}Test"
        ),
        "build_config": f"module:{ownership['module_id']}:build_config",
    }
    if step.name == "loader_leaf_binding":
        module_ids = list(_strings(ownership.get("topology_module_ids")))
        if len(module_ids) < 2:
            raise EvidencePlanError(
                "Loader-leaf task requires validated multi-module ownership anchors."
            )
        return [
            {
                "kind": "loader_module",
                "locator": f"module:{module_id}:loader_leaf",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": module_id,
                "source_set": (
                    "common" if "common" in module_id.casefold() else "loader_leaf"
                ),
            }
            for module_id in module_ids
        ]
    return [
        {
            "kind": kind,
            "locator": locators[kind],
            "ownership": "exclusive",
            "status": "host_reserved",
            "module_id": ownership["module_id"],
            "source_set": (
                "test"
                if kind == "test"
                else "resources" if kind == "resource" else ownership["source_set"]
            ),
        }
        for kind in step.anchor_kinds
    ]


def _requirement_done(requirement_ref: str) -> str:
    return f"requirement_done:{requirement_ref}"


def _requirement_ready(requirement_ref: str) -> str:
    return f"requirement_ready:{requirement_ref}"


def _rewrite_root(
    steps: Sequence[TemplateStep],
    *,
    root: str,
) -> tuple[TemplateStep, ...]:
    return tuple(
        TemplateStep(
            name=step.name,
            outcome=step.outcome,
            consumes=tuple(root if item == ROOT_PROVIDE else item for item in step.consumes),
            provides=step.provides,
            anchor_kinds=step.anchor_kinds,
            branch_features=step.branch_features,
        )
        for step in steps
    )


def _loader_leaf_steps(
    capability: str,
    steps: Sequence[TemplateStep],
) -> tuple[TemplateStep, ...]:
    common = f"common_contract:{capability}"
    rewritten: list[TemplateStep] = []
    replaced = False
    for step in steps:
        if capability in step.provides:
            rewritten.append(
                TemplateStep(
                    name=step.name,
                    outcome=step.outcome,
                    consumes=step.consumes,
                    provides=tuple(
                        common if item == capability else item for item in step.provides
                    ),
                    anchor_kinds=step.anchor_kinds,
                    branch_features=step.branch_features,
                )
            )
            replaced = True
        else:
            rewritten.append(step)
    if not replaced:
        raise EvidencePlanError(
            f"Capability {capability!r} has no common provider for loader leaves."
        )
    rewritten.append(
        TemplateStep(
            name="loader_leaf_binding",
            outcome=(
                f"Bind the common implementation contract for {capability} into every "
                "approved loader module without duplicating domain state ownership"
            ),
            consumes=(common,),
            provides=(capability,),
            anchor_kinds=("build_config", "symbol", "test"),
            branch_features=("needs_loader_leaf",),
        )
    )
    return tuple(rewritten)


def _step_uses_branch(step: TemplateStep, branch: str) -> bool:
    return branch in step.branch_features or (
        branch == "needs_loader_leaf" and step.name == "loader_leaf_binding"
    )


def _bind_consumes_dependencies(
    tasks: Sequence[Mapping[str, Any]],
    *,
    root_provides: set[str],
    emit_trace: bool = True,
) -> tuple[dict[str, Any], ...]:
    providers: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        for provided in _strings(task.get("provides")):
            providers.setdefault(provided, []).append(task_id)
    bound: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        task_id = str(task["task_id"])
        dependencies: list[str] = []
        for consumed in _strings(task.get("consumes")):
            if consumed in root_provides:
                continue
            candidates = providers.get(consumed, [])
            if len(candidates) != 1:
                raise EvidencePlanError(
                    f"Task {task_id} consumes {consumed!r} without exactly one provider."
                )
            provider = candidates[0]
            if provider == task_id:
                raise EvidencePlanError(
                    f"Task {task_id} consumes its own provide {consumed!r}."
                )
            if provider not in dependencies:
                dependencies.append(provider)
        task["depends_on"] = dependencies
        task["task_sha256"] = ""
        task["task_sha256"] = _hash_without(task, "task_sha256")
        bound.append(task)
        if emit_trace:
            emit_root_cause(
                "task_dependency_bound",
                stage="planning",
                operation="bind_task_dependencies",
                gate="task_dependency_graph",
                result="PASS",
                details={
                    "task_id": task_id,
                    "consumes": task.get("consumes"),
                    "depends_on": dependencies,
                },
            )
    return tuple(bound)


def _compile_tasks(
    gaps: Sequence[Mapping[str, Any]],
    reuse: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, Any],
    *,
    root_provides: set[str] | None = None,
    emit_trace: bool = True,
) -> tuple[dict[str, Any], ...]:
    roots = set(root_provides or {ROOT_PROVIDE})
    reuse_by_req = {str(item["requirement_ref"]): item for item in reuse}
    tasks: list[dict[str, Any]] = []
    if emit_trace:
        emit_root_cause(
            "implementation_decomposition_start",
            stage="planning",
            operation="compile_tasks",
            gate="requirement_to_codeplan",
            result="START",
            details={
                "gaps": gaps,
                "reuse_decisions": reuse,
                "target": target,
                "branches": branches,
                "ownership": ownership,
                "root_provides": sorted(roots),
                "template_catalog_schema": TEMPLATE_CATALOG_SCHEMA,
            },
        )

    for gap in gaps:
        requirement_ref = str(gap["requirement_ref"])
        capability = str(gap.get("capability") or gap["missing_provides"][0]).casefold()
        semantic_type = str(gap.get("semantic_type") or "gameplay_mechanic")
        profile = profile_for_capability(capability, semantic_type=semantic_type)
        required_provide = str(gap["missing_provides"][0])
        decision = reuse_by_req.get(requirement_ref, {})
        steps: tuple[TemplateStep, ...] = _semantic_steps(
            capability,
            branches,
            semantic_type=semantic_type,
        )

        dependency_refs = tuple(
            dict.fromkeys(_strings(gap.get("depends_on_requirements")))
        )
        if dependency_refs:
            ready = _requirement_ready(requirement_ref)
            gate = TemplateStep(
                name="prerequisite_gate",
                outcome=(
                    f"Require every host-approved prerequisite requirement before "
                    f"activating implementation of {capability}"
                ),
                consumes=tuple(_requirement_done(dep) for dep in dependency_refs),
                provides=(ready,),
                anchor_kinds=("test",),
                branch_features=(),
            )
            steps = (gate, *_rewrite_root(steps, root=ready))

        if _active(branches, "needs_loader_leaf"):
            steps = _loader_leaf_steps(capability, steps)

        rewritten: list[TemplateStep] = []
        for step in steps:
            provides = tuple(
                required_provide if item == capability else item
                for item in step.provides
            )
            if required_provide in provides:
                provides = tuple(dict.fromkeys((*provides, _requirement_done(requirement_ref))))
            rewritten.append(
                TemplateStep(
                    name=step.name,
                    outcome=step.outcome,
                    consumes=step.consumes,
                    provides=provides,
                    anchor_kinds=step.anchor_kinds,
                    branch_features=step.branch_features,
                )
            )
        steps = tuple(rewritten)

        for index, step in enumerate(steps):
            task_id = _stable_id(
                "task",
                f"{capability}_{step.name}",
                {"gap": gap["gap_id"], "index": index},
            )
            active_predicates = [
                branch
                for branch, value in branches.items()
                if value.get("status") == "ACTIVE" and _step_uses_branch(step, branch)
            ]
            acceptance = [
                f"{task_id}: all declared provides exist and all owned anchors pass their integrity checks"
            ]
            if required_provide in step.provides:
                acceptance.extend(
                    str(item)
                    for item in gap.get("acceptance", ())
                    if _is_public_acceptance(item)
                )
            task: dict[str, Any] = {
                "task_id": task_id,
                "semantic_outcome": step.outcome,
                "gap_refs": [gap["gap_id"]],
                "requirement_refs": [requirement_ref],
                "target_cell": dict(target.get("coordinates") or {}),
                "owned_anchors": _anchors(capability, step, task_id, ownership),
                "reuse_refs": list(
                    dict.fromkeys(
                        [
                            *list(decision.get("component_refs") or ()),
                            *list(decision.get("source_refs") or ()),
                        ]
                    )
                ),
                "consumes": list(step.consumes),
                "provides": list(step.provides),
                "depends_on": [],
                "conditional_predicates": active_predicates,
                "required_gates": list(
                    _required_gates(
                        capability,
                        branches,
                        semantic_type=semantic_type,
                        step=step,
                    )
                ),
                "acceptance": list(dict.fromkeys(acceptance)),
                "done_predicate": {
                    "operator": "all",
                    "checks": [
                        "owned_anchor_hashes_recorded",
                        "declared_provides_observed",
                        "required_gates_passed",
                    ],
                },
                "impact_probes": [
                    "changed_symbols",
                    "changed_resource_ids_and_references",
                    "dependency_and_source_set_edges",
                    "affected_tests_and_acceptance_bindings",
                ],
                "template_id": profile.template_id,
                "template_catalog_schema": TEMPLATE_CATALOG_SCHEMA,
                "template_features": sorted(profile.features),
                "state": "pending",
                "task_sha256": "",
            }
            task["task_sha256"] = _hash_without(task, "task_sha256")
            tasks.append(task)
            if emit_trace:
                emit_root_cause(
                    "implementation_task_compiled",
                    stage="planning",
                    operation="compile_tasks",
                    gate="task_contract",
                    result="PASS",
                    details={
                        "task": task,
                        "step_index": index,
                        "step_count": len(steps),
                        "template_id": profile.template_id,
                    },
                )

    bound = _bind_consumes_dependencies(
        tasks,
        root_provides=roots,
        emit_trace=emit_trace,
    )
    if emit_trace:
        emit_root_cause(
            "implementation_decomposition_result",
            stage="planning",
            operation="compile_tasks",
            gate="requirement_to_codeplan",
            result="PASS",
            details={"tasks": bound, "template_catalog_schema": TEMPLATE_CATALOG_SCHEMA},
        )
    return bound


def _topological(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [str(item.get("task_id") or "") for item in tasks]
    if any(not _ID_RE.fullmatch(item) for item in ids) or len(ids) != len(set(ids)):
        raise EvidencePlanError("Task identifiers are invalid or duplicated.")
    outgoing: dict[str, list[str]] = {item: [] for item in ids}
    indegree = {item: 0 for item in ids}
    for task in tasks:
        task_id = str(task["task_id"])
        for dependency in _strings(task.get("depends_on")):
            if dependency not in outgoing:
                raise EvidencePlanError(
                    f"Task {task_id} references unknown dependency {dependency}."
                )
            if dependency == task_id:
                raise EvidencePlanError(f"Task {task_id} depends on itself.")
            outgoing[dependency].append(task_id)
            indegree[task_id] += 1
    ready = [item for item, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        item = heapq.heappop(ready)
        order.append(item)
        for dependent in sorted(outgoing[item]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(order) != len(ids):
        cyclic = sorted(item for item, degree in indegree.items() if degree > 0)
        raise EvidencePlanError(
            f"Semantic implementation graph contains a cycle: {cyclic[:20]}"
        )
    return order


def _gap_record(
    requirement: Mapping[str, Any],
    verified: set[str],
) -> dict[str, Any] | None:
    missing = [item for item in requirement["provides"] if item not in verified]
    if not missing:
        return None
    gap: dict[str, Any] = {
        "gap_id": _stable_id(
            "gap", str(requirement["capability"]), requirement["requirement_id"]
        ),
        "requirement_ref": requirement["requirement_id"],
        "capability": requirement["capability"],
        "missing_provides": missing,
        "reason": "not supplied by a verified project-bound component receipt",
        "required_gates": ["source_static_validation", "target_compile"],
        "acceptance": list(requirement["acceptance"]),
        "runtime_acceptance": list(requirement.get("runtime_acceptance") or ()),
        "implementation_capabilities": list(
            requirement.get("implementation_capabilities") or ()
        ),
        "artifact_obligations": list(requirement.get("artifact_obligations") or ()),
        "design_resolution_obligations": list(
            requirement.get("design_resolution_obligations") or ()
        ),
        "semantic_type": str(
            requirement.get("semantic_type") or "gameplay_mechanic"
        ),
        "unlock_policy": dict(requirement.get("unlock_policy") or {}),
        "depends_on_requirements": list(_strings(requirement.get("depends_on"))),
        "gap_sha256": "",
    }
    gap["gap_sha256"] = _hash_without(gap, "gap_sha256")
    return gap


def _retained_requirement_roots(
    requirements: Sequence[Mapping[str, Any]], verified: set[str]
) -> set[str]:
    return {
        _requirement_done(str(requirement["requirement_id"]))
        for requirement in requirements
        if set(_strings(requirement.get("provides"))) <= verified
    }


def _component_refs_by_capability(
    components: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for component in components:
        if (
            component.get("verification_status") != "verified"
            or component.get("bound_to_project") is not True
        ):
            continue
        for capability in _strings(component.get("provides")):
            if capability.casefold().startswith("capability:"):
                result.setdefault(_canonical_capability(capability), []).append(
                    str(component["component_id"])
                )
    for decision in decisions:
        if (
            decision.get("action") == "retain"
            and decision.get("evidence_status") == "verified"
        ):
            result.setdefault(
                _canonical_capability(decision["capability"]), []
            ).extend(str(item) for item in decision.get("component_refs", ()))
    return {
        capability: list(dict.fromkeys(refs)) for capability, refs in result.items()
    }


def compile_evidence_first_plan(
    prompt: str,
    game_design: Mapping[str, Any],
    *,
    component_catalog: Any = None,
    reuse_plan: Mapping[str, Any] | None = None,
    target_decision: Mapping[str, Any] | None = None,
    semantic_router: Any | None = None,
) -> dict[str, Any]:
    request_catalog = build_request_catalog(
        prompt,
        game_design,
        router=semantic_router,
    )
    _validate_request_catalog(request_catalog, prompt=prompt)

    pre_retrieval = game_design.get("_pre_retrieval_plan")
    pre_retrieval_sha256 = ""
    if isinstance(pre_retrieval, Mapping):
        from .reuse_planner import validate_pre_retrieval_plan

        validate_pre_retrieval_plan(pre_retrieval, prompt=prompt, design=game_design)
        pre_retrieval_sha256 = str(pre_retrieval.get("plan_sha256") or "")

    components = normalize_component_catalog(game_design, component_catalog)
    reuse_payload = (
        dict(reuse_plan)
        if isinstance(reuse_plan, Mapping)
        else _reuse_payload(game_design)
    )
    reuse_graph = _mapping(reuse_payload.get("capability_graph"))
    graph_plan_sha256 = str(reuse_graph.get("source_plan_sha256") or "")
    if (
        pre_retrieval_sha256
        and reuse_payload
        and graph_plan_sha256 != pre_retrieval_sha256
    ):
        raise EvidencePlanError(
            "Reuse evidence is not bound to the frozen pre-retrieval semantic plan."
        )

    target = _target_decision(game_design, target_decision)
    if target.get("hard_gate_status") != "passed":
        raise EvidencePlanError(
            "Target decision is unresolved; semantic implementation planning is deferred."
        )
    requirements = request_catalog["requirements"]
    decisions = _reuse_decisions(requirements, components, reuse_payload, target)
    verified = _verified_project_provides(components)
    verified.update(
        _canonical_capability(item["capability"])
        for item in decisions
        if item.get("action") == "retain"
        and item.get("evidence_status") == "verified"
    )

    gaps = [
        gap
        for requirement in requirements
        if (gap := _gap_record(requirement, verified)) is not None
    ]
    branches = _branch_predicates(requirements, components, target)
    ownership = _ownership_context(game_design)
    target_topology = _mapping(target.get("project_topology"))
    topology_ids = list(_strings(target_topology.get("module_ids")))
    if topology_ids and not ownership.get("topology_module_ids"):
        ownership["topology_module_ids"] = topology_ids
        ownership["module_id"] = next(
            (item for item in topology_ids if "common" in item.casefold()),
            topology_ids[0],
        )
    if target_topology.get("source_sets") and not ownership.get(
        "topology_source_sets"
    ):
        ownership["topology_source_sets"] = list(
            _strings(target_topology.get("source_sets"))
        )

    root_provides = {
        ROOT_PROVIDE,
        *verified,
        *_retained_requirement_roots(requirements, verified),
    }
    tasks = _compile_tasks(
        gaps,
        decisions,
        target,
        branches,
        ownership,
        root_provides=root_provides,
        emit_trace=True,
    )
    order = _topological(tasks)
    by_id = {str(item["task_id"]): item for item in tasks}
    tasks = tuple(by_id[item] for item in order)

    task_refs_by_req: dict[str, list[str]] = {}
    for task in tasks:
        for requirement_ref in task["requirement_refs"]:
            task_refs_by_req.setdefault(str(requirement_ref), []).append(
                str(task["task_id"])
            )
    component_refs = _component_refs_by_capability(components, decisions)
    bindings = [
        {
            "requirement_ref": requirement["requirement_id"],
            "capability": requirement["capability"],
            "component_refs": component_refs.get(
                _canonical_capability(requirement["capability"]), []
            ),
            "task_refs": task_refs_by_req.get(str(requirement["requirement_id"]), []),
            "acceptance": list(requirement["acceptance"]),
            "status": (
                "retained"
                if set(requirement["provides"]) <= verified
                else "planned_gap"
            ),
        }
        for requirement in requirements
    ]

    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "pre_retrieval_plan_sha256": pre_retrieval_sha256,
        "request_catalog": request_catalog,
        "existing_snapshot": _mapping(
            game_design.get("_existing_project_inventory")
            or game_design.get("_existing_snapshot")
        ),
        "component_catalog": list(components),
        "reuse_decisions": list(decisions),
        "target_decision": target,
        "verified_provides": sorted(verified),
        "gap_catalog": gaps,
        "branch_predicates": branches,
        "ownership_context": ownership,
        "template_catalog": {
            "schema_version": TEMPLATE_CATALOG_SCHEMA,
            "architecture_owner": "host",
            "selection_policy": "canonical_capability_to_exact_or_category_template",
            "small_model_role": "bounded_semantic_classification_and_user_specific_values_only",
            "research_basis": list(RESEARCH_BASIS),
        },
        "root_provides": sorted(root_provides),
        "tasks": list(tasks),
        "acceptance_release_bindings": bindings,
        "run_state": {
            "inventory_revision": _sha(list(components)),
            "graph_revision": _sha(
                [
                    {"id": item["task_id"], "depends_on": item["depends_on"]}
                    for item in tasks
                ]
            ),
            "active_task": "",
            "applied_action_ids": [],
            "completed_task_ids": [],
        },
        "observations": [],
        "checkpoints": [],
        "plan_sha256": "",
    }
    plan["plan_sha256"] = _hash_without(plan, "plan_sha256")
    validate_evidence_first_plan(plan, prompt=prompt)
    return plan


def _validate_components(components: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(components, list):
        raise EvidencePlanError("Component catalog must be a list.")
    ids: set[str] = set()
    output: list[dict[str, Any]] = []
    for raw in components:
        if not isinstance(raw, Mapping):
            raise EvidencePlanError("Component receipt must be an object.")
        component = dict(raw)
        identifier = str(component.get("component_id") or "")
        if not _COMPONENT_ID_RE.fullmatch(identifier) or identifier in ids:
            raise EvidencePlanError("Component identifiers are invalid or duplicated.")
        if component.get("receipt_sha256") != _hash_without(
            component, "receipt_sha256"
        ):
            raise EvidencePlanError(f"Component {identifier} receipt hash mismatch.")
        if (
            component.get("verification_status") == "verified"
            and component.get("evidence_complete") is not True
        ):
            raise EvidencePlanError(
                f"Component {identifier} claims verification without complete evidence."
            )
        ids.add(identifier)
        output.append(component)
    return tuple(output)


def _validate_reuse_decisions(
    decisions: Any,
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(decisions, list):
        raise EvidencePlanError("Reuse decisions must be a list.")
    expected_keys = {
        "decision_id",
        "requirement_ref",
        "capability",
        "action",
        "component_refs",
        "source_refs",
        "external_receipt",
        "evidence_status",
        "residual_work",
        "source_mode",
        "decision_sha256",
    }
    requirement_by_id = {
        str(item["requirement_id"]): item for item in requirements
    }
    component_by_id = {str(item["component_id"]): item for item in components}
    seen_ids: set[str] = set()
    seen_requirements: set[str] = set()
    output: list[dict[str, Any]] = []
    for raw in decisions:
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise EvidencePlanError("Reuse decision fields are invalid.")
        decision = dict(raw)
        decision_id = str(decision.get("decision_id") or "")
        requirement_ref = str(decision.get("requirement_ref") or "")
        if not _ID_RE.fullmatch(decision_id) or decision_id in seen_ids:
            raise EvidencePlanError("Reuse decision IDs are invalid or duplicated.")
        if (
            requirement_ref not in requirement_by_id
            or requirement_ref in seen_requirements
        ):
            raise EvidencePlanError("Reuse decision requirement binding is invalid.")
        capability = str(decision.get("capability") or "")
        if capability != str(requirement_by_id[requirement_ref].get("capability") or ""):
            raise EvidencePlanError("Reuse decision capability changed from its requirement.")
        component_refs = _strings(decision.get("component_refs"))
        source_refs = _strings(decision.get("source_refs"))
        if any(ref not in component_by_id for ref in component_refs):
            raise EvidencePlanError("Reuse decision references an unknown component.")
        action = str(decision.get("action") or "")
        status = str(decision.get("evidence_status") or "")
        external = _mapping(decision.get("external_receipt"))
        if action == "retain":
            if status != "verified" or not component_refs or source_refs or external:
                raise EvidencePlanError("Retain decision lacks exact project evidence.")
            canonical = _canonical_capability(capability)
            for component_ref in component_refs:
                component = component_by_id[component_ref]
                aliases = {
                    _canonical_capability(value)
                    for value in _strings(component.get("provides"))
                    if value.casefold().startswith("capability:")
                }
                if (
                    component.get("verification_status") != "verified"
                    or component.get("bound_to_project") is not True
                    or canonical not in aliases
                ):
                    raise EvidencePlanError(
                        "Retain decision references a component without exact capability evidence."
                    )
        elif action == "adapt":
            if status != "verified_external" or not source_refs or not external:
                raise EvidencePlanError("Adapt decision lacks external evidence.")
            if not _validated_external_reuse(
                external, capability=capability, target=target
            ):
                raise EvidencePlanError("Adapt decision external receipt is invalid.")
        elif action == "fresh":
            if component_refs or source_refs or external:
                raise EvidencePlanError("Fresh decision may not claim reuse evidence.")
            if status not in {"missing", "not_applicable"}:
                raise EvidencePlanError("Fresh decision evidence status is invalid.")
        else:
            raise EvidencePlanError(f"Unsupported reuse action: {action!r}")
        expected_id = _stable_id(
            "reuse", capability, {"action": action, "refs": list(component_refs)}
        )
        if decision_id != expected_id:
            raise EvidencePlanError("Reuse decision ID is not host-derived.")
        if decision.get("decision_sha256") != _hash_without(
            decision, "decision_sha256"
        ):
            raise EvidencePlanError("Reuse decision hash mismatch.")
        seen_ids.add(decision_id)
        seen_requirements.add(requirement_ref)
        output.append(decision)
    if seen_requirements != set(requirement_by_id):
        raise EvidencePlanError("Every requirement requires exactly one reuse decision.")
    return tuple(output)


def validate_evidence_first_plan(
    plan: Mapping[str, Any], *, prompt: str | None = None
) -> None:
    if plan.get("schema_version") != SCHEMA:
        raise EvidencePlanError("Unsupported evidence-first planning schema.")
    if plan.get("plan_sha256") != _hash_without(plan, "plan_sha256"):
        raise EvidencePlanError("Evidence-first plan hash mismatch.")

    request = _mapping(plan.get("request_catalog"))
    if request.get("catalog_sha256") != _hash_without(request, "catalog_sha256"):
        raise EvidencePlanError("Request catalog hash mismatch.")
    if prompt is not None:
        _validate_request_catalog(request, prompt=prompt)
    requirements = request.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise EvidencePlanError("Request catalog has no requirements.")
    requirement_ids = {
        str(item.get("requirement_id") or "")
        for item in requirements
        if isinstance(item, Mapping)
    }
    if len(requirement_ids) != len(requirements) or any(
        not _ID_RE.fullmatch(item) for item in requirement_ids
    ):
        raise EvidencePlanError("Request requirement identifiers are invalid or duplicated.")

    components = _validate_components(plan.get("component_catalog"))
    target = _mapping(plan.get("target_decision"))
    if target.get("decision_sha256") != _hash_without(target, "decision_sha256"):
        raise EvidencePlanError("Target decision hash mismatch.")
    coordinates = _mapping(target.get("coordinates"))
    if (
        target.get("hard_gate_status") != "passed"
        or str(coordinates.get("minecraft_version") or "").casefold()
        in {"", "unresolved"}
        or str(coordinates.get("loader") or "").casefold() in {"", "unresolved"}
    ):
        raise EvidencePlanError("Semantic task graph requires a resolved target hard gate.")

    decisions = _validate_reuse_decisions(
        plan.get("reuse_decisions"), requirements, components, target
    )
    verified = _verified_project_provides(components)
    verified.update(
        _canonical_capability(item["capability"])
        for item in decisions
        if item.get("action") == "retain"
        and item.get("evidence_status") == "verified"
    )
    if plan.get("verified_provides") != sorted(verified):
        raise EvidencePlanError("Verified provides do not match attested components.")

    expected_gaps = [
        gap
        for requirement in requirements
        if (gap := _gap_record(requirement, verified)) is not None
    ]
    if _canonical(plan.get("gap_catalog")) != _canonical(expected_gaps):
        raise EvidencePlanError(
            "Gap catalog is not the exact requirements-minus-verified-provides set difference."
        )

    expected_branches = _branch_predicates(requirements, components, target)
    if _canonical(plan.get("branch_predicates")) != _canonical(expected_branches):
        raise EvidencePlanError("Conditional branch predicates do not match templates.")
    branches = plan.get("branch_predicates")
    if not isinstance(branches, Mapping) or set(branches) != set(_BRANCHES):
        raise EvidencePlanError("Minecraft conditional branch catalog is incomplete.")

    ownership = _mapping(plan.get("ownership_context"))
    topology_ids = list(
        _strings(_mapping(target.get("project_topology")).get("module_ids"))
    )
    if topology_ids:
        if list(_strings(ownership.get("topology_module_ids"))) != topology_ids:
            raise EvidencePlanError("Task ownership topology is not target-bound.")
        if str(ownership.get("module_id") or "") not in topology_ids:
            raise EvidencePlanError("Default task owner is outside approved topology.")

    expected_roots = {
        ROOT_PROVIDE,
        *verified,
        *_retained_requirement_roots(requirements, verified),
    }
    roots = set(_strings(plan.get("root_provides")))
    if roots != expected_roots:
        raise EvidencePlanError("Root provides are not host-derived.")

    template_catalog = _mapping(plan.get("template_catalog"))
    expected_template_catalog = {
        "schema_version": TEMPLATE_CATALOG_SCHEMA,
        "architecture_owner": "host",
        "selection_policy": "canonical_capability_to_exact_or_category_template",
        "small_model_role": "bounded_semantic_classification_and_user_specific_values_only",
        "research_basis": list(RESEARCH_BASIS),
    }
    if _canonical(template_catalog) != _canonical(expected_template_catalog):
        raise EvidencePlanError("Minecraft template catalog receipt is invalid.")

    expected_tasks = _compile_tasks(
        expected_gaps,
        decisions,
        target,
        expected_branches,
        ownership,
        root_provides=expected_roots,
        emit_trace=False,
    )
    order = _topological(expected_tasks)
    expected_by_id = {str(item["task_id"]): item for item in expected_tasks}
    expected_tasks = tuple(expected_by_id[item] for item in order)
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise EvidencePlanError("Semantic task graph must be a list.")
    if _canonical(tasks) != _canonical(expected_tasks):
        raise EvidencePlanError("Semantic tasks are not the deterministic template DAG.")
    _topological(tasks)

    providers: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        for provided in _strings(task.get("provides")):
            providers.setdefault(provided, []).append(task_id)
    gap_ids = {str(item["gap_id"]) for item in expected_gaps}
    covered_gaps: set[str] = set()
    task_ids = {str(item["task_id"]) for item in tasks}
    for task in tasks:
        task_id = str(task["task_id"])
        if task.get("task_sha256") != _hash_without(task, "task_sha256"):
            raise EvidencePlanError(f"Task {task_id} hash mismatch.")
        gap_refs = set(_strings(task.get("gap_refs")))
        requirement_refs = set(_strings(task.get("requirement_refs")))
        if not gap_refs or not gap_refs <= gap_ids:
            raise EvidencePlanError(f"Task {task_id} has an invalid gap binding.")
        if not requirement_refs or not requirement_refs <= requirement_ids:
            raise EvidencePlanError(
                f"Task {task_id} has an invalid requirement binding."
            )
        covered_gaps.update(gap_refs)
        expected_dependencies: set[str] = set()
        for consumed in _strings(task.get("consumes")):
            if consumed in roots:
                continue
            candidates = providers.get(consumed, [])
            if len(candidates) != 1:
                raise EvidencePlanError(
                    f"Task {task_id} consumes {consumed!r} without exactly one provider."
                )
            expected_dependencies.add(candidates[0])
        if set(_strings(task.get("depends_on"))) != expected_dependencies:
            raise EvidencePlanError(
                f"Task {task_id} dependency edges do not exactly bind consumes."
            )
        if not _strings(task.get("provides")):
            raise EvidencePlanError(f"Task {task_id} must declare provides.")
        if not isinstance(task.get("owned_anchors"), list) or not task["owned_anchors"]:
            raise EvidencePlanError(f"Task {task_id} must own at least one anchor.")
        if not _strings(task.get("acceptance")):
            raise EvidencePlanError(f"Task {task_id} must declare acceptance checks.")
        profile = profile_for_capability(
            str(next(iter(requirement_refs and [next(
                requirement.get("capability")
                for requirement in requirements
                if str(requirement.get("requirement_id") or "") in requirement_refs
            )], [""]))),
            semantic_type=str(
                next(
                    (
                        requirement.get("semantic_type")
                        for requirement in requirements
                        if str(requirement.get("requirement_id") or "") in requirement_refs
                    ),
                    "gameplay_mechanic",
                )
            ),
        )
        if task.get("template_id") != profile.template_id:
            raise EvidencePlanError(f"Task {task_id} template identity changed.")
    if covered_gaps != gap_ids:
        raise EvidencePlanError(f"Unbound implementation gaps: {sorted(gap_ids - covered_gaps)}")

    component_refs = _component_refs_by_capability(components, decisions)
    task_refs_by_requirement: dict[str, list[str]] = {}
    for task in expected_tasks:
        for requirement_ref in task["requirement_refs"]:
            task_refs_by_requirement.setdefault(str(requirement_ref), []).append(
                str(task["task_id"])
            )
    expected_bindings = [
        {
            "requirement_ref": requirement["requirement_id"],
            "capability": requirement["capability"],
            "component_refs": component_refs.get(
                _canonical_capability(requirement["capability"]), []
            ),
            "task_refs": task_refs_by_requirement.get(
                str(requirement["requirement_id"]), []
            ),
            "acceptance": list(requirement["acceptance"]),
            "status": (
                "retained"
                if set(requirement["provides"]) <= verified
                else "planned_gap"
            ),
        }
        for requirement in requirements
    ]
    if _canonical(plan.get("acceptance_release_bindings")) != _canonical(
        expected_bindings
    ):
        raise EvidencePlanError("Acceptance release bindings are not host-derived.")
    for binding in expected_bindings:
        if not binding["component_refs"] and not binding["task_refs"]:
            raise EvidencePlanError(
                f"Requirement {binding['requirement_ref']} has no implementation binding."
            )
        if any(task_ref not in task_ids for task_ref in binding["task_refs"]):
            raise EvidencePlanError("Acceptance binding references an unknown task.")


def task_batches(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    validate_evidence_first_plan(plan)
    return tuple(
        {
            "batch_id": task["task_id"],
            "scope": task["semantic_outcome"],
            "depends_on_batches": list(task["depends_on"]),
            "deliverables": list(task["provides"]),
            "exports": [task["task_id"]],
            "task_contract": dict(task),
        }
        for task in plan["tasks"]
    )


__all__ = [
    "SCHEMA",
    "EvidencePlanError",
    "build_request_catalog",
    "compile_evidence_first_plan",
    "normalize_component_catalog",
    "task_batches",
    "validate_evidence_first_plan",
]
