from __future__ import annotations

"""Host-owned evidence-first implementation planning.

The model may describe implementation details later, but it never owns the identities,
coverage claims, dependency edges, or completion rules produced here.  A target-neutral
semantic work plan is frozen before platform/reuse discovery; this compiler then refines
that same work with a resolved target and verified reuse evidence.  Only verified
project-bound provides are removed from the implementation gap.
"""

import hashlib
import heapq
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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

from .canonical_capability_ontology import (
    canonical_domain_map as _canonical_domain_map,
)
from .canonical_capability_ontology import (
    resolve_capabilities_from_phrase_structured,
)

_DOMAIN_TERM_MAP = _canonical_domain_map()


class EvidencePlanError(ValueError):
    """Raised when host-owned planning evidence is incomplete or inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: Any) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else _canonical(value).encode("utf-8")
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
        # Non-Latin input: use a stable hash prefix instead of script-specific romanization.
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        text = f"{fallback}_{digest}"
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
    text = str(value or "").strip().casefold()
    text = text.removeprefix("capability:")
    return "capability:" + text if text else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _semantic_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Split only at authored semantic boundaries, never by size or token budget.

    Newlines are first-class structural boundaries (list items, paragraph breaks).
    Within each line, sentence-ending punctuation (.!?) further splits clauses.
    """
    spans: list[tuple[int, int]] = []
    # Split on newlines first, then apply sentence-boundary regex within each line.
    line_offset = 0
    for raw_line in re.split(r"\r?\n|\r", prompt):
        line_start = line_offset
        line_end = line_offset + len(raw_line)
        line_offset = line_end + len(prompt[line_end:line_end + 1])  # skip the \n char
        # Strip leading bullet markers (•, -, *, digits+dot) within the line.
        stripped = re.sub(r"^[\s\-\*•▶●]+|^\s*\d+\.\s*", "", raw_line)
        if not stripped.strip():
            continue
        # Apply sentence-boundary regex within the stripped content.
        matched_any = False
        for match in _SEMANTIC_BOUNDARY.finditer(raw_line):
            start = line_start + match.start()
            end = line_start + match.end()
            # Strip leading bullet markers from each match.
            inner = raw_line[match.start():match.end()]
            inner_stripped = re.sub(r"^[\s\-\*•▶●]+|^\s*\d+\.\s*", "", inner)
            if not inner_stripped.strip():
                continue
            offset_into_match = len(inner) - len(inner.lstrip()) + (len(inner.lstrip()) - len(inner_stripped.lstrip()))
            start += offset_into_match
            while start < end and prompt[start].isspace():
                start += 1
            while end > start and prompt[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
                matched_any = True
        if not matched_any and stripped.strip():
            # No sentence boundary found — treat the whole line as one span.
            s = line_start + len(raw_line) - len(raw_line.lstrip())
            # Strip bullet prefix.
            raw_stripped = re.sub(r"^[\s\-\*•▶●]+|^\s*\d+\.\s*", "", raw_line.lstrip())
            s = line_start + (len(raw_line) - len(raw_line.lstrip())) + (len(raw_line.lstrip()) - len(raw_stripped))
            e = line_start + len(raw_line.rstrip())
            while s < e and prompt[s].isspace():
                s += 1
            if s < e:
                spans.append((s, e))
    if not spans and prompt.strip():
        spans.append((len(prompt) - len(prompt.lstrip()), len(prompt.rstrip())))
    return tuple(spans)


def _semantic_clause_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Respect authored sentence/list boundaries without any size-based slicing."""
    result: list[tuple[int, int]] = []
    for sentence_start, sentence_end in _semantic_spans(prompt):
        cursor = sentence_start
        for separator in _CLAUSE_SEPARATOR.finditer(prompt, sentence_start, sentence_end):
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


def _capability_from_statement(statement: str) -> str:
    words = re.findall(r"[\w]+", statement, re.UNICODE)
    ignored = {
        "a", "an", "the", "add", "create", "make", "build", "implement", "keep",
        "minecraft", "mod", "with", "to", "for", "that", "and", "then",
    }
    semantic = [item for item in words if item.casefold() not in ignored]
    # Requirement identity is a lossless semantic slug, not the first ontology hit.
    # Ontology expansion belongs downstream and may never replace the authored scope.
    value = "_".join(semantic) or statement
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"semantic_{digest}"
    return normalized


def _matched_source_span(prompt: str, statement: str) -> tuple[int, int] | None:
    folded_prompt = prompt.casefold()
    normalized_statement = statement.strip().rstrip(".?!;:")
    candidates = tuple(
        dict.fromkeys(
            candidate
            for candidate in (
                statement,
                normalized_statement,
                statement.replace(".", " ").replace("_", " "),
                normalized_statement.replace(".", " ").replace("_", " "),
            )
            if candidate
        )
    )
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        start = folded_prompt.find(candidate.casefold())
        if start >= 0:
            match_end = start + len(candidate)
            return next(
                (
                    (left, right)
                    for left, right in _semantic_clause_spans(prompt)
                    if left <= start and match_end <= right
                ),
                (start, match_end),
            )
    return None


def _source_span(prompt: str, statement: str) -> dict[str, Any]:
    matched_span = _matched_source_span(prompt, statement)
    if matched_span is None:
        spans = _semantic_spans(prompt)
        start, end = spans[0] if spans else (0, len(prompt))
    else:
        start, end = matched_span
    matched = prompt[start:end]
    return {
        "source_id": "requested_prompt",
        "char_start": start,
        "char_end": end,
        "text": matched,
        "text_sha256": _sha(matched),
    }


@dataclass(frozen=True)
class SemanticRequirementIR:
    """Formal contract between the Semantic Model layer and the canonical ontology.

    The Semantic Model (currently a stub backed by the alias resolver; later a
    real LLM) produces this IR for every structural clause in the user prompt.
    The host validates the source offsets and sha256, then passes the IR to the
    ontology for canonicalization and dependency expansion only.

    The ontology MUST NOT receive raw user text — only model-produced
    ``gameplay_capability_candidates`` IDs may be passed to the alias/dependency
    expansion step.
    """

    source_start: int                              # char offset in original prompt
    source_end: int                                # char offset in original prompt
    source_sha256: str                             # sha256(prompt[source_start:source_end])
    intent: str                                    # language-neutral intent (stub: raw clause)
    gameplay_capability_candidates: tuple[str, ...]  # canonical IDs proposed by model
    confidence: float                              # 0.0–1.0
    unresolved: bool                               # model could not determine gameplay cap


def _stub_semantic_model(
    clause: str,
    source_start: int,
    source_end: int,
    prompt: str,
) -> SemanticRequirementIR:
    """Stub Semantic Model — uses the ontology alias resolver as a deterministic fallback.

    Replace with ``_model_semantic_ir`` (real LLM path) by passing a router to
    ``build_request_catalog``.  The stub never sets ``unresolved=True`` because it
    has no authority to block generation — only a real Semantic Model that has
    genuinely attempted to interpret the clause may make that determination.
    """
    resolution = resolve_capabilities_from_phrase_structured(clause)
    candidates = tuple(
        dict.fromkeys(
            node.capability_id
            for node in resolution.nodes
            if node.origin == "explicit"
            and not node.capability_id.startswith("unresolved:")
        )
    )
    candidates = candidates[:1]
    return SemanticRequirementIR(
        source_start=source_start,
        source_end=source_end,
        source_sha256=_sha(prompt[source_start:source_end]),
        intent=clause,
        gameplay_capability_candidates=candidates,
        confidence=0.85 if candidates else 0.0,
        # Stub is best-effort — it never blocks generation.
        # Only a real LLM model may set unresolved=True.
        unresolved=False,
    )


def _model_semantic_ir(
    clause: str,
    source_start: int,
    source_end: int,
    prompt: str,
    router: Any,
) -> SemanticRequirementIR:
    """Real Semantic Model path — calls the LLM via router.generate_text.

    The model receives the raw clause and must return:
      - ``gameplay_capability_candidates``: list of canonical dotted IDs
        (e.g. "economy.trade", "spaceship.vehicle") representing the user intent
      - ``unresolved``: true if the model genuinely cannot determine a gameplay
        capability for this clause (blocks generation)
      - ``intent``: language-neutral description of the clause intent

    The ontology alias map is NOT used here — the model is the semantic authority.
    The ontology is only called later in ``_expand_semantic_ir`` for dependency
    expansion on the candidates returned by the model.
    """
    import json as _json

    system_msg = (
        "You are a Minecraft mod semantic planner.  Given a user requirement clause, "
        "identify the canonical gameplay capability IDs it describes.\n\n"
        "Respond with JSON only:\n"
        "{\n"
        '  "intent": "<language-neutral one-line description>",\n'
        '  "gameplay_capability_candidates": ["<dotted.id>", ...],\n'
        '  "unresolved": false\n'
        "}\n\n"
        "Use dotted IDs like economy.trade, spaceship.vehicle, resource.mining, "
        "combat.boss, worldgen.ore, ui.shop, network.action_sync, etc.  "
        "Set unresolved=true ONLY if you genuinely cannot determine any gameplay "
        "capability from the clause."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": clause},
    ]
    try:
        raw = router.generate_text(
            "planner",
            messages,
            response_format="json",
            enable_tools=False,
        )
        data = _json.loads(raw)
        candidates = tuple(
            str(c).strip()
            for c in data.get("gameplay_capability_candidates", [])
            if str(c).strip()
        )
        intent = str(data.get("intent", clause)).strip() or clause
        unresolved = bool(data.get("unresolved", False)) and not candidates
    except Exception:  # noqa: BLE001 - this is the documented deterministic fallback boundary
        # On any model/parse failure fall back to stub — never crash catalog build.
        return _stub_semantic_model(clause, source_start, source_end, prompt)

    return SemanticRequirementIR(
        source_start=source_start,
        source_end=source_end,
        source_sha256=_sha(prompt[source_start:source_end]),
        intent=intent,
        gameplay_capability_candidates=candidates,
        confidence=0.95 if candidates else 0.0,
        unresolved=unresolved,
    )


def _invoke_semantic_model(
    clause: str,
    source_start: int,
    source_end: int,
    prompt: str,
    router: Any | None,
) -> SemanticRequirementIR:
    """Dispatcher: real model when router is available, stub otherwise."""
    if router is not None:
        return _model_semantic_ir(clause, source_start, source_end, prompt, router)
    return _stub_semantic_model(clause, source_start, source_end, prompt)



def _validate_semantic_ir(ir: SemanticRequirementIR, prompt: str) -> None:
    """Host-side tamper and offset validation for a SemanticRequirementIR.

    Raises EvidencePlanError if:
    - Source offsets are out of bounds.
    - Source sha256 does not match the prompt slice.
    - IR is marked unresolved (mandatory requirement cannot be generated).
    """
    if not (0 <= ir.source_start <= ir.source_end <= len(prompt)):
        raise EvidencePlanError(
            f"SemanticRequirementIR offsets [{ir.source_start}:{ir.source_end}] "
            f"out of bounds for prompt length {len(prompt)}."
        )
    actual_sha = _sha(prompt[ir.source_start:ir.source_end])
    if ir.source_sha256 != actual_sha:
        raise EvidencePlanError(
            "SemanticRequirementIR source_sha256 mismatch — source span was tampered."
        )
    if ir.unresolved:
        raise EvidencePlanError(
            "Semantic Model could not resolve a mandatory requirement span to a "
            "gameplay capability; generation is blocked until the model resolves it."
        )


def _expand_semantic_ir(
    ir: SemanticRequirementIR,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Ontology-only step: canonicalize + dependency-expand model-produced candidates.

    The ontology receives ONLY ``ir.gameplay_capability_candidates`` — IDs already
    proposed by the Semantic Model.  It never sees raw user text.

    Returns (gameplay_roots, implementation_deps, unresolved_flag).
    """
    gameplay: list[str] = []
    implementation: list[str] = []
    seen: set[str] = set()

    for candidate in ir.gameplay_capability_candidates:
        low = candidate.casefold().strip()
        # Alias canonicalization: look up in the known domain map.
        canonical_caps = _DOMAIN_TERM_MAP.get(low)
        if canonical_caps:
            for idx, cap in enumerate(canonical_caps):
                if cap not in seen:
                    seen.add(cap)
                    if idx == 0:
                        gameplay.append(cap)
                    else:
                        implementation.append(cap)
        else:
            # Model is authority for unknown IDs — pass through verbatim.
            if candidate not in seen:
                seen.add(candidate)
                gameplay.append(candidate)

    return tuple(gameplay), tuple(implementation), ir.unresolved


def _semantic_ir_variants(
    ir: SemanticRequirementIR,
) -> tuple[SemanticRequirementIR, ...]:
    """Split independent model roots into independently plannable requirements."""
    if len(ir.gameplay_capability_candidates) <= 1:
        return (ir,)
    variants: list[SemanticRequirementIR] = []
    seen_roots: set[str] = set()
    for candidate in ir.gameplay_capability_candidates:
        variant = SemanticRequirementIR(
            source_start=ir.source_start,
            source_end=ir.source_end,
            source_sha256=ir.source_sha256,
            intent=ir.intent,
            gameplay_capability_candidates=(candidate,),
            confidence=ir.confidence,
            unresolved=ir.unresolved,
        )
        gameplay, _implementation, _unresolved = _expand_semantic_ir(variant)
        root = (gameplay[0] if gameplay else candidate).casefold().strip()
        if not root or root in seen_roots:
            continue
        seen_roots.add(root)
        variants.append(variant)
    return tuple(variants) or (ir,)


def _semantic_requirement_fields(
    capability: str,
    ir: SemanticRequirementIR,
    requirement_id: str,
    *,
    is_design_module: bool | None = None,
) -> dict[str, Any]:
    """Build deterministic traceability fields from a model-produced SemanticRequirementIR.

    Design-module capabilities are authoritative domain IDs (e.g. "trade", "quests")
    that must never be replaced by ontology gameplay roots — the design module is the
    single-authority source.  Only pure prompt-derived requirements (where no design
    module covered the clause) get gameplay root promotion via the IR candidates.
    """
    gameplay, implementation, unresolved_flag = _expand_semantic_ir(ir)
    if is_design_module is True:
        # Design module IDs are authority — preserve verbatim, no promotion.
        selected: tuple[str, ...] = (capability,)
    elif is_design_module is False:
        # Prompt provenance is explicit: semantic gameplay roots may replace the
        # raw fallback identity only on prompt-derived requirements.
        selected = gameplay if gameplay else (capability,)
    else:
        # No provenance proof: preserve the explicit capability instead of guessing.
        selected = (capability,)
    primary = selected[0]
    artifact_tasks = tuple(
        _stable_id(
            "task",
            implementation_capability,
            {"requirement_id": requirement_id, "layer": "artifact"},
        )
        for implementation_capability in (*selected, *implementation)
    )
    return {
        "capability": primary,
        "provides": [_canonical_capability(item) for item in selected],
        "gameplay_capabilities": list(selected),
        "implementation_capabilities": list(implementation),
        "artifact_task_ids": list(dict.fromkeys(artifact_tasks)),
        "semantic_status": "UNRESOLVED" if unresolved_flag and not gameplay else "RESOLVED",
        "unresolved_spans": [ir.intent] if unresolved_flag else [],
    }


def _reuse_payload(game_design: Mapping[str, Any]) -> dict[str, Any]:
    direct = game_design.get("_reuse_plan")
    if isinstance(direct, Mapping):
        return dict(direct)
    selection = game_design.get("_platform_selection")
    if isinstance(selection, Mapping):
        value = selection.get("reuse_plan")
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _capability_records(game_design: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return target-neutral capability/statement pairs in deterministic order."""
    reuse = _reuse_payload(game_design)
    records: list[tuple[str, str]] = []
    graph = reuse.get("capability_graph")
    if isinstance(graph, Mapping):
        for capability in _strings(graph.get("nodes")):
            records.append((capability, capability))
    if not records:
        for item in reuse.get("capabilities", ()) if isinstance(reuse.get("capabilities"), list) else ():
            if isinstance(item, Mapping) and str(item.get("capability") or "").strip():
                capability = str(item["capability"]).strip()
                records.append((capability, capability))
    if not records:
        raw_modules = game_design.get("modules")
        if isinstance(raw_modules, list):
            for item in raw_modules:
                if not isinstance(item, Mapping):
                    continue
                capability = str(
                    item.get("capability")
                    or item.get("plugin_id")
                    or item.get("module_id")
                    or item.get("id")
                    or ""
                ).strip()
                if capability:
                    statement = str(
                        item.get("reason")
                        or item.get("description")
                        or item.get("summary")
                        or capability
                    ).strip()
                    records.append((capability, statement))
    if not records:
        for key in ("features", "systems", "requirements", "core_loop", "progression"):
            raw = game_design.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if isinstance(item, Mapping):
                    statement = str(
                        item.get("requirement")
                        or item.get("capability")
                        or item.get("description")
                        or item.get("name")
                        or ""
                    ).strip()
                else:
                    statement = str(item).strip()
                if statement:
                    records.append((_slug(statement), statement))
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for capability, statement in records:
        normalized = capability.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append((capability.strip(), statement.strip() or capability.strip()))
    return tuple(output)


def _catalog_requirement_covers_clause(
    requirement: Mapping[str, Any],
    prompt: str,
    clause: tuple[int, int],
) -> bool:
    """Recognize only an authored requirement that really binds this prompt span."""

    statement = str(requirement.get("statement") or "").strip()
    if not statement:
        return False
    return _matched_source_span(prompt, statement) == clause


_INTERNAL_ACCEPTANCE_MARKERS = (
    "owned anchors",
    "owned_anchor",
    "declared provides",
    "declared_provides",
    "required gates",
    "required_gates",
    "task_sha256",
    "done_predicate",
)


def _is_public_acceptance(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    folded = text.casefold()
    if re.match(r"^task_[a-z0-9_]+\s*:", folded):
        return False
    return not any(marker in folded for marker in _INTERNAL_ACCEPTANCE_MARKERS)


def _requirement_acceptance(
    capability: str,
    candidates: Iterable[Any],
) -> tuple[str, ...]:
    claimed = tuple(
        dict.fromkeys(
            text
            for item in candidates
            if (text := str(item or "").strip()) and _is_public_acceptance(text)
        )
    )
    if claimed:
        return claimed
    return (
        f"Verify the observable player-facing behavior for capability {capability}.",
    )


def _merge_catalog_uncovered_prompt_requirements(
    catalog: Mapping[str, Any],
    prompt: str,
    game_design: Mapping[str, Any],
    router: Any | None = None,
) -> dict[str, Any]:
    """Preserve every authored clause when an earlier catalog was incomplete.

    A stored EvidenceRequestCatalog is useful provenance, but it cannot make a
    later prompt clause disappear merely because another clause already has a
    requirement record.  We retain all existing records and append only exact,
    previously-uncovered clauses, then rebind the catalog hash.
    """

    raw_requirements = catalog.get("requirements")
    if not isinstance(raw_requirements, list):
        return dict(catalog)
    requirements = [dict(item) for item in raw_requirements if isinstance(item, Mapping)]
    covered = {
        clause
        for clause in _semantic_clause_spans(prompt)
        if any(
            _catalog_requirement_covers_clause(requirement, prompt, clause)
            for requirement in requirements
        )
    }
    additions: list[tuple[str, str]] = []
    for start, end in _semantic_clause_spans(prompt):
        if (start, end) in covered:
            continue
        statement = prompt[start:end]
        additions.append((_capability_from_statement(statement), statement))
    if not additions:
        return dict(catalog)

    acceptance_source = _strings(game_design.get("acceptance_tests"))
    for index, (capability, statement) in enumerate(
        additions, start=len(requirements)
    ):
        span = _source_span(prompt, statement)
        matching_acceptance = tuple(
            item
            for item in acceptance_source
            if _word_overlap(item, f"{capability} {statement}")
        )
        ir = _invoke_semantic_model(
            statement, span["char_start"], span["char_end"], prompt, router
        )
        _validate_semantic_ir(ir, prompt)
        variants = _semantic_ir_variants(ir)
        for variant_index, variant_ir in enumerate(variants):
            gameplay, _implementation, _unresolved = _expand_semantic_ir(variant_ir)
            semantic_id = gameplay[0] if gameplay else capability
            discriminator = {
                "prompt_sha256": _sha(prompt),
                "index": index,
                "span": [span["char_start"], span["char_end"]],
            }
            if len(variants) > 1:
                discriminator["semantic_variant"] = variant_index
            requirement_id = _stable_id("req", semantic_id, discriminator)
            semantic = _semantic_requirement_fields(
                capability, variant_ir, requirement_id, is_design_module=False
            )
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "capability": semantic["capability"],
                    "statement": statement,
                    "mandatory": True,
                    "source_span": span,
                    "provides": semantic["provides"],
                    "gameplay_capabilities": semantic["gameplay_capabilities"],
                    "implementation_capabilities": semantic["implementation_capabilities"],
                    "artifact_task_ids": semantic["artifact_task_ids"],
                    "semantic_status": semantic["semantic_status"],
                    "unresolved_spans": semantic["unresolved_spans"],
                    "acceptance": list(
                        _requirement_acceptance(
                            str(semantic["capability"]), matching_acceptance
                        )
                    ),
                }
            )

    merged = dict(catalog)
    merged["requirements"] = requirements
    merged["catalog_sha256"] = ""
    merged["catalog_sha256"] = _hash_without(merged, "catalog_sha256")
    return merged


def build_request_catalog(
    prompt: str,
    game_design: Mapping[str, Any],
    router: Any | None = None,
) -> dict[str, Any]:
    """Build an evidence-first requirement catalog from a user prompt.

    Parameters
    ----------
    prompt:
        Raw user prompt (any UTF-8 script).
    game_design:
        Game design mapping — may contain an existing ``_evidence_request_catalog``
        to merge against.
    router:
        Optional ``ModelRouter`` instance.  When provided, each structural clause
        is interpreted by the real Semantic Model (LLM) via ``_model_semantic_ir``.
        When ``None``, the deterministic alias-resolver stub is used instead.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvidencePlanError("Evidence-first planning requires a non-empty request.")
    existing = game_design.get("_evidence_request_catalog")
    if isinstance(existing, Mapping):
        catalog = dict(existing)
        _validate_request_catalog(catalog, prompt=prompt)
        return _merge_catalog_uncovered_prompt_requirements(
            catalog, prompt, game_design, router=router
        )
    # _capability_records returns design-module-backed entries; prompt clauses are not.
    records_raw = _capability_records(game_design)
    covered_spans = {
        span
        for _capability, statement in records_raw
        for span in (_matched_source_span(prompt, statement),)
        if span is not None
    }
    # 3-tuple: (capability, statement, is_design_module)
    merged_records: list[tuple[str, str, bool]] = [
        (cap, stmt, True) for cap, stmt in records_raw
    ]
    for start, end in _semantic_clause_spans(prompt):
        if (start, end) in covered_spans:
            continue
        statement = prompt[start:end]
        merged_records.append((_capability_from_statement(statement), statement, False))
    # Deduplicate by normalized capability while preserving order.
    seen: set[str] = set()
    records_deduped: list[tuple[str, str, bool]] = []
    for cap, stmt, from_dm in merged_records:
        normalized = cap.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            records_deduped.append((cap.strip(), stmt.strip() or cap.strip(), from_dm))
    records: tuple[tuple[str, str, bool], ...] = tuple(records_deduped)
    if not records:
        raise EvidencePlanError("The request did not yield any semantic requirement.")

    acceptance_source = _strings(game_design.get("acceptance_tests"))
    requirements: list[dict[str, Any]] = []
    for index, (capability, statement, from_design_module) in enumerate(records):
        span = _source_span(prompt, statement)
        matching_acceptance = tuple(
            item
            for item in acceptance_source
            if _word_overlap(item, f"{capability} {statement}")
        )
        ir = _invoke_semantic_model(
            statement, span["char_start"], span["char_end"], prompt, router
        )
        _validate_semantic_ir(ir, prompt)
        variants = (ir,) if from_design_module else _semantic_ir_variants(ir)
        for variant_index, variant_ir in enumerate(variants):
            gameplay, _implementation, _unresolved = _expand_semantic_ir(variant_ir)
            semantic_id = capability if from_design_module else (
                gameplay[0] if gameplay else capability
            )
            discriminator = {
                "prompt_sha256": _sha(prompt),
                "index": index,
                "span": [span["char_start"], span["char_end"]],
            }
            if len(variants) > 1:
                discriminator["semantic_variant"] = variant_index
            requirement_id = _stable_id("req", semantic_id, discriminator)
            semantic = _semantic_requirement_fields(
                capability, variant_ir, requirement_id,
                is_design_module=from_design_module,
            )
            acceptance = _requirement_acceptance(
                str(semantic["capability"]), matching_acceptance
            )
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "capability": semantic["capability"],
                    "statement": statement,
                    "mandatory": True,
                    "source_span": span,
                    "provides": semantic["provides"],
                    "gameplay_capabilities": semantic["gameplay_capabilities"],
                    "implementation_capabilities": semantic["implementation_capabilities"],
                    "artifact_task_ids": semantic["artifact_task_ids"],
                    "semantic_status": semantic["semantic_status"],
                    "unresolved_spans": semantic["unresolved_spans"],
                    "acceptance": list(acceptance),
                }
            )

    purpose = str(game_design.get("pitch") or game_design.get("description") or prompt).strip()
    catalog: dict[str, Any] = {
        "prompt_sha256": _sha(prompt),
        "prompt_char_length": len(prompt),
        "purpose": purpose,
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
    if catalog.get("prompt_sha256") != _sha(prompt) or catalog.get("prompt_char_length") != len(prompt):
        raise EvidencePlanError("Pre-target request catalog is stale for the supplied prompt.")
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise EvidencePlanError("Pre-target request catalog has no requirements.")
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise EvidencePlanError("Pre-target request requirement must be an object.")
        span = _mapping(requirement.get("source_span"))
        start, end = span.get("char_start"), span.get("char_end")
        text = str(span.get("text") or "")
        if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(prompt)):
            raise EvidencePlanError("Pre-target request source span is invalid.")
        if prompt[start:end] != text or span.get("text_sha256") != _sha(text):
            raise EvidencePlanError("Pre-target request source receipt is stale.")
        if bool(requirement.get("mandatory", True)) and requirement.get(
            "semantic_status", "RESOLVED"
        ) == "UNRESOLVED":
            raise EvidencePlanError(
                "Mandatory request text is unresolved; generation is blocked."
            )
        provides = tuple(
            _canonical_capability(value).removeprefix("capability:")
            for value in _strings(requirement.get("provides"))
        )
        if bool(requirement.get("mandatory", True)) and not provides:
            raise EvidencePlanError("Mandatory request requirement has no capability.")
        if provides and all(
            value.startswith(("block_entity.", "packet.", "registry.", "screen."))
            for value in provides
        ):
            raise EvidencePlanError(
                "A technical implementation primitive cannot be a top-level user requirement."
            )
        # Replay stored Semantic Model output only; never reinterpret raw text.
        gameplay_caps = frozenset(
            str(v).removeprefix("capability:")
            for v in requirement.get("gameplay_capabilities", ())
            if str(v).strip()
        )
        provides_set = frozenset(provides)
        stored_status = str(requirement.get("semantic_status", "RESOLVED"))
        if stored_status == "UNRESOLVED" and not gameplay_caps:
            raise EvidencePlanError(
                "Mandatory request text is unresolved; generation is blocked."
            )
        if gameplay_caps and not gameplay_caps.intersection(provides_set):
            raise EvidencePlanError(
                "Requirement semantic binding lost a gameplay capability."
            )


def _word_overlap(left: str, right: str) -> bool:
    token = re.compile(r"[\w]{2,}", re.UNICODE)
    a = {item.casefold() for item in token.findall(left)}
    b = {item.casefold() for item in token.findall(right)}
    return bool(a & b)


def _normalize_sha(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", text):
        text = "sha256:" + text
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
            # Only the dedicated immutable inventory validator may attest a
            # same-project component.  Arbitrary design/model mappings never do.
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
        locator = str(item.get("locator") or item.get("path") or item.get("symbol") or "").strip()
        provides = _strings(item.get("provides"))
        identifier = str(item.get("component_id") or "").strip()
        if not identifier:
            identifier = _stable_id("component", locator or "receipt", {"index": index, "provides": provides})
        if not _COMPONENT_ID_RE.fullmatch(identifier):
            identifier = _stable_id("component", identifier, {"locator": locator, "provides": provides})
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
            item.get("content_sha256") or item.get("sha256") or item.get("content_hash")
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
                and str(evidence.get("locator") or evidence.get("locator_id") or "").strip()
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
        verified = (
            inventory_attested
            and origin in {"same_project", "existing_project", "workspace"}
            and same_project_complete
            and bool(evidence_refs)
        )
        component: dict[str, Any] = {
            "component_id": identifier,
            "kind": str(item.get("kind") or "symbol").strip().casefold(),
            "locator": locator,
            "content_sha256": content_sha256,
            "provides": list(provides),
            "requires": list(_strings(item.get("requires"))),
            "target": _mapping(item.get("target")) or {
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


def _target_decision(game_design: Mapping[str, Any], target_decision: Any = None) -> dict[str, Any]:
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
    policy = "preserve" if raw.get("preserved_existing_target") else (
        "migrate" if raw.get("migration_requested") else "new"
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
            if isinstance(item, Mapping) and str(item.get("module_id") or "")
        ],
        "loaders": list(_strings(inventory_target.get("loaders"))),
        "source_sets": sorted(
            {
                str(source_set)
                for item in topology_modules
                if isinstance(item, Mapping)
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
            if key == selected_key:
                continue
            rejected.append(
                {
                    "target": candidate_target,
                    "total_expected_cost": candidate.get("total_expected_cost"),
                    "reason": "ranked_below_selected_after_hard_gates_and_verified_reuse",
                }
            )
    target_resolved = bool(
        str(target.get("minecraft_version") or "").strip()
        and str(target.get("minecraft_version") or "").strip().casefold() != "unresolved"
        and str(target.get("loader") or "").strip()
        and str(target.get("loader") or "").strip().casefold() != "unresolved"
    )
    result: dict[str, Any] = {
        "policy": policy,
        "coordinates": target,
        "hard_gate_status": "passed" if target_resolved else "deferred",
        "preserved_existing_target": bool(raw.get("preserved_existing_target")),
        "migration_requested": bool(raw.get("migration_requested")),
        "decision_reason": str(raw.get("reason") or optimizer.get("selection_basis") or "host target input"),
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


def _reuse_decisions(
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    reuse_plan: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_by_capability = {
        str(item.get("capability") or "").strip(): item
        for item in reuse_plan.get("capabilities", ())
        if isinstance(item, Mapping) and str(item.get("capability") or "").strip()
    } if isinstance(reuse_plan.get("capabilities"), list) else {}
    result: list[dict[str, Any]] = []
    for requirement in requirements:
        capability = str(requirement["capability"])
        raw = _mapping(raw_by_capability.get(capability))
        requested_component_refs = set(_strings(raw.get("component_refs")))
        matches = [
            str(item["component_id"])
            for item in components
            if _canonical_capability(capability) in {
                    _canonical_capability(value)
                    for value in _strings(item.get("provides"))
                    if value.casefold().startswith("capability:")
                }
            and item.get("verification_status") == "verified"
        ]
        project_matches = [
            str(item["component_id"])
            for item in components
            if str(item["component_id"]) in matches and item.get("bound_to_project") is True
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
        residual = (
            "none; verified project-bound component is retained"
            if action == "retain"
            else str(raw.get("rationale") or "implement and independently verify the missing capability")
        )
        decision: dict[str, Any] = {
            "decision_id": _stable_id("reuse", capability, {"action": action, "refs": refs}),
            "requirement_ref": requirement["requirement_id"],
            "capability": capability,
            "action": action,
            "component_refs": refs,
            "source_refs": source_refs,
            "external_receipt": dict(raw) if action == "adapt" else {},
            "evidence_status": evidence_status,
            "residual_work": residual,
            "source_mode": mode,
            "decision_sha256": "",
        }
        decision["decision_sha256"] = _hash_without(decision, "decision_sha256")
        result.append(decision)
    return tuple(result)


def _validated_external_reuse(
    raw: Mapping[str, Any],
    *,
    capability: str,
    target: Mapping[str, Any],
) -> bool:
    """Recognize only host-produced registry/API/donor receipt shapes.

    These receipts remain adaptation inputs, never project-bound provides; therefore
    they cannot remove an implementation gap.
    """
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
            and parsed.minecraft_version == str(coordinates.get("minecraft_version") or "")
            and parsed.loader == str(coordinates.get("loader") or "").casefold()
        )
    if mode not in {"source_transplant", "adapt"}:
        return False
    if donor.get("schema_version") != "mmm/source-transplant-slice-v1":
        return False
    if _canonical_capability(donor.get("capability")) != _canonical_capability(capability):
        return False
    if not str(donor.get("repository") or "").strip():
        return False
    if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor.get("commit_sha") or "")):
        return False
    if str(donor.get("license_id") or "") not in {
        "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib", "Unlicense", "CC0-1.0",
    }:
        return False
    if donor.get("target_compatibility") not in {"exact", "adapt"}:
        return False
    files = donor.get("files")
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, Mapping):
            return False
        path = str(item.get("path") or "").replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            return False
        if not _SHA_RE.fullmatch(str(item.get("sha256") or "")):
            return False
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(item.get("blob_sha") or "")):
            return False
        if type(item.get("size_bytes")) is not int or item["size_bytes"] < 0:
            return False
    proof = raw.get("proof_receipt")
    if not isinstance(proof, Mapping):
        return False
    from .proof_level import ProofLevel

    if proof.get("schema_version") != "mmm/reuse-proof-receipt-v1":
        return False
    if not ProofLevel.from_value(proof.get("proof_level")).is_verified():
        return False
    if proof.get("authoritative_compile") is not True or proof.get("compile_passed") is not True:
        return False
    if str(proof.get("commit_sha") or "") != str(donor.get("commit_sha") or ""):
        return False
    if _canonical_capability(proof.get("capability")) != _canonical_capability(capability):
        return False
    verified_artifacts = {
        str(item).replace("\\", "/")
        for item in proof.get("verified_artifacts", ())
        if str(item).strip()
    }
    donor_artifacts = {
        str(item.get("path") or "").replace("\\", "/")
        for item in files
        if isinstance(item, Mapping)
    }
    return bool(donor_artifacts and donor_artifacts <= verified_artifacts)


def _branch_predicates(
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    capabilities = tuple(str(item["capability"]).casefold() for item in requirements)
    component_kinds = tuple(str(item.get("kind") or "").casefold() for item in components)
    joined = " ".join(
        [
            *capabilities,
            *(
                str(item.get("statement") or "").casefold()
                for item in requirements
            ),
            *(
                str(_mapping(item.get("source_span")).get("text") or "").casefold()
                for item in requirements
            ),
        ]
    )
    topology = _mapping(target.get("project_topology"))
    loaders = _strings(topology.get("loaders"))
    conditions = {
        "needs_registry": any(
            term in joined
            for term in ("item", "block", "block_entity", "machine", "entity", "recipe", "effect", "enchantment", "fluid", "biome", "dimension", "registry")
        ),
        "needs_datagen": any(term in joined for term in ("recipe", "loot", "tag", "model", "worldgen", "datagen"))
        or "generated_resource" in component_kinds,
        "needs_persistence": any(term in joined for term in ("persistence", "saved", "storage", "serialize", "codec", "world_state")),
        "needs_network": any(term in joined for term in ("network", "payload", "packet", "sync")),
        "needs_client_render": any(term in joined for term in ("gui", "screen", "render", "texture", "model", "client", "hud")),
        "needs_worldgen": any(term in joined for term in ("worldgen", "biome", "configured_feature", "placed_feature", "structure", "dimension")),
        "needs_mixin": any(term in joined for term in ("mixin", "optimization", "performance", "renderer_patch", "injection")),
        "needs_loader_leaf": len(loaders) > 1 or any(term in joined for term in ("loader_leaf", "multiloader", "multi_loader")),
    }
    result: dict[str, dict[str, Any]] = {}
    requirement_refs = [str(item["requirement_id"]) for item in requirements]
    for name in _BRANCHES:
        active = bool(conditions[name])
        result[name] = {
            "predicate": name,
            "status": "ACTIVE" if active else "NOT_APPLICABLE",
            "evidence_refs": requirement_refs if active else ["request-catalog:no-matching-capability"],
            "reason": (
                "activated by requirement/component evidence"
                if active
                else "no requirement or project evidence activates this branch"
            ),
        }
    return result


def _required_gates(capability: str, branches: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    active = {name for name, value in branches.items() if value.get("status") == "ACTIVE"}
    gates = ["source_static_validation", "target_compile"]
    if "needs_datagen" in active and any(term in capability.casefold() for term in ("recipe", "loot", "tag", "model", "worldgen")):
        gates.append("generated_resource_validation")
    if "needs_network" in active and any(term in capability.casefold() for term in ("network", "payload", "packet", "sync")):
        gates.append("network_protocol_validation")
    if "needs_worldgen" in active and any(term in capability.casefold() for term in ("worldgen", "biome", "feature", "structure", "dimension")):
        gates.append("worldgen_runtime_validation")
    if "needs_mixin" in active and any(term in capability.casefold() for term in ("mixin", "optimization", "performance")):
        gates.extend(("behavior_equivalence", "performance_regression"))
    return tuple(dict.fromkeys(gates))


@dataclass(frozen=True)
class _Step:
    name: str
    outcome: str
    consumes: tuple[str, ...]
    provides: tuple[str, ...]
    anchor_kinds: tuple[str, ...]


def _active(branches: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    value = branches.get(name)
    return isinstance(value, Mapping) and value.get("status") == "ACTIVE"


def _semantic_steps(
    capability: str,
    branches: Mapping[str, Mapping[str, Any]],
) -> tuple[_Step, ...]:
    folded = capability.casefold()
    root = "target:frozen"
    if any(term in folded for term in ("worldgen", "biome", "placed_feature", "configured_feature", "structure", "dimension")):
        configured = f"configured:{capability}"
        placed = f"placed:{capability}"
        return (
            _Step("configured_feature", f"Define the configured world-generation contract for {capability}", (root,), (configured,), ("symbol", "resource")),
            _Step("placed_feature", f"Bind placement rules for {capability}", (configured,), (placed,), ("symbol", "resource")),
            _Step("biome_binding", f"Attach {capability} to its approved world-generation targets", (placed,), (capability,), ("symbol", "resource", "test")),
        )
    if any(term in folded for term in ("network", "payload", "packet", "sync")):
        codec = f"payload_codec:{capability}"
        return (
            _Step("payload_codec", f"Define and register the payload codec for {capability}", (root,), (codec,), ("symbol", "registry_id", "test")),
            _Step("server_handler", f"Implement side-safe handling and validation for {capability}", (codec,), (capability,), ("symbol", "test")),
        )
    if any(term in folded for term in ("persistence", "saved", "storage", "serialize", "world_state")):
        codec = f"state_codec:{capability}"
        return (
            _Step("state_codec", f"Define the versioned state/codec contract for {capability}", (root,), (codec,), ("symbol", "test")),
            _Step("state_store", f"Persist, reload, and validate {capability}", (codec,), (capability,), ("symbol", "test")),
        )
    if any(term in folded for term in ("mixin", "optimization", "performance")):
        baseline = f"baseline:{capability}"
        patch = f"patch:{capability}"
        return (
            _Step("baseline_contract", f"Capture correctness and performance baselines for {capability}", (root,), (baseline,), ("test",)),
            _Step("optimization_patch", f"Apply one compatibility-gated optimization for {capability}", (baseline,), (patch,), ("symbol", "build_config")),
            _Step("regression_proof", f"Prove behavior equivalence and performance for {capability}", (patch,), (capability,), ("test",)),
        )
    if any(term in folded for term in ("gui", "screen", "hud", "client_render")):
        contract = f"client_contract:{capability}"
        return (
            _Step("client_contract", f"Define the side-safe client contract for {capability}", (root,), (contract,), ("symbol",)),
            _Step("client_surface", f"Implement and verify the client surface for {capability}", (contract,), (capability,), ("symbol", "resource", "test")),
        )
    if "menu" in folded:
        contract = f"menu_contract:{capability}"
        return (
            _Step("menu_contract", f"Define authoritative menu state and slots for {capability}", (root,), (contract,), ("symbol", "registry_id", "test")),
            _Step("menu_binding", f"Bind and verify the menu interaction for {capability}", (contract,), (capability,), ("symbol", "test")),
        )
    if any(term in folded for term in ("machine", "block_entity")):
        registry = f"registry_id:{capability}"
        shell = f"block_shell:{capability}"
        item = f"block_item:{capability}"
        state = f"block_entity_state:{capability}"
        behavior = f"behavior:{capability}"
        resources = f"resources:{capability}"
        steps: list[_Step] = [
            _Step("registry_identity", f"Reserve stable registry identities for {capability}", (root,), (registry,), ("registry_id",)),
            _Step("block_shell", f"Implement the block shell for {capability}", (registry,), (shell,), ("symbol",)),
            _Step("block_item", f"Bind the block item for {capability}", (shell,), (item,), ("symbol", "resource")),
            _Step("block_entity_type_state", f"Register the block-entity type and define state ownership for {capability}", (item,), (state,), ("symbol", "registry_id", "test")),
            _Step("server_behavior", f"Implement authoritative server behavior for {capability}", (state,), (behavior,), ("symbol", "test")),
        ]
        final_inputs = [behavior]
        if _active(branches, "needs_persistence"):
            codec = f"state_codec:{capability}"
            persisted = f"persisted:{capability}"
            steps.extend(
                (
                    _Step("state_codec", f"Define a versioned persistence codec for {capability}", (state,), (codec,), ("symbol", "test")),
                    _Step("persistence_binding", f"Persist and reload authoritative state for {capability}", (codec,), (persisted,), ("symbol", "test")),
                )
            )
            final_inputs.append(persisted)
        if _active(branches, "needs_network"):
            payload = f"payload_codec:{capability}"
            synced = f"synced:{capability}"
            steps.extend(
                (
                    _Step("payload_codec", f"Define and register the side-safe payload codec for {capability}", (state,), (payload,), ("symbol", "registry_id", "test")),
                    _Step("payload_handler_sync", f"Validate handlers and synchronize {capability}", (payload, behavior), (synced,), ("symbol", "test")),
                )
            )
            final_inputs.append(synced)
        if _active(branches, "needs_client_render"):
            menu = f"menu_contract:{capability}"
            screen = f"screen:{capability}"
            steps.extend(
                (
                    _Step("menu_contract", f"Define authoritative menu slots and state for {capability}", (behavior,), (menu,), ("symbol", "registry_id", "test")),
                    _Step("client_screen", f"Render and bind the client screen for {capability}", (menu,), (screen,), ("symbol", "resource", "test")),
                )
            )
            final_inputs.append(screen)
        steps.extend(
            (
                _Step("resource_binding", f"Bind models, language, loot, and resource references for {capability}", (item,), (resources,), ("resource", "test")),
                _Step("integration_proof", f"Verify the complete semantic outcome for {capability}", (*final_inputs, resources), (capability,), ("test",)),
            )
        )
        return tuple(steps)
    if any(term in folded for term in ("item", "block", "entity", "effect", "enchantment", "fluid")):
        registry = f"registry_id:{capability}"
        behavior = f"behavior:{capability}"
        return (
            _Step("registry_identity", f"Reserve the stable registry identity for {capability}", (root,), (registry,), ("registry_id",)),
            _Step("registered_behavior", f"Implement and register behavior for {capability}", (registry,), (behavior,), ("symbol", "test")),
            _Step("resource_binding", f"Bind all required data and client resources for {capability}", (behavior,), (capability,), ("resource", "test")),
        )
    if any(term in folded for term in ("recipe", "loot", "tag", "advancement")):
        return (
            _Step("data_definition", f"Define and validate data resources for {capability}", (root,), (capability,), ("resource", "test")),
        )
    return (
        _Step("semantic_implementation", f"Implement one independently verifiable outcome for {capability}", (root,), (capability,), ("symbol", "test")),
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
            if isinstance(item, Mapping) and str(item.get("module_id") or "")
        ],
        "topology_source_sets": sorted(
            {
                str(source_set)
                for item in topology_modules
                if isinstance(item, Mapping)
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
    step: _Step,
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
                "source_set": "loader_leaf" if "common" not in module_id.casefold() else "common",
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
            "source_set": ownership["source_set"],
        }
        for kind in step.anchor_kinds
    ]


def _compile_tasks(
    gaps: Sequence[Mapping[str, Any]],
    reuse: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    reuse_by_req = {str(item["requirement_ref"]): item for item in reuse}
    tasks: list[dict[str, Any]] = []
    for gap in gaps:
        requirement_ref = str(gap["requirement_ref"])
        capability = str(gap.get("capability") or gap["missing_provides"][0])
        required_provide = str(gap["missing_provides"][0])
        decision = reuse_by_req.get(requirement_ref, {})
        steps = _semantic_steps(capability, branches)
        if _active(branches, "needs_loader_leaf"):
            common = f"common_contract:{capability}"
            rewritten: list[_Step] = []
            replaced = False
            for step in steps:
                if capability in step.provides:
                    rewritten.append(
                        _Step(
                            step.name,
                            step.outcome,
                            step.consumes,
                            tuple(common if item == capability else item for item in step.provides),
                            step.anchor_kinds,
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
                _Step(
                    "loader_leaf_binding",
                    f"Bind the common contract for {capability} into every approved loader leaf",
                    (common,),
                    (capability,),
                    ("build_config", "symbol", "test"),
                )
            )
            steps = tuple(rewritten)
        steps = tuple(
            _Step(
                step.name,
                step.outcome,
                step.consumes,
                tuple(required_provide if item == capability else item for item in step.provides),
                step.anchor_kinds,
            )
            for step in steps
        )
        for index, step in enumerate(steps):
            task_id = _stable_id(
                "task",
                f"{capability}_{step.name}",
                {"gap": gap["gap_id"], "index": index},
            )
            active_predicates = [
                name
                for name, value in branches.items()
                if value.get("status") == "ACTIVE" and _step_uses_branch(step, name)
            ]
            # Task-local validation is executable DAG state. Public/release
            # acceptance is projected separately from requirement acceptance.
            acceptance = [
                f"{task_id}: all declared provides exist and all owned anchors pass their integrity checks"
            ]
            if index == len(steps) - 1:
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
                "required_gates": list(_required_gates(capability, branches)),
                "acceptance": acceptance,
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
                "state": "pending",
                "task_sha256": "",
            }
            task["task_sha256"] = _hash_without(task, "task_sha256")
            tasks.append(task)
    return _bind_consumes_dependencies(tasks, root_provides={"target:frozen"})


def _bind_consumes_dependencies(
    tasks: Sequence[Mapping[str, Any]],
    *,
    root_provides: set[str],
) -> tuple[dict[str, Any], ...]:
    """Bind every non-root consume to exactly one provider and a direct DAG edge."""
    providers: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        for provided in _strings(task.get("provides")):
            providers.setdefault(provided, []).append(task_id)
    ambiguous = {
        capability: ids
        for capability, ids in providers.items()
        if len(ids) != 1
    }
    if ambiguous:
        raise EvidencePlanError(
            "Semantic provides require one host-selected provider: "
            + _canonical(ambiguous)
        )
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
                raise EvidencePlanError(f"Task {task_id} consumes its own provide {consumed!r}.")
            if provider not in dependencies:
                dependencies.append(provider)
        task["depends_on"] = dependencies
        task["task_sha256"] = ""
        task["task_sha256"] = _hash_without(task, "task_sha256")
        bound.append(task)
    return tuple(bound)


def _step_uses_branch(step: _Step, branch: str) -> bool:
    text = f"{step.name} {step.outcome}".casefold()
    terms = {
        "needs_registry": ("registry", "block", "item", "entity"),
        "needs_datagen": ("resource", "data", "world-generation"),
        "needs_persistence": ("state", "persist", "codec"),
        "needs_network": ("payload", "handler", "sync"),
        "needs_client_render": ("client", "screen", "resource", "model"),
        "needs_worldgen": ("world-generation", "feature", "biome", "placement"),
        "needs_mixin": ("optimization", "performance", "baseline", "equivalence"),
        "needs_loader_leaf": ("loader", "common contract"),
    }
    return any(term in text for term in terms[branch])


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
                raise EvidencePlanError(f"Task {task_id} references unknown dependency {dependency}.")
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
        raise EvidencePlanError(f"Semantic implementation graph contains a cycle: {cyclic[:20]}")
    return order


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
        prompt, game_design, router=semantic_router
    )
    # Do not let an unresolved mandatory span or a primitive-only capability
    # reach reuse discovery or coder generation.  Direct catalog inspection may
    # expose unresolved work for review; production compilation may not proceed.
    _validate_request_catalog(request_catalog, prompt=prompt)
    pre_retrieval = game_design.get("_pre_retrieval_plan")
    pre_retrieval_sha256 = ""
    if isinstance(pre_retrieval, Mapping):
        from .reuse_planner import validate_pre_retrieval_plan

        validate_pre_retrieval_plan(
            pre_retrieval,
            prompt=prompt,
            design=game_design,
        )
        pre_retrieval_sha256 = str(pre_retrieval.get("plan_sha256") or "")
    components = normalize_component_catalog(game_design, component_catalog)
    reuse_payload = dict(reuse_plan) if isinstance(reuse_plan, Mapping) else _reuse_payload(game_design)
    reuse_graph = _mapping(reuse_payload.get("capability_graph"))
    graph_plan_sha256 = str(reuse_graph.get("source_plan_sha256") or "")
    if pre_retrieval_sha256 and reuse_payload and graph_plan_sha256 != pre_retrieval_sha256:
        raise EvidencePlanError(
            "Reuse evidence is not bound to the frozen pre-retrieval semantic plan."
        )
    target = _target_decision(game_design, target_decision)
    if target.get("hard_gate_status") != "passed":
        raise EvidencePlanError(
            "Target decision is unresolved; semantic implementation planning is deferred."
        )
    decisions = _reuse_decisions(
        request_catalog["requirements"],
        components,
        reuse_payload,
        target,
    )
    verified = _verified_project_provides(components)
    verified.update(
        _canonical_capability(item["capability"])
        for item in decisions
        if item.get("action") == "retain" and item.get("evidence_status") == "verified"
    )

    gaps: list[dict[str, Any]] = []
    for requirement in request_catalog["requirements"]:
        missing = [item for item in requirement["provides"] if item not in verified]
        if not missing:
            continue
        gap: dict[str, Any] = {
            "gap_id": _stable_id("gap", str(requirement["capability"]), requirement["requirement_id"]),
            "requirement_ref": requirement["requirement_id"],
            "capability": requirement["capability"],
            "missing_provides": missing,
            "reason": "not supplied by a verified project-bound component receipt",
            "required_gates": ["source_static_validation", "target_compile"],
            "acceptance": list(requirement["acceptance"]),
            "gap_sha256": "",
        }
        gap["gap_sha256"] = _hash_without(gap, "gap_sha256")
        gaps.append(gap)

    branches = _branch_predicates(request_catalog["requirements"], components, target)
    ownership = _ownership_context(game_design)
    target_topology = _mapping(target.get("project_topology"))
    topology_ids = list(_strings(target_topology.get("module_ids")))
    if topology_ids and not ownership.get("topology_module_ids"):
        ownership["topology_module_ids"] = topology_ids
        common = next(
            (item for item in topology_ids if "common" in item.casefold()),
            topology_ids[0],
        )
        ownership["module_id"] = common
    if target_topology.get("source_sets") and not ownership.get("topology_source_sets"):
        ownership["topology_source_sets"] = list(
            _strings(target_topology.get("source_sets"))
        )
    tasks = _compile_tasks(
        gaps,
        decisions,
        target,
        branches,
        ownership,
    )
    order = _topological(tasks)
    by_id = {str(item["task_id"]): item for item in tasks}
    tasks = tuple(by_id[item] for item in order)
    task_refs_by_req: dict[str, list[str]] = {}
    for task in tasks:
        for requirement_ref in task["requirement_refs"]:
            task_refs_by_req.setdefault(str(requirement_ref), []).append(str(task["task_id"]))
    component_refs_by_capability: dict[str, list[str]] = {}
    for component in components:
        if component.get("verification_status") != "verified" or component.get("bound_to_project") is not True:
            continue
        for capability in component["provides"]:
            if not str(capability).casefold().startswith("capability:"):
                continue
            component_refs_by_capability.setdefault(
                _canonical_capability(capability), []
            ).append(str(component["component_id"]))
    for decision in decisions:
        if decision.get("action") != "retain" or decision.get("evidence_status") != "verified":
            continue
        component_refs_by_capability.setdefault(_canonical_capability(decision["capability"]), []).extend(
            str(item) for item in decision.get("component_refs", ())
        )
    component_refs_by_capability = {
        capability: list(dict.fromkeys(refs))
        for capability, refs in component_refs_by_capability.items()
    }
    bindings = [
        {
            "requirement_ref": requirement["requirement_id"],
            "capability": requirement["capability"],
            "component_refs": component_refs_by_capability.get(_canonical_capability(requirement["capability"]), []),
            "task_refs": task_refs_by_req.get(str(requirement["requirement_id"]), []),
            "acceptance": list(requirement["acceptance"]),
            "status": "retained" if set(requirement["provides"]) <= verified else "planned_gap",
        }
        for requirement in request_catalog["requirements"]
    ]
    root_provides = ["target:frozen", *sorted(verified)]
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
        "root_provides": root_provides,
        "tasks": list(tasks),
        "acceptance_release_bindings": bindings,
        "run_state": {
            "inventory_revision": _sha(list(components)),
            "graph_revision": _sha([{"id": item["task_id"], "depends_on": item["depends_on"]} for item in tasks]),
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


def validate_evidence_first_plan(plan: Mapping[str, Any], *, prompt: str | None = None) -> None:
    if plan.get("schema_version") != SCHEMA:
        raise EvidencePlanError("Unsupported evidence-first planning schema.")
    if plan.get("plan_sha256") != _hash_without(plan, "plan_sha256"):
        raise EvidencePlanError("Evidence-first plan hash mismatch.")
    pre_retrieval_sha256 = str(plan.get("pre_retrieval_plan_sha256") or "")
    if pre_retrieval_sha256 and not _SHA_RE.fullmatch(pre_retrieval_sha256):
        raise EvidencePlanError("Pre-retrieval semantic plan reference is invalid.")
    request = _mapping(plan.get("request_catalog"))
    if request.get("catalog_sha256") != _hash_without(request, "catalog_sha256"):
        raise EvidencePlanError("Request catalog hash mismatch.")
    if prompt is not None and (
        request.get("prompt_sha256") != _sha(prompt)
        or request.get("prompt_char_length") != len(prompt)
    ):
        raise EvidencePlanError("Request catalog is stale for the supplied prompt.")
    requirements = request.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise EvidencePlanError("Request catalog has no requirements.")
    requirement_ids: set[str] = set()
    requirement_by_id: dict[str, Mapping[str, Any]] = {}
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise EvidencePlanError("Request requirement must be an object.")
        identifier = str(requirement.get("requirement_id") or "")
        if not _ID_RE.fullmatch(identifier) or identifier in requirement_ids:
            raise EvidencePlanError("Request requirement identifiers are invalid or duplicated.")
        requirement_ids.add(identifier)
        requirement_by_id[identifier] = requirement
        span = _mapping(requirement.get("source_span"))
        text = str(span.get("text") or "")
        if span.get("text_sha256") != _sha(text):
            raise EvidencePlanError(f"Requirement {identifier} source hash is stale.")
        if prompt is not None:
            start, end = span.get("char_start"), span.get("char_end")
            if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(prompt)):
                raise EvidencePlanError(f"Requirement {identifier} source span is invalid.")
            if prompt[start:end] != text:
                raise EvidencePlanError(f"Requirement {identifier} source span changed.")

    components = plan.get("component_catalog")
    if not isinstance(components, list):
        raise EvidencePlanError("Component catalog must be a list.")
    component_ids: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            raise EvidencePlanError("Component receipt must be an object.")
        identifier = str(component.get("component_id") or "")
        if not _COMPONENT_ID_RE.fullmatch(identifier) or identifier in component_ids:
            raise EvidencePlanError("Component identifiers are invalid or duplicated.")
        component_ids.add(identifier)
        if component.get("receipt_sha256") != _hash_without(component, "receipt_sha256"):
            raise EvidencePlanError(f"Component {identifier} receipt hash mismatch.")
        if component.get("verification_status") == "verified" and component.get("evidence_complete") is not True:
            raise EvidencePlanError(f"Component {identifier} claims verification without complete evidence.")

    target_payload = _mapping(plan.get("target_decision"))
    if target_payload.get("decision_sha256") != _hash_without(
        target_payload, "decision_sha256"
    ):
        raise EvidencePlanError("Target decision hash mismatch.")
    coordinates = _mapping(target_payload.get("coordinates"))
    if (
        target_payload.get("hard_gate_status") != "passed"
        or str(coordinates.get("minecraft_version") or "").casefold() in {"", "unresolved"}
        or str(coordinates.get("loader") or "").casefold() in {"", "unresolved"}
    ):
        raise EvidencePlanError("Semantic task graph requires a resolved target hard gate.")

    decisions = plan.get("reuse_decisions")
    if not isinstance(decisions, list):
        raise EvidencePlanError("Reuse decisions must be a list.")
    expected_decision_keys = {
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
    component_by_id = {
        str(item["component_id"]): item
        for item in components
        if isinstance(item, Mapping)
    }
    decision_ids: set[str] = set()
    decision_requirements: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != expected_decision_keys:
            raise EvidencePlanError("Reuse decision fields are invalid.")
        decision_id = str(decision.get("decision_id") or "")
        requirement_ref = str(decision.get("requirement_ref") or "")
        if not _ID_RE.fullmatch(decision_id) or decision_id in decision_ids:
            raise EvidencePlanError("Reuse decision identifiers are invalid or duplicated.")
        if requirement_ref not in requirement_ids or requirement_ref in decision_requirements:
            raise EvidencePlanError("Reuse decision requirement binding is invalid or duplicated.")
        expected_requirement = requirement_by_id[requirement_ref]
        capability = str(decision.get("capability") or "")
        if capability != str(expected_requirement.get("capability") or ""):
            raise EvidencePlanError("Reuse decision capability changed from its requirement.")
        component_refs = _strings(decision.get("component_refs"))
        source_refs = _strings(decision.get("source_refs"))
        if not set(component_refs) <= component_ids:
            raise EvidencePlanError("Reuse decision references an unknown component.")
        action = str(decision.get("action") or "")
        status = str(decision.get("evidence_status") or "")
        external_receipt = _mapping(decision.get("external_receipt"))
        if action == "retain":
            if status != "verified" or not component_refs or source_refs or external_receipt:
                raise EvidencePlanError("Retain decision lacks exact same-project evidence.")
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
                        "Retain decision references a component without an exact verified capability alias."
                    )
        elif action == "adapt":
            if status != "verified_external" or not source_refs or not external_receipt:
                raise EvidencePlanError("Adapt decision lacks a validated external receipt.")
            if not _validated_external_reuse(
                external_receipt,
                capability=capability,
                target=target_payload,
            ):
                raise EvidencePlanError("Adapt decision external receipt is invalid.")
            if source_refs[0] != f"external-reuse:{_sha(external_receipt)}":
                raise EvidencePlanError("Adapt decision source receipt hash mismatch.")
            for component_ref in component_refs:
                if component_by_id[component_ref].get("evidence_complete") is not True:
                    raise EvidencePlanError("Adapt decision references an incomplete candidate component.")
        elif action == "fresh":
            if component_refs or source_refs or external_receipt:
                raise EvidencePlanError("Fresh decision may not claim reuse evidence.")
            if status not in {"missing", "not_applicable"}:
                raise EvidencePlanError("Fresh decision evidence status is invalid.")
        else:
            raise EvidencePlanError(f"Unsupported reuse action: {action!r}")
        expected_id = _stable_id(
            "reuse",
            capability,
            {"action": action, "refs": list(component_refs)},
        )
        if decision_id != expected_id:
            raise EvidencePlanError("Reuse decision ID is not host-derived.")
        if decision.get("decision_sha256") != _hash_without(
            decision, "decision_sha256"
        ):
            raise EvidencePlanError(f"Reuse decision {decision_id} hash mismatch.")
        decision_ids.add(decision_id)
        decision_requirements.add(requirement_ref)
    if decision_requirements != requirement_ids:
        raise EvidencePlanError(
            f"Requirements without exactly one reuse decision: {sorted(requirement_ids - decision_requirements)}"
        )

    expected_verified = _verified_project_provides(components)
    expected_verified.update(
        _canonical_capability(item["capability"])
        for item in decisions
        if item.get("action") == "retain" and item.get("evidence_status") == "verified"
    )
    if plan.get("verified_provides") != sorted(expected_verified):
        raise EvidencePlanError("Verified provides do not match attested project components.")

    gaps = plan.get("gap_catalog")
    if not isinstance(gaps, list):
        raise EvidencePlanError("Gap catalog must be a list.")
    gap_ids: set[str] = set()
    gap_requirements: set[str] = set()
    for gap in gaps:
        if not isinstance(gap, Mapping):
            raise EvidencePlanError("Gap record must be an object.")
        identifier = str(gap.get("gap_id") or "")
        requirement_ref = str(gap.get("requirement_ref") or "")
        if not _ID_RE.fullmatch(identifier) or identifier in gap_ids:
            raise EvidencePlanError("Gap identifiers are invalid or duplicated.")
        if requirement_ref not in requirement_ids:
            raise EvidencePlanError(f"Gap {identifier} has an unknown requirement.")
        if gap.get("gap_sha256") != _hash_without(gap, "gap_sha256"):
            raise EvidencePlanError(f"Gap {identifier} hash mismatch.")
        gap_ids.add(identifier)
        gap_requirements.add(requirement_ref)

    expected_gaps: list[dict[str, Any]] = []
    for requirement in requirements:
        missing = [item for item in requirement["provides"] if item not in expected_verified]
        if not missing:
            continue
        expected_gap: dict[str, Any] = {
            "gap_id": _stable_id(
                "gap",
                str(requirement["capability"]),
                requirement["requirement_id"],
            ),
            "requirement_ref": requirement["requirement_id"],
            "capability": requirement["capability"],
            "missing_provides": missing,
            "reason": "not supplied by a verified project-bound component receipt",
            "required_gates": ["source_static_validation", "target_compile"],
            "acceptance": list(requirement["acceptance"]),
            "gap_sha256": "",
        }
        expected_gap["gap_sha256"] = _hash_without(expected_gap, "gap_sha256")
        expected_gaps.append(expected_gap)
    if _canonical(gaps) != _canonical(expected_gaps):
        raise EvidencePlanError("Gap catalog is not the exact requirements-minus-verified-provides set difference.")

    expected_branches = _branch_predicates(requirements, components, target_payload)
    if _canonical(plan.get("branch_predicates")) != _canonical(expected_branches):
        raise EvidencePlanError("Conditional branch predicates do not match request evidence.")
    ownership = _mapping(plan.get("ownership_context"))
    topology_ids = list(
        _strings(_mapping(target_payload.get("project_topology")).get("module_ids"))
    )
    if topology_ids:
        if list(_strings(ownership.get("topology_module_ids"))) != topology_ids:
            raise EvidencePlanError("Task ownership topology is not bound to the target decision.")
        if str(ownership.get("module_id") or "") not in topology_ids:
            raise EvidencePlanError("Default task owner is outside the approved topology.")
    expected_tasks = _compile_tasks(
        expected_gaps,
        decisions,
        target_payload,
        expected_branches,
        ownership,
    )
    expected_order = _topological(expected_tasks)
    expected_by_id = {str(item["task_id"]): item for item in expected_tasks}
    expected_tasks = tuple(expected_by_id[item] for item in expected_order)

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise EvidencePlanError("Semantic task graph must be a list.")
    if _canonical(tasks) != _canonical(expected_tasks):
        raise EvidencePlanError("Semantic tasks are not the deterministic host gap DAG.")
    _topological(tasks)
    roots = set(_strings(plan.get("root_provides")))
    providers: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        for provided in _strings(task.get("provides")):
            providers.setdefault(provided, []).append(task_id)
    ambiguous = {key: value for key, value in providers.items() if len(value) != 1}
    if ambiguous:
        raise EvidencePlanError(
            "Semantic provides have ambiguous providers: " + _canonical(ambiguous)
        )
    covered_gap_refs: set[str] = set()
    task_ids = {str(item["task_id"]) for item in tasks}
    for task in tasks:
        task_id = str(task["task_id"])
        if task.get("task_sha256") != _hash_without(task, "task_sha256"):
            raise EvidencePlanError(f"Task {task_id} hash mismatch.")
        gap_refs = set(_strings(task.get("gap_refs")))
        requirement_refs = set(_strings(task.get("requirement_refs")))
        if not gap_refs or not gap_refs <= gap_ids:
            raise EvidencePlanError(f"Task {task_id} has an unknown or empty gap binding.")
        if not requirement_refs or not requirement_refs <= requirement_ids:
            raise EvidencePlanError(f"Task {task_id} has an unknown requirement binding.")
        covered_gap_refs.update(gap_refs)
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
        actual_dependencies = set(_strings(task.get("depends_on")))
        if actual_dependencies != expected_dependencies:
            raise EvidencePlanError(
                f"Task {task_id} dependency edges do not exactly bind consumes; "
                f"expected={sorted(expected_dependencies)}, actual={sorted(actual_dependencies)}"
            )
        if not _strings(task.get("provides")):
            raise EvidencePlanError(f"Task {task_id} must declare provides.")
        if not isinstance(task.get("owned_anchors"), list) or not task["owned_anchors"]:
            raise EvidencePlanError(f"Task {task_id} must own at least one anchor.")
        if not _strings(task.get("acceptance")):
            raise EvidencePlanError(f"Task {task_id} must declare acceptance checks.")
    if gap_ids != covered_gap_refs:
        raise EvidencePlanError(f"Unbound implementation gaps: {sorted(gap_ids - covered_gap_refs)}")

    bindings = plan.get("acceptance_release_bindings")
    if not isinstance(bindings, list):
        raise EvidencePlanError("Acceptance release bindings must be a list.")
    bound_requirements: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise EvidencePlanError("Acceptance binding must be an object.")
        requirement_ref = str(binding.get("requirement_ref") or "")
        if requirement_ref not in requirement_ids:
            raise EvidencePlanError("Acceptance binding references an unknown requirement.")
        if not set(_strings(binding.get("component_refs"))) <= component_ids:
            raise EvidencePlanError("Acceptance binding references an unknown component.")
        if not set(_strings(binding.get("task_refs"))) <= task_ids:
            raise EvidencePlanError("Acceptance binding references an unknown task.")
        if not _strings(binding.get("component_refs")) and not _strings(binding.get("task_refs")):
            raise EvidencePlanError(f"Requirement {requirement_ref} has no retained component or planned task.")
        bound_requirements.add(requirement_ref)
    if bound_requirements != requirement_ids:
        raise EvidencePlanError(f"Unbound requirements: {sorted(requirement_ids - bound_requirements)}")

    expected_component_refs: dict[str, list[str]] = {}
    for component in components:
        if component.get("verification_status") != "verified" or component.get("bound_to_project") is not True:
            continue
        for capability in _strings(component.get("provides")):
            if capability.casefold().startswith("capability:"):
                expected_component_refs.setdefault(
                    _canonical_capability(capability), []
                ).append(str(component["component_id"]))
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
            "component_refs": list(
                dict.fromkeys(
                    expected_component_refs.get(
                        _canonical_capability(requirement["capability"]), []
                    )
                )
            ),
            "task_refs": task_refs_by_requirement.get(
                str(requirement["requirement_id"]), []
            ),
            "acceptance": list(requirement["acceptance"]),
            "status": (
                "retained"
                if set(requirement["provides"]) <= expected_verified
                else "planned_gap"
            ),
        }
        for requirement in requirements
    ]
    if _canonical(bindings) != _canonical(expected_bindings):
        raise EvidencePlanError("Acceptance release bindings are not host-derived.")

    branches = plan.get("branch_predicates")
    if not isinstance(branches, Mapping) or set(branches) != set(_BRANCHES):
        raise EvidencePlanError("Minecraft conditional branch catalog is incomplete.")
    for name, branch in branches.items():
        if not isinstance(branch, Mapping) or branch.get("status") not in {"ACTIVE", "NOT_APPLICABLE"}:
            raise EvidencePlanError(f"Branch {name} has an invalid state.")


def task_batches(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expose task records as host batches without transferring graph ownership."""
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
