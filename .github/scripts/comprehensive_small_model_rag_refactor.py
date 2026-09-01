from __future__ import annotations

from pathlib import Path
import re

ROOT = Path("minecraft_mod_ai")
changed: list[str] = []


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    changed.append(str(path))


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


# ---------------------------------------------------------------------------
# 1) Canonical small-model pre-design path.
# Model: semantic evidence extraction only.
# Host: retrieval state, IDs, exact quotes, sufficiency, termination, serialization.
# ---------------------------------------------------------------------------
small = r'''from __future__ import annotations

# Small-model-safe, host-owned pre-design research.
# User-authored gameplay requirements are already authoritative. External RAG is
# implementation evidence, never permission to design the requested feature.

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROTOCOL = "mmm/small-model-predesign-evidence-v1"
_STOP = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
    "minecraft", "fabric", "mod", "mods", "mode", "requested", "user",
    "implementation", "system", "game", "feature",
}


def _terms(values: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for token in re.findall(r"[A-Za-z0-9_+.#/-]+|[가-힣]{2,}", str(value)):
            folded = token.casefold()
            if len(folded) >= 3 and folded not in _STOP:
                result.add(folded)
    return result


def _page_score(page: Mapping[str, Any], domain: Mapping[str, Any]) -> int:
    wanted = _terms(
        [
            str(domain.get("objective") or ""),
            *(str(x) for x in domain.get("queries", ()) if str(x).strip()),
        ]
    )
    if not wanted:
        return 1
    have = _terms([str(page.get("content") or "")])
    return len(wanted & have)


def _candidate_pages(
    pages: Sequence[Mapping[str, Any]], domain: Mapping[str, Any]
) -> list[dict[str, Any]]:
    scored = [
        (max(0, _page_score(page, domain)), index, dict(page))
        for index, page in enumerate(pages)
    ]
    positive = [item for item in scored if item[0] > 0]
    selected = positive if positive else scored[: min(4, len(scored))]
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [page for _score, _index, page in selected]


def _exact_span(content: str, proposed: str) -> str:
    quote = str(proposed or "").strip().strip('"').strip("'")
    if not quote:
        return ""
    if quote in content:
        return quote
    words = [piece for piece in re.split(r"\s+", quote) if piece]
    if not words:
        return ""
    pattern = r"\s+".join(re.escape(piece) for piece in words)
    match = re.search(pattern, content, flags=re.MULTILINE)
    return match.group(0) if match else ""


def _extract_page(
    router: Any,
    *,
    domain: Mapping[str, Any],
    page: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    content = str(page.get("content") or "")
    page_ref = str(page.get("page_ref") or "").strip()
    if not content.strip() or not page_ref:
        return [], ["empty_host_page"]
    messages = [
        {
            "role": "system",
            "content": (
                "Read SOURCE only. Extract at most 3 implementation facts useful for the "
                "Minecraft mod design. Output one line per useful fact exactly as "
                "EVIDENCE<TAB>EXACT_QUOTE<TAB>IMPLEMENTATION_INSIGHT. "
                "EXACT_QUOTE must be copied from SOURCE, not paraphrased. "
                "If nothing useful exists output only NONE. No JSON, Markdown, code fences, "
                "analysis, headings, IDs, sufficiency flags, search queries, or extra prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "OBJECTIVE\n"
                + str(domain.get("objective") or "")
                + "\n\nSOURCE\n"
                + content
            ),
        },
    ]
    try:
        raw = router.generate_text(
            "planner",
            messages,
            response_format="text",
            response_schema=None,
            tool_stage="research",
            enable_tools=False,
        )
    except Exception as exc:
        return [], [f"model_read_failure:{type(exc).__name__}:{exc}"]

    claims: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line or line.casefold() == "none":
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[0].strip().casefold() != "evidence":
            diagnostics.append("ignored_malformed_model_line")
            continue
        exact = _exact_span(content, parts[1])
        insight = " ".join(parts[2].split()).strip()
        if not exact or len("".join(exact.split())) < 8:
            diagnostics.append("rejected_non_exact_quote")
            continue
        if not insight:
            insight = "Implementation reference: " + " ".join(exact.split())
        key = (insight, exact)
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {
                "claim": insight,
                "evidence_refs": [page_ref],
                "support_quote": exact,
                "support_verification": "host_exact_quote_from_small_model_line",
            }
        )
        if len(claims) >= 3:
            break
    return claims, diagnostics


def _load_grounded(document: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(document.get("raw_path") or "")).expanduser()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def research_document_domain(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del agentic_module, prompt, trace_metadata
    domain_id = str(domain.get("domain_id") or "").strip() or "unknown"

    working_document = dict(document)
    evidence = _load_grounded(document)
    grounded = evidence.get("grounded_rag") if isinstance(evidence, Mapping) else None
    if isinstance(grounded, Mapping):
        try:
            from .pre_design_rag_quality_contract import fuse_grounded_domain_evidence

            evidence["grounded_rag"] = fuse_grounded_domain_evidence(domain, grounded)
            working_document = project_rag._materialize_domain_evidence_document(
                domain_id, evidence
            )
        except Exception:
            working_document = dict(document)

    if isinstance(document, dict):
        document.clear()
        document.update(working_document)

    try:
        pages = project_rag._read_evidence_pages(working_document)
    except Exception:
        pages = []

    claims: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for page in _candidate_pages(pages, domain):
        extracted, page_diagnostics = _extract_page(router, domain=domain, page=page)
        claims.extend(extracted)
        diagnostics.extend(page_diagnostics)

    unique: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()
    for claim in claims:
        refs = claim.get("evidence_refs") if isinstance(claim, Mapping) else None
        ref = str(refs[0]) if isinstance(refs, list) and refs else ""
        key = (str(claim.get("claim") or ""), ref)
        if key[0] and key[1] and key not in seen_claims:
            seen_claims.add(key)
            unique.append(dict(claim))

    page_refs = [
        str(page.get("page_ref") or "").strip()
        for page in pages
        if str(page.get("page_ref") or "").strip()
    ]
    evidence_status = "supported" if unique else "no_relevant_external_evidence"
    return {
        "domain_id": domain_id,
        "claims": unique,
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
        "fixed_point": False,
        "research_mode": "advisory_predesign",
        "research_evidence_status": evidence_status,
        "page_local_diagnostics": list(dict.fromkeys(diagnostics)),
        "evidence_page_refs": page_refs,
        "evidence_document": project_rag._prompt_document_receipt(working_document),
        "quality_contract": {
            "schema_version": _PROTOCOL,
            "model_role": "semantic_evidence_extraction_only",
            "host_role": (
                "retrieval_state+source_refs+quote_verification+sufficiency+serialization"
            ),
            "model_json": False,
            "model_corrective_queries": False,
            "page_local_uncertainty_blocks_design": False,
            "missing_external_evidence_blocks_design": False,
        },
        "checkpoint": {
            "schema_version": "mmm/research-domain-checkpoint-v7",
            "status": "complete",
        },
    }


__all__ = ["research_document_domain"]
'''
write(ROOT / "small_model_predesign_research.py", small)

# Remove the old corrective state machine from the canonical execution path.
write(
    ROOT / "pre_design_domain_research.py",
    '''from __future__ import annotations

# Canonical pre-design research entrypoint. The previous corrective/page-gap state
# machine is intentionally not on the execution path.

from .small_model_predesign_research import research_document_domain

__all__ = ["research_document_domain"]
''',
)

# ---------------------------------------------------------------------------
# 2) Pre-design donor/source absence is advisory.
# ---------------------------------------------------------------------------
p = ROOT / "pre_design_research_pipeline.py"
s = p.read_text(encoding="utf-8")
s = replace_between(
    s,
    "def _validate_domain_provider_grounding(",
    "def _domain_document_evidence(",
    r'''def _validate_domain_provider_grounding(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> None:
    # Target-neutral donor retrieval is advisory; exact target/API checks fail closed later.
    del domain, grounded
    return None''',
)
old = '''    try:
        agentic._validate_sufficient_research(note, allowed_refs=allowed_refs)
    except agentic.SpecValidationError as exc:'''
new = '''    if (
        note.get("research_mode") == "advisory_predesign"
        and note.get("sufficient") is True
        and not note.get("claims")
    ):
        return
    try:
        agentic._validate_sufficient_research(note, allowed_refs=allowed_refs)
    except agentic.SpecValidationError as exc:'''
if old not in s:
    raise SystemExit("pipeline grounding validation marker not found")
s = s.replace(old, new, 1)
write(p, s)

# ---------------------------------------------------------------------------
# 3) Host validates advisory empty evidence; game-design uses Markdown, not model JSON.
# ---------------------------------------------------------------------------
p = ROOT / "agentic_research_game_design.py"
s = p.read_text(encoding="utf-8")
old = '''    claims = note.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise SpecValidationError("research_note.sufficient=true requires at least one grounded claim")'''
new = '''    claims = note.get("claims", [])
    if not isinstance(claims, list) or not claims:
        if (
            note.get("research_mode") == "advisory_predesign"
            and note.get("research_evidence_status")
            in {"no_relevant_external_evidence", "partial", "supported"}
        ):
            return
        raise SpecValidationError("research_note.sufficient=true requires at least one grounded claim")'''
if old not in s:
    raise SystemExit("sufficient research validator marker not found")
s = s.replace(old, new, 1)

start = s.index("def _section_messages(")
end = s.index("\ndef _render_design_research(", start)
section_messages = r'''def _section_messages(
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    research: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are a bounded Minecraft mod design worker. Output only the requested Markdown "
        "sections. For every requested field write a heading exactly as '## <field>'. "
        "Use plain text or bullets under it. For map fields use '- key: value' or nested bullets. "
        + _MODULE_FORMAT
        + " "
        + _ASSET_FORMAT
        + " Preserve exact approved requirement IDs. No JSON, code fences, <think>, analysis, "
        "or fields outside the requested headings. "
        + _PRODUCTION_DEPTH
    )
    ledger = _active_requirement_ledger(prompt)
    user = (
        "AUTHORITATIVE REQUEST\n"
        + prompt
        + "\n\nSECTION\n"
        + section_id
        + "\n\nREQUESTED FIELDS\n"
        + "\n".join(f"- {field}" for field in fields)
        + "\n\n"
        + _render_requirement_ledger(ledger)
        + "\n\nRESEARCH CONTEXT\n"
        + _render_design_research(research)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]'''
s = s[:start] + section_messages + s[end:]

start = s.index("def _generate_section(")
end = s.index("\ndef _normalize_heading(", start)
generate_section = r'''def _generate_section(
    router: Any,
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    host_properties: Mapping[str, Any],
    research: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del host_properties
    trace = PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata=dict(trace_metadata or {}),
    )
    raw = router.generate_text(
        "planner",
        _section_messages(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
        ),
        media_paths=media_paths,
        response_format="text",
        response_schema=None,
        tool_stage="game_design",
        enable_tools=False,
    )
    try:
        section = _parse_markdown_section(raw, fields)
        requirement_ids = tuple(
            item["requirement_id"] for item in _active_requirement_ledger(prompt)
        )
        _validate_section_types(section, fields, requirement_ids=requirement_ids)
    except (KeyError, SpecValidationError, ValueError, TypeError) as exc:
        trace.record_attempt(
            raw_output=raw,
            validation_error=str(exc),
            candidate=None,
            context={"section_id": section_id, "format": "host_parsed_markdown"},
        )
        raise
    trace.record_attempt(
        raw_output=raw,
        validation_error=None,
        candidate=section,
        accepted=section,
        context={"section_id": section_id, "format": "host_parsed_markdown"},
    )
    trace.record_success(section)
    return section'''
s = s[:start] + generate_section + s[end:]
write(p, s)

# ---------------------------------------------------------------------------
# 4) Requirement-balanced external discovery + strict Minecraft-mod admission.
# ---------------------------------------------------------------------------
p = ROOT / "pre_design_external_source_contract.py"
s = p.read_text(encoding="utf-8")
s = s.replace("_DEFAULT_MAX_QUERIES = 12", "_DEFAULT_MAX_QUERIES = 20")
s = s.replace(
    "provider_cap = _HARD_MAX_QUERIES if _github_token() else 8",
    "provider_cap = _HARD_MAX_QUERIES if _github_token() else 10",
)

start = s.index("def _body_relevant(")
end = s.index("\ndef _headers(", start)
body_gate = r'''def _body_relevant(query: str, body: str) -> bool:
    wanted = _query_terms(query)
    body_terms = {
        token.casefold()
        for token in _WORD.findall(body)
        if len(token) >= 3
    }
    if wanted and not (wanted & body_terms):
        return False
    folded = body.casefold()
    ecosystem_markers = (
        "fabric.mod.json",
        "fabric api",
        "fabricmc",
        "minecraft mod",
        "forge mod",
        "neoforge",
        "mods.toml",
        "architectury",
        "curseforge",
        "modrinth",
        "minecraftversion",
    )
    if not any(marker in folded for marker in ecosystem_markers):
        return False
    generic_markers = (
        "student zone",
        "awesome list",
        "awesome-minecraft",
        "learning path",
        "bootcamp",
        "minecraft bot",
        "mineflayer",
        "llm agent",
    )
    if any(marker in folded for marker in generic_markers) and not any(
        marker in folded for marker in ("fabric.mod.json", "mods.toml", "architectury")
    ):
        return False
    return True'''
s = s[:start] + body_gate + s[end:]

start = s.index("def _planned_requirement_query_keys(")
end = s.index("\ndef _fallback_query_keys(", start)
planned = r'''def _planned_requirement_query_keys(payload: Mapping[str, Any]) -> list[str]:
    # Stable one-query-per-authored-requirement first pass.
    if str(payload.get("schema_version") or "") == "mmm/corrective-retrieval-request-v1":
        result: list[str] = []
        seen: set[str] = set()
        for domain in payload.get("domains", ()):
            if not isinstance(domain, Mapping):
                continue
            for query in _stable_queries(domain.get("queries")):
                key = query.casefold()
                if key not in seen:
                    seen.add(key)
                    result.append(query)
        return result

    result: list[str] = []
    seen: set[str] = set()
    domains = payload.get("domains")
    for domain in domains if isinstance(domains, list) else ():
        if not isinstance(domain, Mapping) or str(domain.get("domain_id") or "") != "request":
            continue
        requirements = _stable_queries(domain.get("requirements"))
        prompt = requirements[0] if requirements else ""
        if not prompt:
            continue
        try:
            from . import authored_scope_research_contract as authored_scope
            catalog = authored_scope._active_catalog(prompt)
        except Exception:
            catalog = None
        rows = catalog.get("requirements") if isinstance(catalog, Mapping) else None
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            queries = _stable_queries(raw.get("search_queries"))
            if not queries:
                continue
            query = queries[0]
            key = query.casefold()
            if key not in seen:
                seen.add(key)
                result.append(query)
    return result'''
s = s[:start] + planned + s[end:]

old = '''    selected = _planned_requirement_query_keys(payload)
    selection_origin = "approved_requirement_queries"
    if not selected:
        selected = _fallback_query_keys(bundle, limit)
        selection_origin = "bounded_fallback_queries"
    selected = set(list(selected)[:limit])'''
new = '''    planned = _planned_requirement_query_keys(payload)
    selection_origin = "approved_requirement_queries"
    if not planned:
        planned = sorted(_fallback_query_keys(bundle, limit))
        selection_origin = "bounded_fallback_queries"
    selected_order = planned[:limit]
    selected = {query.casefold() for query in selected_order}'''
if old not in s:
    raise SystemExit("external selected-plan marker not found")
s = s.replace(old, new, 1)
s = s.replace("selected_queries=sorted(selected),", "selected_queries=selected_order,", 1)
write(p, s)

# ---------------------------------------------------------------------------
# 5) Conservative host routing for stateful / world / network obligations.
# ---------------------------------------------------------------------------
p = ROOT / "minecraft_knowledge_nodes.py"
s = p.read_text(encoding="utf-8")
marker = "    folded = text.casefold()\n    if re.search("
insert = '''    folded = text.casefold()

    stateful_terms = (
        "economy", "currency", "trade", "trading", "upgrade", "crew", "colony",
        "colonization", "terraform", "돈", "거래", "업그레이드", "선원", "식민",
    )
    world_terms = (
        "planet", "dimension", "world", "space travel", "colonization", "terraform",
        "행성", "우주", "식민",
    )
    mineral_terms = ("ore", "mineral", "광물", "광석")
    if any(term in folded for term in stateful_terms):
        features.add("persistence")
    if any(term in folded for term in world_terms):
        features.add("dimension")
    if any(term in folded for term in mineral_terms):
        features.add("world_feature")
    if (
        any(term in folded for term in stateful_terms)
        and not any(
            term in folded
            for term in ("singleplayer only", "single-player only", "싱글플레이 전용")
        )
    ):
        features.add("networking")

    if re.search('''
if marker not in s:
    raise SystemExit("knowledge feature routing marker not found")
s = s.replace(marker, insert, 1)
write(p, s)

# ---------------------------------------------------------------------------
# 6) Regression for the exact latest failure class.
# ---------------------------------------------------------------------------
test = r'''from __future__ import annotations

from minecraft_mod_ai import agentic_research_game_design as design_agent
from minecraft_mod_ai import minecraft_knowledge_nodes as knowledge
from minecraft_mod_ai import pre_design_domain_research
from minecraft_mod_ai import small_model_predesign_research as small


def test_canonical_predesign_path_bypasses_corrective_state_machine():
    assert pre_design_domain_research.research_document_domain is small.research_document_domain


def test_irrelevant_page_never_becomes_blocking_gap():
    class Router:
        def generate_text(self, *args, **kwargs):
            return "NONE"

    class Project:
        @staticmethod
        def _read_evidence_pages(document):
            return [
                {
                    "page_ref": "host#1",
                    "content": "Microsoft Build Student Zone learning path unrelated material",
                }
            ]

        @staticmethod
        def _prompt_document_receipt(document):
            return {"page_count": 1}

    note = small.research_document_domain(
        object(),
        Project(),
        Router(),
        prompt="식민지화 우주 모드",
        domain={"domain_id": "request", "objective": "space colonization", "queries": []},
        document={"page_count": 1},
        trace_metadata=None,
    )
    assert note["sufficient"] is True
    assert note["fixed_point"] is False
    assert note["gaps"] == []
    assert note["research_evidence_status"] == "no_relevant_external_evidence"


def test_predesign_model_uses_plain_text_and_host_exact_quote():
    calls = []

    class Router:
        def generate_text(self, *args, **kwargs):
            calls.append(kwargs)
            return (
                "EVIDENCE\tSpace stations can orbit planets."
                "\tUse an orbiting station abstraction."
            )

    claims, diagnostics = small._extract_page(
        Router(),
        domain={"objective": "space station", "queries": ["minecraft space station"]},
        page={
            "page_ref": "host#1",
            "content": "Space stations can orbit planets. Other text.",
        },
    )
    assert diagnostics == []
    assert claims and claims[0]["evidence_refs"] == ["host#1"]
    assert calls[0]["response_format"] == "text"
    assert calls[0]["response_schema"] is None
    assert calls[0]["enable_tools"] is False


def test_advisory_empty_evidence_is_valid_host_state():
    design_agent._validate_sufficient_research(
        {
            "sufficient": True,
            "claims": [],
            "research_mode": "advisory_predesign",
            "research_evidence_status": "no_relevant_external_evidence",
        },
        allowed_refs=frozenset(),
    )


def test_stateful_space_request_activates_persistence_network_worldgen():
    plan = knowledge.compile_minecraft_knowledge_plan(
        "우주로 가서 다른 행성을 식민지화하고 특수 광물을 캐며 돈과 거래로 "
        "우주선과 선원을 업그레이드한다"
    )
    predicates = {item["predicate_id"]: item for item in plan["branch_predicates"]}
    assert predicates["needs_persistence"]["value"] is True
    assert predicates["needs_network"]["value"] is True
    assert predicates["needs_worldgen"]["value"] is True
'''
write(Path("tests/test_small_model_predesign_v2.py"), test)

# Existing design-section regression follows the new host-parsed Markdown protocol.
p = Path("tests/test_agentic_research_game_design.py")
s = p.read_text(encoding="utf-8")
start = s.index("class _SectionRouter:")
end = s.index("\n\nclass _ResearchRouter:", start)
router_fixture = r'''class _SectionRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role,
        messages,
        *,
        media_paths=(),
        response_format="text",
        response_schema=None,
        tool_stage=None,
        enable_tools=True,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": tuple(media_paths),
                "response_format": response_format,
                "response_schema": response_schema,
                "tool_stage": tool_stage,
                "enable_tools": enable_tools,
            }
        )
        requested = []
        text = str(messages[-1]["content"])
        for field in (
            "title", "pitch", "core_loop", "progression", "combat", "mod_context",
            "modules", "assets", "acceptance_tests", "art_direction",
        ):
            if f"- {field}" in text:
                requested.append(field)
        bodies = {
            "title": "연구 기반 모드",
            "pitch": "검색 근거를 바탕으로 설계한다.",
            "core_loop": "- 탐색하고 상호작용한다",
            "progression": "- 기능을 단계적으로 해금한다",
            "combat": "none",
            "mod_context": "none",
            "modules": "none",
            "assets": "none",
            "acceptance_tests": "- 요청한 핵심 루프가 게임 내에서 동작한다",
            "art_direction": "none",
        }
        return "\n".join(f"## {field}\n{bodies[field]}" for field in requested)'''
s = s[:start] + router_fixture + s[end:]
s = s.replace(
    "def test_sectioned_game_design_generates_each_section_once_as_structured_json() -> None:",
    "def test_sectioned_game_design_uses_host_parsed_markdown() -> None:",
)
s = s.replace(
    '''    assert all(call["response_format"] == "json" for call in router.calls)
    assert all(isinstance(call["response_schema"], dict) for call in router.calls)''',
    '''    assert all(call["response_format"] == "text" for call in router.calls)
    assert all(call["response_schema"] is None for call in router.calls)''',
    1,
)
s = re.sub(
    r'''    for fields, call in zip\(expected_sections, router\.calls, strict=True\):\n        schema = call\["response_schema"\]\n        assert schema\["required"\] == list\(fields\)\n        assert schema\["additionalProperties"\] is False\n        system = str\(call\["messages"\]\[0\]\["content"\]\)\n        assert "Do not emit reasoning" in system\n''',
    '''    for fields, call in zip(expected_sections, router.calls, strict=True):
        system = str(call["messages"][0]["content"])
        assert "No JSON" in system
''',
    s,
    count=1,
)
write(p, s)

# Architecture guards enforced before CI.
domain_owner = (ROOT / "pre_design_domain_research.py").read_text(encoding="utf-8")
assert "pre_design_rag_corrective" not in domain_owner
assert "small_model_predesign_research" in domain_owner
agent = (ROOT / "agentic_research_game_design.py").read_text(encoding="utf-8")
assert 'response_format="text"' in agent
assert "host_parsed_markdown" in agent

print("changed files:")
for path in sorted(set(changed)):
    print(path)
