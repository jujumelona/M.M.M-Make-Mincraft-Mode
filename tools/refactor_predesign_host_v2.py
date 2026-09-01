from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing expected block in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: str, start: str, end: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    first = text.find(start)
    if first < 0:
        raise RuntimeError(f"missing start marker in {path}: {start!r}")
    last = text.find(end, first)
    if last < 0:
        raise RuntimeError(f"missing end marker in {path}: {end!r}")
    target.write_text(text[:first] + new + text[last:], encoding="utf-8")


def patch_external_source_contract() -> None:
    path = "minecraft_mod_ai/pre_design_external_source_contract.py"
    replace_once(
        path,
        '''    # GitHub unauthenticated repository search is only 10 requests/minute.  Never
    # configure the pre-design phase to deterministically exceed that ceiling itself;
    # leave two requests of headroom for other process activity.
    provider_cap = _HARD_MAX_QUERIES if _github_token() else 8
    return max(1, min(value, _HARD_MAX_QUERIES, provider_cap))
''',
        '''    # This value is a fallback batch budget only. Approved requirement coverage
    # is never truncated to fit a provider request count. Provider rate limits are
    # observed dynamically and recorded as provider state instead.
    return max(1, min(value, _HARD_MAX_QUERIES))
''',
    )
    replace_once(
        path,
        '''    selected_order = planned[:limit]
    selected = {query.casefold() for query in selected_order}
''',
        '''    # The normal authored plan contributes one first-pass query per requirement.
    # Never silently starve later requirements because an unrelated global cap was hit.
    selected_order = list(dict.fromkeys(planned))
    selected = {query.casefold() for query in selected_order}
''',
    )
    replace_once(
        path,
        '''def _fallback_query_keys(bundle: Mapping[str, Any], limit: int) -> set[str]:
    all_queries: list[str] = []
    for domain in bundle.get("domains", ()) if isinstance(bundle.get("domains"), list) else ():
        if not isinstance(domain, Mapping):
            continue
        for row in domain.get("queries", ()) if isinstance(domain.get("queries"), list) else ():
            if isinstance(row, Mapping):
                query = _clean_query(row.get("query"))
                if query:
                    all_queries.append(query)
    if len(all_queries) <= limit:
        return {query.casefold() for query in all_queries}
    stride = max(1, math.floor(len(all_queries) / limit))
    chosen = all_queries[::stride][:limit]
    return {query.casefold() for query in chosen}
''',
        '''def _fallback_query_keys(bundle: Mapping[str, Any], limit: int) -> set[str]:
    del limit
    all_queries: list[str] = []
    for domain in bundle.get("domains", ()) if isinstance(bundle.get("domains"), list) else ():
        if not isinstance(domain, Mapping):
            continue
        for row in domain.get("queries", ()) if isinstance(domain.get("queries"), list) else ():
            if isinstance(row, Mapping):
                query = _clean_query(row.get("query"))
                if query:
                    all_queries.append(query)
    return {query.casefold() for query in all_queries}
''',
    )
    replace_once(path, "import math\n", "")
    replace_between(
        path,
        "def _body_relevant(query: str, body: str) -> bool:\n",
        "def _headers() -> dict[str, str]:\n",
        '''_GENERIC_QUERY_TERMS = frozenset({
    "build", "building", "create", "custom", "craft", "crafting", "make",
    "minecraft", "fabric", "forge", "neoforge", "mod", "mods", "mode",
    "source", "implementation", "system", "feature", "space", "game",
})
_GENERIC_REPOSITORY_MARKERS = (
    "studentsatbuild", "student zone", "awesome-minecraft", "awesome minecraft",
    "stockmarket", "stock market", "mindcraft-bots", "minecraft bot", "mineflayer",
    "llm agent", "learning path", "bootcamp", "tutorial collection", "games list",
)


def _specific_query_terms(query: str) -> set[str]:
    return {term for term in _query_terms(query) if term not in _GENERIC_QUERY_TERMS}


def _term_overlap(wanted: set[str], available: set[str]) -> bool:
    for left in wanted:
        for right in available:
            if left == right:
                return True
            if min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)):
                return True
    return False


def _repository_candidate_relevant(query: str, repository: Mapping[str, Any]) -> bool:
    full_name = str(repository.get("full_name") or "").strip()
    description = str(repository.get("description") or "").strip()
    topics = repository.get("topics")
    topic_text = " ".join(str(item) for item in topics) if isinstance(topics, list) else ""
    folded = " ".join((full_name, description, topic_text)).casefold()
    if not folded or any(marker in folded for marker in _GENERIC_REPOSITORY_MARKERS):
        return False
    if "minecraft" not in folded:
        return False
    terms = {token.casefold() for token in _WORD.findall(folded) if len(token) >= 3}
    specific = _specific_query_terms(query)
    if specific and not _term_overlap(specific, terms):
        return False
    return True


def _body_relevant(query: str, body: str) -> bool:
    wanted = _query_terms(query)
    body_terms = {token.casefold() for token in _WORD.findall(body) if len(token) >= 3}
    specific = {term for term in wanted if term not in _GENERIC_QUERY_TERMS}
    if specific and not _term_overlap(specific, body_terms):
        return False
    if not specific and wanted and not _term_overlap(wanted, body_terms):
        return False
    folded = body.casefold()
    ecosystem_markers = (
        "fabric.mod.json", "fabric api", "fabricmc", "minecraft mod",
        "mod for minecraft", "forge mod", "neoforge", "mods.toml",
        "architectury", "curseforge", "modrinth", "minecraftversion",
    )
    if not any(marker in folded for marker in ecosystem_markers):
        return False
    if any(marker in folded for marker in _GENERIC_REPOSITORY_MARKERS) and not any(
        marker in folded for marker in ("fabric.mod.json", "mods.toml", "architectury")
    ):
        return False
    return True


''',
    )
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    needle = '''                if not full_name or "/" not in full_name:
                    _emit_source_trace(
                        "github_repository_skipped",
                        query=query,
                        candidate_index=candidate_index,
                        reason="invalid_full_name",
                    )
                    continue
                _emit_source_trace(
'''
    replacement = '''                if not full_name or "/" not in full_name:
                    _emit_source_trace(
                        "github_repository_skipped",
                        query=query,
                        candidate_index=candidate_index,
                        reason="invalid_full_name",
                    )
                    continue
                if not _repository_candidate_relevant(query, repository):
                    _emit_source_trace(
                        "github_repository_skipped",
                        query=query,
                        candidate_index=candidate_index,
                        repository=full_name,
                        reason="repository_not_minecraft_mod_query_relevant",
                    )
                    continue
                _emit_source_trace(
'''
    if needle not in text:
        raise RuntimeError("missing repository admission injection point")
    target.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def patch_corrective_queries() -> None:
    replace_between(
        "minecraft_mod_ai/pre_design_rag_corrective.py",
        "def _generate_gap_queries(\n",
        "def _read_and_verify_document(\n",
        '''def _generate_gap_queries(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    domain: Mapping[str, Any],
    gaps: Sequence[str],
    prior_queries: Sequence[str],
    seen: set[str],
    raw_prompt: str,
    progress_label: str,
) -> list[str]:
    """Consume unsearched host-approved queries; never ask the model to author JSON."""
    del agentic_module, project_rag, router, gaps, prior_queries
    candidates: list[str] = []
    raw_queries = domain.get("queries")
    if isinstance(raw_queries, Sequence) and not isinstance(raw_queries, (str, bytes, bytearray)):
        for raw in raw_queries:
            if isinstance(raw, Mapping):
                direct = str(raw.get("query") or "").strip()
                if direct:
                    candidates.append(direct)
                nested = raw.get("search_queries")
                if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                    candidates.extend(str(item).strip() for item in nested if str(item).strip())
            elif str(raw).strip():
                candidates.append(str(raw).strip())
    result = _correction_queries(candidates, seen=seen, raw_prompt=raw_prompt)
    _emit_corrective_trace(
        "host_corrective_query_plan",
        domain_id=str(domain.get("domain_id") or ""),
        progress_label=progress_label,
        candidate_count=len(candidates),
        selected_queries=result,
        model_called=False,
    )
    return result


''',
    )


def patch_authored_query_plan() -> None:
    replace_between(
        "minecraft_mod_ai/authored_scope_research_contract.py",
        "def _call_retrieval_planner(\n",
        "def _validate_dependency_dag(\n",
        '''def _call_retrieval_planner(
    router: Any,
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
) -> Any:
    """Build query structure deterministically; the small model owns no JSON protocol."""
    del router, prompt
    from .canonical_capability_ontology import search_queries_for_capability

    rows: list[dict[str, Any]] = []
    for item in requirements:
        rid = str(item.get("requirement_id") or "").strip()
        capability = str(item.get("capability") or "").strip()
        if not rid:
            continue
        raw_deps = item.get("depends_on")
        deps = (
            [str(dep).strip() for dep in raw_deps if str(dep).strip() and str(dep).strip() != rid]
            if isinstance(raw_deps, list)
            else []
        )
        queries = list(search_queries_for_capability(capability)) if capability else []
        concept = re.sub(r"[^A-Za-z0-9]+", " ", capability.replace("_", " ")).strip()
        if not concept:
            semantic = str(item.get("semantic_statement") or "")
            concept = " ".join(_QUERY_WORD.findall(semantic))[:120].strip()
        if not concept:
            concept = "requested minecraft mechanic"
        queries.extend((
            f"minecraft mod {concept} implementation",
            f"minecraft fabric {concept} source",
        ))
        cleaned: list[str] = []
        for query in queries:
            value = _query_text(query)
            if _is_english_retrieval_query(value) and value.casefold() not in {q.casefold() for q in cleaned}:
                cleaned.append(value)
            if len(cleaned) >= 5:
                break
        if len(cleaned) < 2:
            raise ValueError(f"host retrieval planner could not build two queries for {rid}")
        rows.append({
            "requirement_id": rid,
            "depends_on": list(dict.fromkeys(deps)),
            "search_queries": cleaned,
        })
    return {"requirements": rows}


''',
    )


def patch_research_facades() -> None:
    replace_between(
        "minecraft_mod_ai/agentic_research_game_design.py",
        "def _research_domain_with_agent(\n",
        "def generate_sectioned_game_design(\n",
        '''def _research_domain_with_agent(
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compatibility facade: host receipts only, never model-authored research JSON."""
    del router, prompt, trace_metadata
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    evidence = _domain_evidence_slice(domain_id, deterministic)
    return {
        "domain_id": domain_id,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
        "fixed_point": False,
        "research_mode": "advisory_predesign",
        "research_evidence_status": (
            "host_receipts_available" if _allowed_research_refs(evidence) else "no_relevant_external_evidence"
        ),
        "quality_contract": {
            "model_role": "none_for_receipt_sufficiency",
            "host_role": "scope+retrieval+evidence_refs+sufficiency+serialization",
            "model_json": False,
        },
    }


''',
    )
    replace_between(
        "minecraft_mod_ai/agentic_pre_design_rag.py",
        "def _research_document_domain(\n",
        "def _materialize_domain_evidence_document(\n",
        '''def _research_document_domain(
    agentic_module: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Legacy API facade delegated to the canonical host-owned research owner."""
    import sys
    from .small_model_predesign_research import research_document_domain

    return research_document_domain(
        agentic_module,
        sys.modules[__name__],
        router,
        prompt=prompt,
        domain=domain,
        document=document,
        trace_metadata=trace_metadata,
    )


''',
    )


def patch_existing_test() -> None:
    path = "tests/test_agentic_research_game_design.py"
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    start = text.find("def test_research_domain_accepts_only_host_issued_grounding_ref(monkeypatch) -> None:\n")
    end = text.find("def test_sufficient_research_rejects_empty_and_invented_refs() -> None:\n", start)
    if start < 0 or end < 0:
        raise RuntimeError("missing legacy research-domain test block")
    replacement = '''def test_research_domain_legacy_facade_is_host_owned() -> None:
    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model called: {name}")

    result = agentic._research_domain_with_agent(
        NeverModel(),
        prompt="기능을 조사해서 설계해줘",
        domain={"domain_id": "request", "objective": "요청 조사", "queries": ["minecraft mod feature"]},
        deterministic=_deterministic_research(),
        trace_metadata=None,
    )
    assert result["sufficient"] is True
    assert result["research_mode"] == "advisory_predesign"
    assert result["quality_contract"]["model_json"] is False


'''
    target.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def write_regressions() -> None:
    Path("tests/test_final_host_owned_predesign.py").write_text(
        '''from __future__ import annotations


def test_unauthenticated_external_query_budget_is_not_eight(monkeypatch):
    from minecraft_mod_ai import pre_design_external_source_contract as external
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("MMM_PREDESIGN_EXTERNAL_SOURCE_QUERIES", raising=False)
    assert external._max_queries() == 20


def test_all_approved_requirement_queries_are_attempted(monkeypatch):
    from minecraft_mod_ai import pre_design_external_source_contract as external
    calls = []
    monkeypatch.setattr(external, "_retrieve_github_source_body", lambda query: calls.append(query) or {
        "records": [], "search_requests": 1, "source_requests": 0,
        "provider_status": "available", "saturation_reason": "test", "errors": []
    })
    queries = [f"minecraft mod requirement {i} source" for i in range(10)]
    payload = {"schema_version": "mmm/corrective-retrieval-request-v1", "domains": [{"domain_id": "request", "queries": queries}]}
    bundle = {"domains": [{"domain_id": "request", "queries": [{"query": q, "external_rag": {"records": []}} for q in queries]}]}
    external._augment_bundle(payload, bundle)
    assert calls == queries


def test_generic_github_repository_is_rejected_before_body():
    from minecraft_mod_ai import pre_design_external_source_contract as external
    assert not external._repository_candidate_relevant(
        "minecraft mod build space station modules",
        {"full_name": "microsoft/StudentsAtBuild", "description": "Minecraft student learning path", "topics": []},
    )
    assert external._repository_candidate_relevant(
        "minecraft mod space rocket vehicle",
        {"full_name": "Advanced-Rocketry/AdvancedRocketry", "description": "Advanced Rocketry Minecraft space mod", "topics": ["minecraft", "mod"]},
    )
    assert not external._body_relevant(
        "minecraft mod build space station modules",
        "Student Zone learning path for Minecraft mod build tutorials.",
    )
    assert external._body_relevant(
        "minecraft mod space rocket vehicle",
        "Advanced Rocketry is a Minecraft mod for building rocket vehicles and travelling through space.",
    )


def test_corrective_queries_are_host_owned_and_do_not_call_model():
    from minecraft_mod_ai import pre_design_rag_corrective as corrective
    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model/project helper called: {name}")
    seen = {"minecraft alien mob entity mod"}
    result = corrective._generate_gap_queries(
        NeverModel(), NeverModel(), NeverModel(),
        domain={"domain_id": "request", "queries": ["minecraft alien mob entity mod", "minecraft colony settlement building mod"]},
        gaps=["colonization evidence missing"], prior_queries=["minecraft alien mob entity mod"],
        seen=seen, raw_prompt="우주 식민지화 모드", progress_label="test",
    )
    assert result == ["minecraft colony settlement building mod"]


def test_requirement_retrieval_planner_is_host_owned():
    from minecraft_mod_ai import authored_scope_research_contract as authored
    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model called: {name}")
    result = authored._call_retrieval_planner(
        NeverModel(), "식민지화",
        [{"requirement_id": "req-colony", "capability": "planet_colonization", "depends_on": []}],
    )
    row = result["requirements"][0]
    assert row["requirement_id"] == "req-colony"
    assert len(row["search_queries"]) >= 2
    assert all("minecraft" in query.casefold() for query in row["search_queries"])


def test_legacy_domain_agent_is_host_advisory_no_model_call():
    from minecraft_mod_ai import agentic_research_game_design as agentic
    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model called: {name}")
    note = agentic._research_domain_with_agent(
        NeverModel(), prompt="x", domain={"domain_id": "request"}, deterministic={}, trace_metadata=None
    )
    assert note["sufficient"] is True
    assert note["quality_contract"]["model_json"] is False
''',
        encoding="utf-8",
    )


def main() -> None:
    patch_external_source_contract()
    patch_corrective_queries()
    patch_authored_query_plan()
    patch_research_facades()
    patch_existing_test()
    write_regressions()


if __name__ == "__main__":
    main()
