from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}: {old!r}")
    file_path.write_text(text.replace(old, new), encoding="utf-8")


def replace_between(path: str, start: str, end: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    start_index = text.find(start)
    if start_index < 0:
        raise SystemExit(f"{path}: start marker not found: {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise SystemExit(f"{path}: end marker not found: {end!r}")
    file_path.write_text(text[:start_index] + new + text[end_index:], encoding="utf-8")


replace_between(
    "minecraft_mod_ai/authored_production.py",
    "def _require_bound_target(",
    "\n\ndef compile_authored_design(",
    '''def _bound_target(design: Mapping[str, Any]) -> dict[str, str]:
    """Return the complete host-selected target, or no target when none exists.

    Saved authored production may be target-agnostic. Complete absence must not abort
    production; Official RAG is skipped later before a worker is created. A partial
    target is never accepted because it would make the host contract ambiguous.
    """
    candidates: list[Mapping[str, Any]] = [design]
    for key in ("platform", "target", "toolchain", "build", "existing_project"):
        value = design.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)

    selection = design.get("_platform_selection")
    if isinstance(selection, Mapping):
        candidates.append(selection)
        selected_target = selection.get("target")
        if isinstance(selected_target, Mapping):
            candidates.append(selected_target)

    saw_partial = False
    for candidate in candidates:
        present = [candidate.get(key) not in (None, "") for key in _TARGET_KEYS]
        if all(present):
            return {key: str(candidate[key]) for key in _TARGET_KEYS}
        saw_partial = saw_partial or any(present)

    if saw_partial:
        raise ValueError(
            "Saved authored production received an incomplete platform target; "
            "minecraft_version, loader and mappings must be provided together."
        )
    return {}
''',
)
replace_once(
    "minecraft_mod_ai/authored_production.py",
    "    target = _require_bound_target(design)\n",
    "    target = _bound_target(design)\n",
)

replace_once(
    "minecraft_mod_ai/central_research.py",
    '''    selection = game_design.get("_platform_selection")
    if isinstance(selection, Mapping):
        target = selection.get("target")
        if target is not None:
            payload["_mmm_platform_target"] = _canonical_platform_target(target)
''',
    '''    selection = game_design.get("_platform_selection")
    target = selection.get("target") if isinstance(selection, Mapping) else None
    if target is None:
        direct_target = {
            "minecraft_version": game_design.get("minecraft_version"),
            "loader": game_design.get("loader"),
            "mappings": game_design.get("mappings"),
        }
        if any(value not in (None, "") for value in direct_target.values()):
            target = direct_target
    if target is not None:
        payload["_mmm_platform_target"] = _canonical_platform_target(target)
''',
)
replace_once(
    "minecraft_mod_ai/central_research.py",
    '''    target = _canonical_platform_target(research_brief.get("_mmm_platform_target"))
    adapter = adapter_for_target(target["minecraft_version"], target["loader"])

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
''',
    '''    raw_target = research_brief.get("_mmm_platform_target")
    target = _canonical_platform_target(raw_target) if raw_target is not None else None
    adapter = (
        adapter_for_target(target["minecraft_version"], target["loader"])
        if target is not None
        else None
    )

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    deferred: list[str] = []
''',
)
replace_once(
    "minecraft_mod_ai/central_research.py",
    '''            )
            continue

        query_results: list[dict[str, Any]] = []
''',
    '''            )
            continue
        if adapter is None:
            deferred.append(domain.domain_id)
            results.append(
                {
                    "domain_id": domain.domain_id,
                    "strategy": "deferred_until_platform_selected",
                    "queries": [],
                }
            )
            continue

        query_results: list[dict[str, Any]] = []
''',
)
replace_once(
    "minecraft_mod_ai/central_research.py",
    '        "deferred_official_domains": [],\n',
    '        "deferred_official_domains": deferred,\n',
)

replace_once(
    "minecraft_mod_ai/parallel_runtime_contract.py",
    '''        selected_retrieve = retrieve or original_default_retrieve
        adapter, domains = _require_parallel_research_contract(
            central_module,
            research_brief,
        )
        raw_domains = research_brief["domains"]

        official_domains = [
            domain for domain in domains if "official_docs" in domain.providers
        ]
        if not official_domains:
            return build_research_graph(
                research_brief,
                retrieve=selected_retrieve,
            )

        query_criteria, domain_queries, domain_criteria = _coverage_query_plan(
''',
    '''        selected_retrieve = retrieve or original_default_retrieve
        raw_domains = research_brief.get("domains")
        if not isinstance(raw_domains, list) or not raw_domains:
            raise ParallelResearchContractError(
                "official-doc research requires at least one research domain"
            )

        domains: list[Any] = []
        for index, raw_domain in enumerate(raw_domains):
            try:
                domain = central_module._research_domain(raw_domain)
            except Exception as exc:  # noqa: BLE001 - contract boundary
                raise ParallelResearchContractError(
                    f"invalid research domain at index {index}"
                ) from exc
            domains.append(domain)

        official_domains = [
            domain for domain in domains if "official_docs" in domain.providers
        ]
        if not official_domains:
            return build_research_graph(
                research_brief,
                retrieve=selected_retrieve,
            )

        # No target means Official RAG is inapplicable, not that the whole production
        # request is invalid. Exit through the canonical graph before pool/index/prefetch
        # creation. A present but malformed target still enters the strict contract.
        if research_brief.get("_mmm_platform_target") is None:
            return build_research_graph(
                research_brief,
                retrieve=selected_retrieve,
            )

        adapter, verified_domains = _require_parallel_research_contract(
            central_module,
            research_brief,
        )
        domains = verified_domains

        query_criteria, domain_queries, domain_criteria = _coverage_query_plan(
''',
)

replace_between(
    "tests/test_central_research.py",
    "def test_targetless_official_research_fails_before_retrieval() -> None:",
    "\ndef test_retrieve_domain_evidence_covers_authored_and_declared_criteria_without_speculative_corrections() -> None:",
    '''def test_targetless_official_research_skips_worker_without_retrieval() -> None:
    brief = normalize_research_brief('Research all routed facts.', {}, _candidate([_domain('official_one', providers=['official_docs'])]))
    calls: list[str] = []

    def fake_retrieve(query: str, **_kwargs: object) -> RetrievalReceipt:
        calls.append(query)
        return _receipt(query)

    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)
    assert calls == []
    assert evidence['target'] is None
    assert evidence['deferred_official_domains'] == ['official_one']
    assert evidence['unresolved_official_domains'] == []
    assert evidence['domains'][0]['strategy'] == 'deferred_until_platform_selected'


def test_targetless_mixed_research_keeps_generic_route() -> None:
    brief = normalize_research_brief(
        'Research all routed facts.',
        {},
        _candidate([
            _domain('official_one', providers=['official_docs']),
            _domain('generic_one', providers=['github']),
        ]),
    )

    def must_not_retrieve(*_args: object, **_kwargs: object) -> RetrievalReceipt:
        raise AssertionError('Official retrieval must not run without a target')

    evidence = retrieve_domain_evidence(brief, retrieve=must_not_retrieve)
    assert evidence['deferred_official_domains'] == ['official_one']
    assert evidence['domains'][0]['strategy'] == 'deferred_until_platform_selected'
    assert evidence['domains'][1]['strategy'] == 'routed_to_other_providers'


def test_partial_parallel_target_still_fails_closed() -> None:
    brief = normalize_research_brief('Research all routed facts.', {}, _candidate([_domain('official_one', providers=['official_docs'])]))
    brief['_mmm_platform_target'] = {'minecraft_version': '1.20.1', 'loader': 'fabric'}
    with pytest.raises(
        ParallelResearchContractError,
        match='minecraft_version, loader, and mappings',
    ):
        retrieve_domain_evidence(
            brief,
            retrieve=lambda *_args, **_kwargs: _receipt('unused'),
        )


def test_saved_design_top_level_target_reaches_central_rag() -> None:
    adapter = adapter_for_target('1.20.1', 'fabric')
    design = {
        'minecraft_version': adapter.minecraft_version,
        'loader': adapter.loader,
        'mappings': adapter.yarn_mappings,
    }
    brief = normalize_research_brief(
        'Research exact target evidence.',
        design,
        _candidate([_domain('official_one', providers=['official_docs'])]),
    )
    assert brief['_mmm_platform_target'] == design

''',
)

Path("tests/test_authored_production_optional_target.py").write_text(
    '''from __future__ import annotations

import pytest

from minecraft_mod_ai.authored_production import _bound_target


def test_bound_target_allows_completely_absent_target() -> None:
    assert _bound_target({"authored_plan": {"text": "saved"}}) == {}


def test_bound_target_preserves_complete_top_level_target() -> None:
    target = {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }
    assert _bound_target(target) == target


def test_bound_target_reads_platform_selection_target() -> None:
    target = {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }
    assert _bound_target({"_platform_selection": {"target": target}}) == target


def test_bound_target_rejects_partial_target() -> None:
    with pytest.raises(ValueError, match="incomplete platform target"):
        _bound_target(
            {"target": {"minecraft_version": "1.21.1", "loader": "fabric"}}
        )
''',
    encoding="utf-8",
)
