from __future__ import annotations

from functools import wraps
from types import SimpleNamespace

from minecraft_mod_ai import retrieval_cpu_budget_contract as policy


class _Explorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def explore(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.calls[-1]


def _wrapped_search_chain():
    def lexical(index_path, query):
        return {"mode": "lexical", "index_path": index_path, "query": query}

    @wraps(lexical)
    def hybrid(index_path, query):
        return {"mode": "dense", "index_path": index_path, "query": query}

    hybrid._mmm_small_model_hybrid_code_rag = True  # type: ignore[attr-defined]

    @wraps(hybrid)
    def demand_driven(index_path, query):
        return hybrid(index_path, query)

    demand_driven._mmm_demand_driven_dense_pre_design = True  # type: ignore[attr-defined]
    return lexical, demand_driven


def test_repository_grounding_never_implicitly_loads_dense_models() -> None:
    explorer = _Explorer()

    result = policy._lexical_repository_exploration(
        explorer,
        "register custom block",
        diagnostics=("src/main/java/example/Mod.java",),
        line_budget=96,
        degraded=[],
        lane="task",
    )

    assert result["semantic"] is False
    assert result["rerank"] is False
    assert result["diagnostic_paths"] == ("src/main/java/example/Mod.java",)


def test_pre_design_dense_wrappers_unwrap_to_lexical_owner() -> None:
    lexical, current = _wrapped_search_chain()

    assert policy._lexical_pre_design_owner(current) is lexical


def test_install_defaults_to_cheap_retrieval(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    lexical, current = _wrapped_search_chain()
    repository_grounding = SimpleNamespace(_explore_with_degraded_fallback=object())
    pre_design = SimpleNamespace(_search_code_index=current)

    policy.install(repository_grounding, pre_design)

    assert repository_grounding._explore_with_degraded_fallback is policy._lexical_repository_exploration
    assert pre_design._search_code_index is lexical


def test_explicit_dense_opt_in_preserves_existing_paths(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    _lexical, current = _wrapped_search_chain()
    grounding_owner = object()
    repository_grounding = SimpleNamespace(_explore_with_degraded_fallback=grounding_owner)
    pre_design = SimpleNamespace(_search_code_index=current)

    policy.install(repository_grounding, pre_design)

    assert repository_grounding._explore_with_degraded_fallback is grounding_owner
    assert pre_design._search_code_index is current
