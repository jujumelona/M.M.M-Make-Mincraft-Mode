from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
_TARGET_KEYWORDS = {"minecraft_version", "loader", "mappings"}
_STRICT_CALL_NAMES = {"retrieve_official_evidence", "retrieve_target_agentic_evidence"}


def _python_sources() -> list[Path]:
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py") if path.is_file())


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keyword_names(node: ast.Call) -> set[str]:
    return {keyword.arg for keyword in node.keywords if keyword.arg is not None}


def test_all_direct_official_rag_calls_are_target_bound() -> None:
    """No direct strict/live Official RAG call may omit the immutable target triple."""
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            if call_name not in _STRICT_CALL_NAMES:
                continue
            missing = sorted(_TARGET_KEYWORDS - _keyword_names(node))
            if missing:
                relative = path.relative_to(PACKAGE_ROOT.parent).as_posix()
                violations.append(
                    f"{relative}:{node.lineno}: {call_name} missing {', '.join(missing)}"
                )
    assert not violations, "Targetless strict RAG submission(s):\n" + "\n".join(violations)


def test_public_central_research_routes_through_target_guarded_parallel_boundary() -> None:
    """The public central entrypoint must keep the pre-worker target applicability gate."""
    central_path = PACKAGE_ROOT / "central_research.py"
    parallel_path = PACKAGE_ROOT / "parallel_runtime_contract.py"
    central = central_path.read_text(encoding="utf-8")
    parallel = parallel_path.read_text(encoding="utf-8")

    assert "from .parallel_runtime_contract import retrieve_domain_evidence as parallel_retrieve" in central
    assert "return parallel_retrieve(research_brief, retrieve=retrieve)" in central
    assert 'if research_brief.get("_mmm_platform_target") is None:' in parallel
    guard = parallel.index('if research_brief.get("_mmm_platform_target") is None:')
    pool = parallel.index("ThreadPoolExecutor(", guard)
    assert guard < pool


def test_targetless_central_graph_defers_only_official_lane() -> None:
    """Target absence is an Official-RAG applicability decision, not a production abort."""
    central_path = PACKAGE_ROOT / "central_research.py"
    source = central_path.read_text(encoding="utf-8")
    assert '"strategy": "routed_to_other_providers"' in source
    assert '"strategy": "deferred_until_platform_selected"' in source
    assert '"deferred_official_domains": deferred' in source
