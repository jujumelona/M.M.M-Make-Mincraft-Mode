from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITICAL_RUNTIME_PATHS = (
    "minecraft_mod_ai/llama_*.py",
    "minecraft_mod_ai/qwen*.py",
    "minecraft_mod_ai/generation_output_budget.py",
    "minecraft_mod_ai/model_context_budget.py",
    "minecraft_mod_ai/model_output_atomicity_contract.py",
    "minecraft_mod_ai/acceptance_contracts.py",
    "minecraft_mod_ai/quality_evidence.py",
    "minecraft_mod_ai/evidence_first_planning.py",
    "minecraft_mod_ai/research_rag_performance.py",
    "minecraft_mod_ai/source_observation_context.py",
    "minecraft_mod_ai/generator.py",
    "minecraft_mod_ai/texture_equivalence_cache.py",
    "minecraft_mod_ai/extended_registration_contract.py",
)
DEEP_SOFTWARE_GATE_PATHS = (
    "tools/verify_integrity_minecraft.py",
    "tests/test_implementation_ir.py",
    "minecraft_mod_ai/performance_final_tuning.py",
    "minecraft_mod_ai/performance_final_contract.py",
    "minecraft_mod_ai/work_graph.py",
    "minecraft_mod_ai/work_graph_receipt_read.py",
    "minecraft_mod_ai/work_graph_state_transition_contract.py",
    "minecraft_mod_ai/production_contract.py",
    "minecraft_mod_ai/resource_contracts.py",
    "minecraft_mod_ai/model_registry.py",
)
PUSH_GATES = (
    ".github/workflows/main-ci.yml",
    ".github/workflows/production-kill-gates.yml",
    ".github/workflows/cumulative-regression-guard.yml",
    ".github/workflows/real-model-colab-e2e.yml",
)
SOFTWARE_GATES = PUSH_GATES[:3]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_critical_runtime_changes_reach_end_to_end_gates() -> None:
    for workflow in PUSH_GATES:
        text = _text(workflow)
        assert "push:" in text, workflow
        assert "branches: [main]" in text, workflow
        for path in CRITICAL_RUNTIME_PATHS:
            assert path in text, f"{workflow} missing {path}"


def test_integrity_and_ir_changes_reenter_deep_software_gates() -> None:
    for workflow in SOFTWARE_GATES:
        text = _text(workflow)
        for path in DEEP_SOFTWARE_GATE_PATHS:
            assert path in text, f"{workflow} missing {path}"


def test_failed_runtime_gates_feed_full_debug_owner() -> None:
    text = _text(".github/workflows/full-debug-gate.yml")
    for upstream in (
        "CI",
        "Partial CI",
        "Cumulative Regression Guard",
        "Production Kill Gates",
        "Runtime Generation Regression",
        "Deterministic Debug Full E2E",
        "Production Path Contract",
        "Candidate Verifier Targeted",
        "Real Model Colab E2E",
    ):
        assert upstream in text, upstream
