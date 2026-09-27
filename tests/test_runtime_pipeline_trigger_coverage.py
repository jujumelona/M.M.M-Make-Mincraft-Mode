from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNTIME_PATHS = (
    "minecraft_mod_ai/llama_*.py",
    "minecraft_mod_ai/qwen*.py",
    "minecraft_mod_ai/generation_output_budget.py",
    "minecraft_mod_ai/model_context_budget.py",
    "minecraft_mod_ai/model_output_atomicity_contract.py",
)
SOURCE_OWNED_RUNTIME_PATHS = (
    "minecraft_mod_ai/acceptance_contracts.py",
    "minecraft_mod_ai/quality_evidence.py",
    "minecraft_mod_ai/evidence_first_planning.py",
    "minecraft_mod_ai/research_rag_performance.py",
    "minecraft_mod_ai/source_observation_context.py",
    "minecraft_mod_ai/generator.py",
    "minecraft_mod_ai/texture_equivalence_cache.py",
    "minecraft_mod_ai/extended_registration_contract.py",
)
AUTHORED_PIPELINE_GATE_PATHS = (
    "minecraft_mod_ai/authored_document_contract.py",
    "minecraft_mod_ai/authored_section_ids.py",
    "minecraft_mod_ai/authored_ir_parser.py",
    "minecraft_mod_ai/authored_execution_schema.py",
    "minecraft_mod_ai/complete_planner.py",
    "minecraft_mod_ai/authored_production.py",
    "minecraft_mod_ai/planning_detail_template.py",
    "minecraft_mod_ai/planning_detail_applicability.py",
    "minecraft_mod_ai/implementation_ir.py",
    "minecraft_mod_ai/implementation_decisions.py",
    "minecraft_mod_ai/implementation_graph_execution.py",
    "minecraft_mod_ai/task_template_catalog.py",
)
DEEP_SOFTWARE_GATE_PATHS = (
    *SOURCE_OWNED_RUNTIME_PATHS,
    *AUTHORED_PIPELINE_GATE_PATHS,
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


def test_model_runtime_changes_reach_real_model_and_software_gates() -> None:
    for workflow in PUSH_GATES:
        text = _text(workflow)
        assert "push:" in text, workflow
        assert "branches: [main]" in text, workflow
        for path in MODEL_RUNTIME_PATHS:
            assert path in text, f"{workflow} missing {path}"


def test_source_owned_and_integrity_changes_reenter_deep_software_gates() -> None:
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
