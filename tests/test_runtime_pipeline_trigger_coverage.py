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
EXECUTION_CONTRACT_SOURCE_PATHS = (
    "minecraft_mod_ai/execution_contract_policy.py",
    "minecraft_mod_ai/java_generation_policy.py",
    "minecraft_mod_ai/typed_host_generation_contract.py",
    "minecraft_mod_ai/generation_target_compile.py",
    "minecraft_mod_ai/generation_compile_state.py",
    "minecraft_mod_ai/generation_verification_contract.py",
    "minecraft_mod_ai/generation_evidence_controller.py",
    "minecraft_mod_ai/generation_verifier_resilience.py",
)
EXECUTION_CONTRACT_TARGETED_GATES = (
    ".github/workflows/production-path-contract.yml",
    ".github/workflows/runtime-generation-regression.yml",
    ".github/workflows/candidate-verifier-targeted.yml",
    ".github/workflows/implementation-ir-regression.yml",
    ".github/workflows/deterministic-debug-full-e2e.yml",
)
SOURCE_OWNED_RUNTIME_PATHS = (
    "minecraft_mod_ai/acceptance_contracts.py",
    "minecraft_mod_ai/quality_evidence.py",
    "minecraft_mod_ai/structural_minecraft_runtime_contract.py",
    "minecraft_mod_ai/research_rag_performance.py",
    "minecraft_mod_ai/source_observation_context.py",
    "minecraft_mod_ai/generator.py",
    "minecraft_mod_ai/texture_equivalence_cache.py",
    "minecraft_mod_ai/extended_registration_contract.py",
)
AUTHORED_PIPELINE_GATE_PATHS = (
    "minecraft_mod_ai/authored_section_ids.py",
    "minecraft_mod_ai/authored_structured_design.py",
    "minecraft_mod_ai/complete_planner.py",
    "minecraft_mod_ai/authored_production.py",
    "minecraft_mod_ai/planning_contract_ssot.py",
    "minecraft_mod_ai/planning_detail_contract.py",
    "minecraft_mod_ai/planning_detail_slots.py",
    "minecraft_mod_ai/planning_detail_template.py",
    "minecraft_mod_ai/planning_handoff_contract.py",
    "minecraft_mod_ai/typed_plan_ir.py",
    "minecraft_mod_ai/typed_plan_production.py",
    "minecraft_mod_ai/typed_host_generation_contract.py",
    "minecraft_mod_ai/implementation_decisions.py",
    "minecraft_mod_ai/task_artifact_contract.py",
    "minecraft_mod_ai/task_template_catalog.py",
)
DEEP_SOFTWARE_GATE_PATHS = (
    *SOURCE_OWNED_RUNTIME_PATHS,
    *AUTHORED_PIPELINE_GATE_PATHS,
    "tools/verify_integrity_minecraft.py",
    "tests/test_typed_plan_ir.py",
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



def test_execution_contract_changes_reach_every_generation_gate() -> None:
    workflows = (*SOFTWARE_GATES, *EXECUTION_CONTRACT_TARGETED_GATES)
    for workflow in workflows:
        text = _text(workflow)
        for path in EXECUTION_CONTRACT_SOURCE_PATHS:
            assert path in text, f"{workflow} missing canonical contract path {path}"


def test_execution_contract_changes_reach_real_model_gate() -> None:
    text = _text(".github/workflows/real-model-colab-e2e.yml")
    for path in EXECUTION_CONTRACT_SOURCE_PATHS:
        assert path in text, f"real-model gate missing canonical contract path {path}"


def test_targeted_generation_gates_execute_ssot_regression() -> None:
    for workflow in (
        ".github/workflows/production-path-contract.yml",
        ".github/workflows/runtime-generation-regression.yml",
        ".github/workflows/candidate-verifier-targeted.yml",
        ".github/workflows/implementation-ir-regression.yml",
    ):
        text = _text(workflow)
        assert "tests/test_execution_contract_policy.py" in text, workflow
