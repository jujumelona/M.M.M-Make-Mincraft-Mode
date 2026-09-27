from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITICAL_RUNTIME_PATHS = (
    "minecraft_mod_ai/llama_*.py",
    "minecraft_mod_ai/qwen*.py",
    "minecraft_mod_ai/generation_output_budget.py",
    "minecraft_mod_ai/model_context_budget.py",
    "minecraft_mod_ai/model_output_atomicity_contract.py",
)
PUSH_GATES = (
    ".github/workflows/main-ci.yml",
    ".github/workflows/production-kill-gates.yml",
    ".github/workflows/cumulative-regression-guard.yml",
    ".github/workflows/real-model-colab-e2e.yml",
)


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_critical_runtime_changes_reach_end_to_end_gates() -> None:
    for workflow in PUSH_GATES:
        text = _text(workflow)
        assert "push:" in text, workflow
        assert "branches: [main]" in text, workflow
        for path in CRITICAL_RUNTIME_PATHS:
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
