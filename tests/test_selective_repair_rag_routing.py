from __future__ import annotations

from minecraft_mod_ai import small_model_retrieval_efficiency_contract as selective


def _messages(reason: str):
    return [
        {"role": "system", "content": "Repair the rejected coder result."},
        {
            "role": "user",
            "content": f"Execution & Validation Failure: failed with reason: {reason}",
        },
    ]


def test_compile_api_and_dependency_failures_request_retrieval() -> None:
    assert selective._needs_retrieval_repair(
        _messages("javac cannot find symbol RegistryKey")
    )
    assert selective._needs_retrieval_repair(
        _messages("API mismatch: no suitable method register()")
    )
    assert selective._needs_retrieval_repair(
        _messages("Gradle dependency package does not exist")
    )


def test_host_only_failures_do_not_request_retrieval() -> None:
    assert not selective._needs_retrieval_repair(
        _messages("expected_sha256 is missing for replace operation")
    )
    assert not selective._needs_retrieval_repair(
        _messages("duplicate patch path in transaction")
    )
    assert not selective._needs_retrieval_repair(
        _messages("runtime test timed out without a compiler diagnostic")
    )


def test_runtime_research_router_exposes_selective_repair_contract() -> None:
    from minecraft_mod_ai.custom_generation_search_contract import (
        _ResearchEvidenceRouter,
    )

    assert getattr(
        _ResearchEvidenceRouter.generate_text,
        "_mmm_selective_repair_rag",
        False,
    )
