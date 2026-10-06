from __future__ import annotations

from minecraft_mod_ai import rag_index
from minecraft_mod_ai import small_model_retrieval_efficiency_contract as selective
from minecraft_mod_ai.production_tools import ProductionToolService


def _messages(reason: str):
    return [
        {"role": "system", "content": "Repair the rejected coder result."},
        {
            "role": "user",
            "content": f"Execution & Validation Failure: failed with reason: {reason}",
        },
    ]


def _install_selective_contract() -> None:
    selective.install()


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


def test_selective_repair_contract_is_explicitly_installable() -> None:
    _install_selective_contract()
    assert getattr(
        ProductionToolService.index_project_rag,
        "_mmm_explicit_semantic_index_policy",
        False,
    )
    assert getattr(
        rag_index._expand_sqlite_relationships,
        "_mmm_bidirectional_dependency_graph",
        False,
    )
