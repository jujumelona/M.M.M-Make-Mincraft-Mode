from __future__ import annotations

"""Dependency-free authored document and execution section identifiers."""

EXECUTION_SECTION_ORDER = (
    "state_model",
    "behavior_contract",
    "algorithm",
    "authority_and_network",
    "persistence",
    "resources_and_ui",
    "failure_and_limits",
    "integration",
)
EXECUTION_SECTION_SET = frozenset(EXECUTION_SECTION_ORDER)

DOCUMENT_SECTION_ORDER = (
    "overview",
    "behavior_contract",
    "state_model",
    "algorithm",
    "integration",
    "authority_and_network",
    "persistence",
    "resources_and_ui",
    "failure_and_limits",
    "reuse_assessment",
    "verification",
    "conclusion",
)
DOCUMENT_SECTION_SET = frozenset(DOCUMENT_SECTION_ORDER)
CONTEXT_SECTION_SET = frozenset({
    "overview", "reuse_assessment", "verification", "conclusion",
})
REQUIRED_EXECUTION_SECTIONS = frozenset(EXECUTION_SECTION_ORDER)

__all__ = [
    "CONTEXT_SECTION_SET",
    "DOCUMENT_SECTION_ORDER",
    "DOCUMENT_SECTION_SET",
    "EXECUTION_SECTION_ORDER",
    "EXECUTION_SECTION_SET",
    "REQUIRED_EXECUTION_SECTIONS",
]
