from __future__ import annotations

"""Dependency-free authored document and execution section identifiers.

AUTHORING_SECTION_ORDER is the document/worksheet taxonomy. EXECUTION_SECTION_ORDER
is the dependency-oriented lowering order for sections that create executable work.
Keep both here so planning, authored Markdown normalization, and production cannot
silently drift onto different section vocabularies.
"""

AUTHORING_SECTION_ORDER = (
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
)

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
    *AUTHORING_SECTION_ORDER,
    "conclusion",
)
DOCUMENT_SECTION_SET = frozenset(DOCUMENT_SECTION_ORDER)
CONTEXT_SECTION_SET = frozenset({
    "overview", "reuse_assessment", "verification", "conclusion",
})
REQUIRED_EXECUTION_SECTIONS = frozenset(EXECUTION_SECTION_ORDER)

__all__ = [
    "AUTHORING_SECTION_ORDER",
    "CONTEXT_SECTION_SET",
    "DOCUMENT_SECTION_ORDER",
    "DOCUMENT_SECTION_SET",
    "EXECUTION_SECTION_ORDER",
    "EXECUTION_SECTION_SET",
    "REQUIRED_EXECUTION_SECTIONS",
]
