from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai import evidence_first_planning as evidence
from minecraft_mod_ai.production_boundary_contract import (
    _install_planner_public_acceptance_guard,
)


PROMPT = "Add a weather compass that reports the current weather to the player."
INTERNAL_ACCEPTANCE = (
    "Task task_internal: done_predicate verifies all declared provides and owned anchors."
)


def _verified_polluted_catalog() -> dict:
    _install_planner_public_acceptance_guard()
    clean = evidence.build_request_catalog(PROMPT, {})
    assert clean["requirements"]

    polluted = deepcopy(clean)
    polluted["requirements"][0]["acceptance"] = [INTERNAL_ACCEPTANCE]
    polluted["catalog_sha256"] = ""
    polluted["catalog_sha256"] = evidence._hash_without(
        polluted,
        "catalog_sha256",
    )
    return polluted


def test_verified_reused_catalog_migrates_internal_public_acceptance() -> None:
    polluted = _verified_polluted_catalog()

    migrated = evidence.build_request_catalog(
        PROMPT,
        {"_evidence_request_catalog": polluted},
    )

    statement = " ".join(migrated["requirements"][0]["acceptance"]).casefold()
    for marker in (
        "task_",
        "done_predicate",
        "declared provides",
        "owned anchor",
    ):
        assert marker not in statement
    assert migrated["catalog_sha256"] == evidence._hash_without(
        migrated,
        "catalog_sha256",
    )


def test_reused_catalog_migration_does_not_accept_tampered_hash() -> None:
    polluted = _verified_polluted_catalog()
    polluted["requirements"][0]["acceptance"][0] += " tampered"

    with pytest.raises(evidence.EvidencePlanError, match="hash mismatch"):
        evidence.build_request_catalog(
            PROMPT,
            {"_evidence_request_catalog": polluted},
        )
