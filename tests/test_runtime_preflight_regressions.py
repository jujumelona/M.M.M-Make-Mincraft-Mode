from __future__ import annotations

import json
from typing import Any

from minecraft_mod_ai import causal_frontier_adapter as frontier_module
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse
from minecraft_mod_ai.retrieval_progress import _stable_value, evidence_fingerprint
from minecraft_mod_ai.runtime_preflight import run_runtime_preflight


def test_unordered_retrieval_values_have_one_canonical_fingerprint() -> None:
    assert _stable_value({"facts": {"b", "a"}}, drop_volatile=False) == {
        "facts": ["a", "b"]
    }
    assert evidence_fingerprint({"facts": {"a", "b"}}) == evidence_fingerprint(
        {"facts": frozenset(("b", "a"))}
    )


def test_model_free_runtime_preflight_is_idempotent() -> None:
    run_runtime_preflight()
    run_runtime_preflight()
