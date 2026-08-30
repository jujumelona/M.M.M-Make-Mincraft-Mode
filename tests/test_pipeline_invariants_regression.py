from __future__ import annotations

import inspect
from types import SimpleNamespace

from minecraft_mod_ai import evidence_obligation_contract as obligations
from minecraft_mod_ai import grounded_rag_runtime_contract as grounded
from minecraft_mod_ai import reuse_planner


def _catalog(count: int = 2) -> dict:
    return {
        "schema_version": "mmm/approved-requirement-graph-v1",
        "catalog_sha256": "sha256:test",
        "requirements": [
            {
                "requirement_id": f"req_{index}",
                "capability": f"capability_{index}",
                "semantic_statement": f"implement capability {index}",
            }
            for index in range(count)
        ],
    }


def test_pre_target_freeze_only_retrieves_target_neutral_reuse_evidence():
    brief = obligations.build_evidence_obligation_brief(
        "test",
        _catalog(),
        {"_platform_selection": {"target_frozen": False}},
    )
    nodes = brief["evidence_obligation_dag"]["nodes"]
    assert len(nodes) == 2
    assert {node["kind"] for node in nodes} == {"reusable_implementation"}
    assert brief["target_frozen"] is False
    deferred = set(brief["deferred_obligation_kinds"])
    assert {
        "target_compatibility",
        "implementation_api",
        "dependency_closure",
        "license_provenance",
        "validation_mechanism",
    } <= deferred


def test_frozen_target_expands_complete_obligation_dag():
    brief = obligations.build_evidence_obligation_brief(
        "test",
        _catalog(1),
        {"_platform_selection": {"target_frozen": True}},
    )
    nodes = brief["evidence_obligation_dag"]["nodes"]
    assert len(nodes) == len(obligations._OBLIGATIONS)
    assert {node["kind"] for node in nodes} == {
        str(spec["kind"]) for spec in obligations._OBLIGATIONS
    }
    assert brief["target_frozen"] is True
    assert brief["deferred_obligation_kinds"] == []


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediateCoordinator:
    max_workers = 2

    def __init__(self):
        self.local_ready = False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))

    def retrieve_many(self, queries, versions):
        return {}

    def repositories_for_capabilities(
        self, capabilities, capability_graph=None
    ):
        return {
            capability: ("https://github.com/example/donor",)
            for capability in capabilities
        }


def test_grounded_bundle_waits_for_local_index_before_base_bundle(monkeypatch):
    coordinator = _ImmediateCoordinator()
    monkeypatch.setattr(grounded, "_COORDINATOR", coordinator)
    monkeypatch.setattr(grounded, "_INSTALLED", False)
    monkeypatch.setattr(
        grounded,
        "_augment",
        lambda agentic, payload, **kwargs: payload,
    )
    old_external = grounded._grounded._external_retrieval

    def ensure_local_index(agentic_module, router):
        coordinator.local_ready = True
        return {"status": "available", "index_path": "/tmp/index"}

    monkeypatch.setattr(
        grounded._grounded,
        "_ensure_local_index",
        ensure_local_index,
    )

    def base(router, brief):
        assert coordinator.local_ready
        return {"versions": [], "domains": []}

    agentic = SimpleNamespace(_forced_rag_bundle=base)

    def public_discovery(capabilities, client, *, capability_graph=None):
        if client is None:
            raise AssertionError("public discovery must not run without a client")
        return {capability: () for capability in capabilities}

    reuse = SimpleNamespace(
        _parallel_donor_repository_discovery=public_discovery
    )
    try:
        grounded.install(agentic, reuse)
        assert agentic._forced_rag_bundle(None, {"domains": []}) == {
            "versions": [],
            "domains": [],
        }
        donors = reuse._parallel_donor_repository_discovery(
            ("capability",),
            None,
            capability_graph={},
        )
        assert donors["capability"] == (
            "https://github.com/example/donor",
        )
    finally:
        grounded._grounded._external_retrieval = old_external


def test_reuse_planner_does_not_gate_grounded_donors_on_public_discovery():
    source = inspect.getsource(reuse_planner.optimize_platform_and_reuse)
    assert "grounded_donors_available" in source
    assert "__mmm_grounded_donors__" in source
    assert "client if evidence_discovery_enabled else None" in source
