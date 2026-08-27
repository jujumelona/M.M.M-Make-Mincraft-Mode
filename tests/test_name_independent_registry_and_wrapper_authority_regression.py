from __future__ import annotations

"""Regression tests verifying name-independent implementation inclusion in
registry evidence / canonical graph closure and single-authority invariants
for proof build transitions and wrapper bindings.

Invariants under test
---------------------
1. **Registry evidence is name-independent**: ``find_verified_component``
   resolves candidates by *capability* identity, not by ``component_id`` name.
   Two components with entirely different names but the same capability must
   be interchangeable in discovery.

2. **Canonical graph closure**: ``ArtifactDependencyGraph.is_closure_complete``
   must only evaluate structural reachability – a graph whose nodes have zero
   unresolved/ambiguous edges is closure-complete regardless of the node names
   or class identifiers contained inside.

3. **Proof-build single authority**: ``validate_proof_transition`` must enforce
   a single linear lifecycle path; there is no way for two independent callers
   to simultaneously advance the same proof level without one of them failing.

4. **Wrapper-binding single authority**: ``owns_contract_marker`` must resolve
   to exactly one effective owner per marker across the full ``__wrapped__``
   chain, regardless of how many layers carry a copy of the marker.
"""

from functools import wraps
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.artifact_dependency_graph import (
    ArtifactDependencyGraph,
    ArtifactKind,
    ArtifactNode,
    UnresolvedArtifactEdge,
)
from minecraft_mod_ai.component_registry import (
    VerifiedComponent,
    find_verified_component,
)
from minecraft_mod_ai.proof_level import ProofLevel, validate_proof_transition
from minecraft_mod_ai.reuse_build_verifier import BuildVerificationReceipt
from minecraft_mod_ai.runtime_contract_wrappers import (
    contract_markers,
    contract_wraps,
    has_contract_marker,
    owns_contract_marker,
)
from minecraft_mod_ai.runtime_wrapper_integrity import (
    iter_installed_wrappers,
    wrapped_chain,
)
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice


# ---------------------------------------------------------------------------
# 1. Registry evidence – name-independent implementation class inclusion
# ---------------------------------------------------------------------------


class TestRegistryEvidenceNameIndependence:
    """Implementation classes must be discoverable by capability, not name."""

    @staticmethod
    def _make_component(component_id: str, capabilities: tuple[str, ...], **kw) -> VerifiedComponent:
        return VerifiedComponent(
            component_id=component_id,
            capabilities=capabilities,
            minecraft_version=kw.get("minecraft_version", "1.21.1"),
            loader=kw.get("loader", "fabric"),
            source_origin="test",
            source_commit="a" * 40,
            license_id="MIT",
            public_symbols=(),
            required_dependencies=(),
            test_receipts=("build:PASS",),
            artifact={},
        )

    def test_find_by_capability_ignores_component_name(self) -> None:
        """Two components with different names but same capability must both
        be discoverable; the winner is deterministic by receipts then id."""
        alpha = self._make_component("zzz_last_alphabetically", ("combat.damage",))
        beta = self._make_component("aaa_first_alphabetically", ("combat.damage",))

        found = find_verified_component(
            [alpha, beta],
            capability="combat.damage",
            minecraft_version="1.21.1",
            loader="fabric",
        )
        # The name ordering must not affect discovery – both match.
        assert found is not None
        assert "combat.damage" in found.capabilities

    def test_different_names_same_capability_resolve_identically(self) -> None:
        """Renaming a component must not break registry evidence resolution."""
        original = self._make_component("mod_original_boss", ("boss.entity",))
        renamed = self._make_component("mod_refactored_boss_v2", ("boss.entity",))

        result_orig = find_verified_component(
            [original],
            capability="boss.entity",
            minecraft_version="1.21.1",
            loader="fabric",
        )
        result_renamed = find_verified_component(
            [renamed],
            capability="boss.entity",
            minecraft_version="1.21.1",
            loader="fabric",
        )
        assert result_orig is not None
        assert result_renamed is not None
        assert result_orig.capabilities == result_renamed.capabilities

    def test_capability_not_present_returns_none(self) -> None:
        """A component with unrelated capabilities must never match."""
        comp = self._make_component("unrelated_item", ("item.equipment",))
        found = find_verified_component(
            [comp],
            capability="combat.damage",
            minecraft_version="1.21.1",
            loader="fabric",
        )
        assert found is None

    def test_multiple_capabilities_all_discoverable(self) -> None:
        """A single component advertising N capabilities must be found under each."""
        multi = self._make_component(
            "multi_cap_module",
            ("energy.generator", "energy.storage", "block_entity.tick"),
        )
        for cap in ("energy.generator", "energy.storage", "block_entity.tick"):
            result = find_verified_component(
                [multi], capability=cap, minecraft_version="1.21.1", loader="fabric",
            )
            assert result is not None, f"capability {cap!r} not discoverable"
            assert result.component_id == "multi_cap_module"

    def test_case_insensitive_capability_lookup(self) -> None:
        """Capability matching must be case-insensitive."""
        comp = self._make_component("boss_mod", ("Boss.Entity",))
        found = find_verified_component(
            [comp], capability="boss.entity", minecraft_version="1.21.1", loader="fabric",
        )
        assert found is not None

    def test_receipt_count_breaks_ties_before_name(self) -> None:
        """When two components share a capability, the one with more test
        receipts wins – name is a secondary tiebreaker only."""
        fewer = self._make_component("zzz_name", ("item.weapon",))
        more = VerifiedComponent(
            component_id="aaa_name",
            capabilities=("item.weapon",),
            minecraft_version="1.21.1",
            loader="fabric",
            source_origin="test",
            source_commit="b" * 40,
            license_id="MIT",
            public_symbols=(),
            required_dependencies=(),
            test_receipts=("build:PASS", "runtime:PASS", "playtest:PASS"),
            artifact={},
        )
        found = find_verified_component(
            [fewer, more],
            capability="item.weapon",
            minecraft_version="1.21.1",
            loader="fabric",
        )
        assert found is not None
        assert found.component_id == "aaa_name", (
            "component with more receipts must win regardless of name ordering"
        )


# ---------------------------------------------------------------------------
# 2. Canonical graph closure – name-independent structural completeness
# ---------------------------------------------------------------------------


class TestCanonicalGraphClosureNameIndependence:
    """Graph closure must only care about structural edge resolution, not node
    identifiers or class names inside nodes."""

    @staticmethod
    def _make_node(rel_path: str, kind: ArtifactKind = ArtifactKind.JAVA_SOURCE) -> ArtifactNode:
        return ArtifactNode(
            id=rel_path,
            kind=kind,
            namespace="com.example",
            logical_id=rel_path.rsplit("/", 1)[-1].removesuffix(".java"),
            environment="main",
            rel_path=rel_path,
            symbols_defined=(rel_path.rsplit("/", 1)[-1].removesuffix(".java"),),
        )

    def test_clean_graph_closure_complete(self) -> None:
        """A graph with no unresolved/ambiguous edges is closure-complete
        regardless of node names."""
        graph = ArtifactDependencyGraph()
        graph.add_node(self._make_node("src/main/java/Alpha.java"))
        graph.add_node(self._make_node("src/main/java/Beta.java"))
        graph.add_edge("src/main/java/Alpha.java", "src/main/java/Beta.java")

        assert graph.is_closure_complete(
            ["src/main/java/Alpha.java", "src/main/java/Beta.java"]
        )

    def test_unresolved_edge_breaks_closure(self) -> None:
        """A single unresolved edge inside the closure must fail."""
        graph = ArtifactDependencyGraph()
        graph.add_node(self._make_node("src/main/java/Main.java"))
        graph.unresolved_edges = (
            UnresolvedArtifactEdge(
                source_id="src/main/java/Main.java",
                requested_target="MissingClass",
                relation="import",
                reason="symbol not found",
            ),
        )

        assert not graph.is_closure_complete(["src/main/java/Main.java"])

    def test_unresolved_edge_outside_closure_does_not_affect(self) -> None:
        """Unresolved edges whose source is outside the queried closure must
        not invalidate it."""
        graph = ArtifactDependencyGraph()
        graph.add_node(self._make_node("src/main/java/Inside.java"))
        graph.add_node(self._make_node("src/main/java/Outside.java"))
        graph.unresolved_edges = (
            UnresolvedArtifactEdge(
                source_id="src/main/java/Outside.java",
                requested_target="Foo",
                relation="import",
                reason="missing",
            ),
        )

        assert graph.is_closure_complete(["src/main/java/Inside.java"])

    def test_renamed_nodes_same_structure_same_closure(self) -> None:
        """Renaming every node identifier must not change closure status."""
        for prefix in ("com/original/", "com/refactored/v2/"):
            graph = ArtifactDependencyGraph()
            a = f"src/main/java/{prefix}A.java"
            b = f"src/main/java/{prefix}B.java"
            graph.add_node(self._make_node(a))
            graph.add_node(self._make_node(b))
            graph.add_edge(a, b)
            assert graph.is_closure_complete([a, b]), (
                f"closure with prefix {prefix!r} unexpectedly incomplete"
            )

    def test_donor_slice_closure_complete_drives_metadata_match(self) -> None:
        """DonorSlice.metadata_match must be True only when both
        target_compatibility is exact/metadata_exact AND closure_complete is True."""
        base = dict(
            capability="item.weapon",
            repository="example/mod",
            commit_sha="c" * 40,
            license_id="MIT",
            source_url="https://example.com",
            files=(
                DonorFile(path="A.java", blob_sha="b", sha256="sha", size_bytes=10, symbols=("A",)),
            ),
            seed_files=("A.java",),
            source_symbols=("A",),
            required_dependencies=(),
            donor_tests=(),
            confidence=0.9,
        )

        # closure_complete + exact → metadata_match
        ds = DonorSlice(**base, target_compatibility="exact", closure_complete=True)
        assert ds.metadata_match is True

        # closure_complete=False → no metadata_match
        ds = DonorSlice(**base, target_compatibility="exact", closure_complete=False)
        assert ds.metadata_match is False

        # non-exact + closure_complete → no metadata_match
        ds = DonorSlice(**base, target_compatibility="adapt", closure_complete=True)
        assert ds.metadata_match is False


# ---------------------------------------------------------------------------
# 3. Proof-build single authority
# ---------------------------------------------------------------------------


class TestProofBuildSingleAuthority:
    """Proof lifecycle transitions must form a single linear authority chain."""

    def test_forward_transition_requires_receipt_for_verified_levels(self) -> None:
        """Transitioning to a verified proof level without a receipt must fail."""
        for target in (
            ProofLevel.COMPILE_VERIFIED,
            ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
            ProofLevel.PARTIAL_REUSE,
        ):
            valid, reason = validate_proof_transition(
                ProofLevel.CLOSURE_COMPLETE, target, receipt=None,
            )
            assert not valid, f"transition to {target.value} accepted without receipt"
            assert "MISSING_RECEIPT" in reason

    def test_forward_transition_with_receipt_succeeds(self) -> None:
        """A valid forward transition with a receipt must succeed."""
        valid, reason = validate_proof_transition(
            ProofLevel.CLOSURE_COMPLETE,
            ProofLevel.COMPILE_VERIFIED,
            receipt={"compile": True},
        )
        assert valid, f"valid transition rejected: {reason}"

    def test_illegal_skip_transition_rejected(self) -> None:
        """Skipping intermediate states (DISCOVERED → COMPILE_VERIFIED) must fail."""
        valid, reason = validate_proof_transition(
            ProofLevel.DISCOVERED,
            ProofLevel.COMPILE_VERIFIED,
            receipt={"compile": True},
        )
        assert not valid
        assert "ILLEGAL_TRANSITION" in reason

    def test_identity_transition_always_valid(self) -> None:
        """Same-state transitions are always allowed (idempotent)."""
        for level in ProofLevel:
            valid, reason = validate_proof_transition(level, level)
            assert valid, f"identity transition {level.value} → {level.value} rejected"
            assert reason == "identity_transition"

    def test_every_verified_level_requires_receipt(self) -> None:
        """Every level in the verified set must reject receipt-less transitions
        from any permitted source."""
        receipt_required_levels = {
            ProofLevel.COMPILE_VERIFIED,
            ProofLevel.BEHAVIOR_VERIFIED,
            ProofLevel.HOST_VERIFIED,
            ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
            ProofLevel.PARTIAL_REUSE,
        }
        from minecraft_mod_ai.proof_level import _LEGAL_TRANSITIONS

        for target in receipt_required_levels:
            for src, allowed in _LEGAL_TRANSITIONS.items():
                if target in allowed:
                    valid, reason = validate_proof_transition(src, target, receipt=None)
                    assert not valid, (
                        f"{src.value} → {target.value} should require receipt"
                    )

    def test_build_receipt_schema_version_is_singleton(self) -> None:
        """BuildVerificationReceipt.to_dict always emits exactly one
        schema_version key – i.e. the schema authority is singular."""
        receipt = BuildVerificationReceipt(
            build_tool="gradle_wrapper",
            command=("./gradlew", "compileJava"),
            exit_code=0,
            stdout="BUILD SUCCESSFUL",
            stderr="",
            compile_passed=True,
            tests_passed=True,
            unresolved_symbols=(),
            missing_resources=(),
        )
        d = receipt.to_dict()
        assert d["schema_version"] == "mmm/build-verification-receipt-v1"
        # Ensure no duplicate keys (dict guarantees this, but assert schema
        # string is exact single value)
        assert isinstance(d["schema_version"], str)
        assert d["schema_version"].count("/") == 1


# ---------------------------------------------------------------------------
# 4. Wrapper-binding single authority
# ---------------------------------------------------------------------------


class TestWrapperBindingSingleAuthority:
    """Each contract marker in a wrapper chain must have exactly one
    effective owner – the deepest layer carrying it."""

    def test_single_owner_in_two_layer_chain(self) -> None:
        """contract_wraps must not propagate inner markers to the outer layer."""
        def inner(x: object) -> object:
            return x

        inner._mmm_test_marker = True  # type: ignore[attr-defined]

        @contract_wraps(inner)
        def outer(x: object) -> object:
            return inner(x)

        assert owns_contract_marker(inner, "_mmm_test_marker")
        assert not owns_contract_marker(outer, "_mmm_test_marker")
        assert has_contract_marker(outer, "_mmm_test_marker")

    def test_legacy_wraps_copies_marker_but_owner_stays_inner(self) -> None:
        """Standard functools.wraps copies __dict__, but ownership must
        still resolve to the deepest layer."""
        def inner(x: object) -> object:
            return x

        inner._mmm_legacy_marker = True  # type: ignore[attr-defined]

        @wraps(inner)
        def outer(x: object) -> object:
            return inner(x)

        # Legacy wraps copies markers
        assert "_mmm_legacy_marker" in outer.__dict__
        # But ownership is NOT the outer
        assert not owns_contract_marker(outer, "_mmm_legacy_marker")
        assert owns_contract_marker(inner, "_mmm_legacy_marker")

    def test_three_layer_chain_single_owner(self) -> None:
        """In a three-layer chain, only the deepest layer owns the marker."""
        def base(x: object) -> object:
            return x

        base._mmm_deep = True  # type: ignore[attr-defined]

        @wraps(base)
        def mid(x: object) -> object:
            return base(x)

        @wraps(mid)
        def top(x: object) -> object:
            return mid(x)

        # wraps copies marker to all layers
        assert top.__dict__.get("_mmm_deep") is True
        assert mid.__dict__.get("_mmm_deep") is True
        # Only the deepest owns it
        assert owns_contract_marker(base, "_mmm_deep")
        assert not owns_contract_marker(mid, "_mmm_deep")
        assert not owns_contract_marker(top, "_mmm_deep")

    def test_independent_markers_on_different_layers(self) -> None:
        """Different markers introduced at different layers must each have
        exactly one owner."""
        def base(x: object) -> object:
            return x

        base._mmm_base_marker = True  # type: ignore[attr-defined]

        @contract_wraps(base)
        def outer(x: object) -> object:
            return base(x)

        outer._mmm_outer_marker = True  # type: ignore[attr-defined]

        assert contract_markers(base) == frozenset({"_mmm_base_marker"})
        assert contract_markers(outer) == frozenset({"_mmm_outer_marker"})

    def test_installed_wrappers_single_authority_regression(self) -> None:
        """Every installed MMM wrapper in the current runtime must have at
        most one effective owner per contract marker.  This is the key
        regression guard against marker duplication bugs."""
        import minecraft_mod_ai  # noqa: F401  # trigger runtime bootstrap

        violations: list[str] = []
        for binding, outer in iter_installed_wrappers():
            chain = wrapped_chain(outer)
            marker_owners: dict[str, list[int]] = {}
            for idx, layer in enumerate(chain):
                for marker in contract_markers(layer):
                    marker_owners.setdefault(marker, []).append(idx)
            for marker, idxs in marker_owners.items():
                if len(idxs) > 1:
                    violations.append(
                        f"{binding}: {marker} owned at layers {idxs}"
                    )
        assert not violations, (
            "wrapper binding single-authority violated:\n"
            + "\n".join(sorted(set(violations)))
        )
