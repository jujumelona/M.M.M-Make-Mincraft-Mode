from __future__ import annotations

from dataclasses import replace

from minecraft_mod_ai import grounded_source_reuse
from minecraft_mod_ai.host_version_catalog import host_target
from minecraft_mod_ai.reuse_proof_executor import ReuseProofReceipt
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice, donor_closure_sha256


def _discovery(monkeypatch, repositories):
    monkeypatch.delenv("MMM_REFERENCE_MOD_URLS", raising=False)
    monkeypatch.setattr(grounded_source_reuse, "_adapter", lambda _: host_target("26.1.2"))
    monkeypatch.setattr("minecraft_mod_ai.reuse_discovery.discover_repositories_for_graph",
                        lambda *args, **kwargs: {"space travel": repositories})
    return {"_pre_retrieval_plan": {"plan_sha256": "sha256:" + "a" * 64,
                                     "capability_graph": {"nodes": ["space travel"]}}}


def _donor(repository):
    path = "src/main/java/space/Travel.java"
    return DonorSlice(
        capability="space travel", repository=repository, commit_sha="b" * 40, license_id="MIT",
        source_url="https://github.com/" + repository, target_compatibility="exact",
        files=(DonorFile(path, "c" * 40, "sha256:" + "d" * 64, 1, ()),), seed_files=(path,),
        source_symbols=(), required_dependencies=(), donor_tests=(), confidence=0.8,
    )


def _proof(donor, *, passed=True):
    return ReuseProofReceipt(
        candidate_id=f"{donor.repository}@{donor.commit_sha}", capability=donor.capability,
        commit_sha=donor.commit_sha, closure_hash=donor_closure_sha256(donor),
        proof_level="COMPILE_VERIFIED" if passed else "UNVERIFIED",
        compile_passed=passed, tests_passed=False, unresolved_symbols=(), missing_resources=(),
        adaptations_applied=(), verified_capabilities=(donor.capability,) if passed else (),
        residual_capabilities=(), authoritative_compile=passed,
        verified_artifacts=tuple(row.path for row in donor.files) if passed else (),
        failure_message="Old API requires adaptation" if not passed else "",
    )


def test_proven_ranked_candidate_avoids_downloading_the_remaining_frontier(monkeypatch, tmp_path):
    repositories = tuple(f"example/travel{i}" for i in range(12))
    design = _discovery(monkeypatch, repositories)
    inspected, proved = [], []

    def inspect(**kwargs):
        repository = kwargs["repository"]
        inspected.append(repository)
        donor = _donor(repository)
        return replace(donor, target_compatibility="adapt") if repository == repositories[0] else donor

    def prove(donor, **kwargs):
        proved.append(donor.repository)
        return _proof(donor)

    result = grounded_source_reuse.build_repository_reuse_plan(
        design, discovery_client=object(), donor_inspector=inspect, proof_executor=prove,
        target_workspace=tmp_path,
    )
    assert inspected == list(repositories[:2])
    assert proved == [repositories[1]]
    assert result["capabilities"][0]["donor"]["repository"] == repositories[1]
    assert result["host_discovered_repositories"]["space travel"] == list(repositories)


def test_rejected_batches_do_not_truncate_later_usable_candidates(monkeypatch, tmp_path):
    repositories = tuple(f"example/travel{i}" for i in range(12))
    design = _discovery(monkeypatch, repositories)
    inspected, proved = [], []

    def inspect(**kwargs):
        inspected.append(kwargs["repository"])
        return _donor(kwargs["repository"])

    def prove(donor, **kwargs):
        proved.append(donor.repository)
        return _proof(donor, passed=donor.repository == repositories[-1])

    result = grounded_source_reuse.build_repository_reuse_plan(
        design, discovery_client=object(), donor_inspector=inspect, proof_executor=prove,
        target_workspace=tmp_path,
    )
    assert inspected == proved == list(repositories)
    assert result["capabilities"][0]["donor"]["repository"] == repositories[-1]
    assert len(result["reference_sources"]) == 11


def test_provider_outage_stops_repeated_source_inspections(monkeypatch, tmp_path):
    import httpx

    repositories = tuple(f"example/travel{index}" for index in range(12))
    design = _discovery(monkeypatch, repositories)
    inspected = []

    def inspect(**kwargs):
        inspected.append(kwargs["repository"])
        raise httpx.ConnectError("source provider unavailable")

    def cannot_prove(*args, **kwargs):
        raise AssertionError("No source download, no compile proof")

    receipt = grounded_source_reuse.build_repository_reuse_plan(
        design, discovery_client=object(), donor_inspector=inspect,
        proof_executor=cannot_prove, target_workspace=tmp_path,
    )
    assert inspected == [repositories[0]]
    assert receipt["capabilities"][0]["mode"] == "fresh"
    assert receipt["capabilities"][0]["search_status"] == "provider_error"
    assert receipt["capabilities"][0]["failure_code"] == "DONOR_INSPECTION_PROVIDER_UNAVAILABLE"
    assert receipt["inspection_receipts"][0]["provider_unavailable"]


def test_predesign_candidate_is_proved_before_any_new_provider_search(monkeypatch, tmp_path):
    """A real research candidate should be inspected, not re-discovered."""
    design = _discovery(monkeypatch, ())
    design["_planning_state"] = {"repository_candidates": [{
        "repository": "example/predesign-source",
        "source_ids": ["curseforge:42"],
        "source_urls": ["https://github.com/example/predesign-source"],
        "query_sha256": ["sha256:research"],
        "domain_ids": ["space-travel"],
        "origin": "host_grounded_retrieval",
        "reference_only": True,
        "source_reuse_authority": "verification_required",
        "evidence_text": "Spaceship and planet travel gameplay",
        "evidence_tokens": ["spaceship", "planet", "travel"],
        "candidate_score": 2.0,
        "catalog_evidence": [{
            "provider": "curseforge", "source_id": "curseforge:42",
            "title": "Planet Travel", "summary": "Spaceship and planet travel",
            "source_url": "https://github.com/example/predesign-source",
        }],
    }]}
    attempted = []

    def fail_discovery(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("Already proven predesign donor must not trigger a new search")

    monkeypatch.setattr(
        "minecraft_mod_ai.reuse_discovery.discover_repositories_for_graph",
        fail_discovery,
    )
    inspected = []

    def inspect(**kwargs):
        inspected.append(kwargs["repository"])
        return _donor(kwargs["repository"])

    result = grounded_source_reuse.build_repository_reuse_plan(
        design, discovery_client=object(), donor_inspector=inspect,
        proof_executor=lambda donor, **kwargs: _proof(donor),
        target_workspace=tmp_path,
    )
    assert inspected == ["example/predesign-source"]
    assert not attempted
    assert result["capabilities"][0]["mode"] == "source_transplant"
    assert result["planning_rag_candidate_repository_count"] == 1


def test_unusable_predesign_candidate_then_expands_search(monkeypatch, tmp_path):
    """A research hit that cannot be inspected must not stop fresh discovery."""
    design = _discovery(monkeypatch, ("example/fallback",))
    design["_planning_state"] = {"repository_candidates": [{
        "repository": "example/unusable",
        "origin": "host_grounded_retrieval", "reference_only": True,
        "source_reuse_authority": "verification_required",
        "source_ids": ["modrinth:foo"],
        "source_urls": ["https://github.com/example/unusable"],
    }]}
    inspected = []

    def inspect(**kwargs):
        inspected.append(kwargs["repository"])
        if kwargs["repository"] == "example/unusable":
            return None
        return _donor(kwargs["repository"])

    result = grounded_source_reuse.build_repository_reuse_plan(
        design, discovery_client=object(), donor_inspector=inspect,
        proof_executor=lambda donor, **kwargs: _proof(donor),
        target_workspace=tmp_path,
    )
    assert inspected == ["example/unusable", "example/fallback"]
    assert result["capabilities"][0]["mode"] == "source_transplant"
