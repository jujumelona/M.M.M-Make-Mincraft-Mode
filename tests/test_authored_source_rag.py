from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import authored_source_rag


def test_fresh_search_hits_never_become_coder_source(monkeypatch):
    def download(*args):
        raise AssertionError("Unproved search results must not be downloaded as donors")

    monkeypatch.setattr("minecraft_mod_ai.source_transplant.materialize_source_slices", download)
    with authored_source_rag.verified_source_rag(
        {"capabilities": [{"mode": "fresh", "capability": "space travel"}]},
        adapter=SimpleNamespace(), package_name="test.mod", mod_id="test",
    ) as rag:
        assert rag.context(["space travel"], "planet transport") == []


@pytest.mark.parametrize("version", ["1.21.11", "26.1.2"])
def test_downloaded_adapted_source_is_retrieved_by_owning_feature(monkeypatch, tmp_path, version):
    source = tmp_path / "PlanetTravel.java"
    source.write_text("package donor.space; public class PlanetTravel { public static void travelToPlanet() {} }")
    calls = []

    def download(root, plan):
        calls.append(root)
        return {"donors": [{
            "capability": "space travel", "repository": "example/space",
            "commit_sha": "a" * 40, "license_id": "MIT", "required_dependencies": [],
            "files": [{"source_path": "src/main/java/donor/space/PlanetTravel.java", "path": str(source)}],
        }]}

    def adapt(blobs, target):
        assert target["target_package"] == "test.mod"
        data = next(iter(blobs.values())).replace(b"donor.space", b"test.mod")
        return {"src/main/java/test/mod/PlanetTravel.java": data}, {}

    monkeypatch.setattr("minecraft_mod_ai.source_transplant.materialize_source_slices", download)
    monkeypatch.setattr("minecraft_mod_ai.reuse_adapters.apply_deterministic_adapters", adapt)
    from minecraft_mod_ai.host_version_catalog import host_target

    adapter = host_target(version)
    plan = {"bound_target": {"minecraft_version": version, "loader": "fabric"},
            "capabilities": [{"mode": "source_transplant", "capability": "space travel"}]}
    with authored_source_rag.verified_source_rag(plan, adapter=adapter, package_name="test.mod", mod_id="test") as rag:
        result = rag.context(["space travel"], "travelToPlanet")
        assert result and "test.mod" in result[0]["code"]
        assert "travelToPlanet" in result[0]["code"]
        assert result[0]["repository"] == "example/space"
        assert result[0]["source_commit"] == "a" * 40
        assert result[0]["adapted_sha256"] == hashlib.sha256(
            source.read_bytes().replace(b"donor.space", b"test.mod")).hexdigest()
        assert rag.context(["economy trading"], "balance") == []
        assert len(json.dumps(result)) < 7000
        bindings = {}
        authored_source_rag.bind_owner_source_context(
            bindings, owner=SimpleNamespace(module_id="planet_transport", config={"requirement_refs": ["planet"]}),
            request_scope={"requirements": [{"requirement_id": "planet", "statement": "Travel to planets"}]},
            reuse_plan={"capability_graph": {"requirement_bindings": [
                {"requirement_id": "planet", "capability": "space travel"}]}},
            rag=rag, requirement="travelToPlanet",
        )
        assert "travelToPlanet" in json.loads(bindings["verified_source_code_json"])[0]["code"]
        assert json.loads(bindings["request_features_json"]) == [{"requirement_id": "planet", "statement": "Travel to planets"}]
        assert "missing glue" in bindings["source_integration_instruction"]
    assert len(calls) == 1


def test_unbound_target_cannot_supply_reused_api(monkeypatch):
    import pytest

    adapter = SimpleNamespace(minecraft_version="1.21.11", loader="fabric")
    plan = {"bound_target": {"minecraft_version": "1.20.1", "loader": "fabric"},
            "capabilities": [{"mode": "source_transplant", "capability": "space travel"}]}
    with pytest.raises(ValueError, match="SOURCE_REUSE_TARGET_MISMATCH"):
        with authored_source_rag.verified_source_rag(plan, adapter=adapter, package_name="test.mod", mod_id="test"):
            pass


def test_older_source_with_failed_proof_reaches_model_as_adaptation_reference(monkeypatch, tmp_path):
    from minecraft_mod_ai.authored_reuse_bridge import (
        materialize_verified_authored_sources,
    )
    from minecraft_mod_ai.host_version_catalog import host_target
    from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice

    path = "src/main/java/old/space/PlanetTravel.java"
    code = b"package old.space; public class PlanetTravel { public void travelToPlanet() {} }"
    donor = DonorSlice(
        capability="space travel", repository="example/oldspace", commit_sha="b" * 40,
        license_id="MIT", source_url="https://github.com/example/oldspace", target_compatibility="adapt",
        files=(DonorFile(path=path, blob_sha="c" * 40, sha256="sha256:" + hashlib.sha256(code).hexdigest(),
                         size_bytes=len(code), symbols=("PlanetTravel",)),),
        seed_files=(path,), source_symbols=("PlanetTravel",), required_dependencies=(), donor_tests=(), confidence=0.8,
    )
    monkeypatch.setattr("minecraft_mod_ai.source_transplant.materialize_pinned_donor",
                        lambda *args: {path: code, "LICENSE": b"MIT Copyright example"})
    adapter = host_target("26.1.2")
    plan = {"bound_target": {"minecraft_version": "26.1.2", "loader": "fabric"},
            "capabilities": [{"capability": "space travel", "mode": "fresh"}],
            "reference_sources": [{"capability": "space travel", "donor": donor.to_dict(),
                                   "migration_diagnostics": "Yarn API needs Mojang port"}]}
    with authored_source_rag.verified_source_rag(plan, adapter=adapter, package_name="test.space", mod_id="space") as rag:
        snippets = rag.context(["space travel"], "travelToPlanet")
        assert snippets[0]["authority"] == "reference_only_requires_target_adaptation"
        assert "Yarn API" in snippets[0]["migration_diagnostics"]
        assert "old.space" in snippets[0]["code"]
        plan["reference_license_notices"] = rag.license_notices
    receipt = materialize_verified_authored_sources(str(tmp_path), plan,
                                                  minecraft_version="26.1.2", loader="fabric")
    assert receipt["donor_count"] == 0
    assert not list(tmp_path.rglob("*.java"))
    notices = list(tmp_path.rglob("*.LICENSE"))
    assert len(notices) == 1 and "Copyright example" in notices[0].read_text()


def test_failed_deterministic_proof_preserves_downloadable_reference(monkeypatch, tmp_path):
    from minecraft_mod_ai import grounded_source_reuse
    from minecraft_mod_ai.host_version_catalog import host_target
    from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice

    monkeypatch.delenv("MMM_REFERENCE_MOD_URLS", raising=False)
    monkeypatch.setattr(grounded_source_reuse, "_adapter", lambda *args: host_target("26.1.2"))
    monkeypatch.setattr("minecraft_mod_ai.reuse_discovery.discover_repositories_for_graph", lambda *args, **kwargs: {
        "space travel": ("example/oldspace",)})
    donor = DonorSlice(
        capability="space travel", repository="example/oldspace", commit_sha="b" * 40, license_id="MIT",
        source_url="https://github.com/example/oldspace", target_compatibility="adapt",
        files=(DonorFile("src/main/java/Travel.java", "c" * 40, "sha256:" + "d" * 64, 1, ()),),
        seed_files=("src/main/java/Travel.java",), source_symbols=(), required_dependencies=(), donor_tests=(), confidence=0.8,
    )

    def cannot_migrate(*args, **kwargs):
        raise RuntimeError("Yarn migration requires model adaptation")

    result = grounded_source_reuse.build_repository_reuse_plan({
        "_pre_retrieval_plan": {"plan_sha256": "sha256:" + "a" * 64,
                                "capability_graph": {"nodes": ["space travel"]}},
    }, discovery_client=object(), donor_inspector=lambda **kwargs: donor,
        proof_executor=cannot_migrate, target_workspace=tmp_path)
    assert result["capabilities"][0]["mode"] == "fresh"
    assert result["reference_sources"][0]["donor"]["commit_sha"] == "b" * 40
    assert "model adaptation" in result["reference_sources"][0]["migration_diagnostics"]


def test_unavailable_optional_reference_does_not_abort_fresh_implementation(monkeypatch):
    from dataclasses import replace

    from minecraft_mod_ai.host_version_catalog import host_target
    from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice, SourceTransplantError

    path = "src/main/java/old/space/Travel.java"
    code = b"package old.space; public class Travel { public void travelToPlanet() {} }"
    donor = DonorSlice(
        capability="space travel", repository="example/unavailable", commit_sha="b" * 40,
        license_id="MIT", source_url="https://github.com/example/unavailable", target_compatibility="adapt",
        files=(DonorFile(path, "c" * 40, "sha256:" + hashlib.sha256(code).hexdigest(), len(code), ()),),
        seed_files=(path,), source_symbols=(), required_dependencies=(), donor_tests=(), confidence=0.8,
    )

    def download(candidate):
        if candidate.repository == "example/unavailable":
            raise SourceTransplantError("GitHub donor request failed: rate limited")
        return {path: code, "LICENSE": b"MIT Copyright available"}

    monkeypatch.setattr("minecraft_mod_ai.source_transplant.materialize_pinned_donor", download)
    plan = {"bound_target": {"minecraft_version": "26.1.2", "loader": "fabric"},
            "capabilities": [{"capability": "space travel", "mode": "fresh"}],
            "reference_sources": [{"capability": "space travel", "donor": candidate.to_dict()}
                                  for candidate in (donor, replace(donor, repository="example/available"))]}
    with authored_source_rag.verified_source_rag(
        plan, adapter=host_target("26.1.2"), package_name="test.space", mod_id="space",
    ) as rag:
        assert rag.context(["space travel"], "travelToPlanet")[0]["repository"] == "example/available"
        assert rag.reference_failures[0]["repository"] == "example/unavailable"
        assert "rate limited" in rag.reference_failures[0]["error"]
        assert len(rag.license_notices) == 1


def test_unavailable_installed_donor_still_aborts_production(monkeypatch):
    from minecraft_mod_ai.host_version_catalog import host_target
    from minecraft_mod_ai.source_transplant import SourceTransplantError

    def unavailable(*args):
        raise SourceTransplantError("Pinned installed source unavailable")

    monkeypatch.setattr("minecraft_mod_ai.source_transplant.materialize_source_slices", unavailable)
    plan = {"bound_target": {"minecraft_version": "26.1.2", "loader": "fabric"},
            "capabilities": [{"capability": "space travel", "mode": "source_transplant"}]}
    with pytest.raises(SourceTransplantError, match="installed source unavailable"):
        with authored_source_rag.verified_source_rag(
            plan, adapter=host_target("26.1.2"), package_name="test.space", mod_id="space",
        ):
            pass


def test_verified_source_fallback_reads_real_java_when_index_finds_nothing(tmp_path):
    from minecraft_mod_ai.authored_source_rag import VerifiedSourceRAG

    root = tmp_path / "source"
    java = root / "src/main/java/donor/PlanetTravel.java"
    java.parent.mkdir(parents=True)
    java.write_text(
        "package donor;\\n"
        "public class PlanetTravel {\\n"
        "  public void travelToPlanet() { System.out.println(1); }\\n"
        "}\\n", encoding="utf-8",
    )

    class EmptyIndex:
        def search(self, query, limit=8):
            return []

    rag = VerifiedSourceRAG()
    rag.sources.append(("space travel", EmptyIndex(), root, {
        "repository": "example/space", "commit_sha": "a" * 40,
        "license_id": "MIT",
    }))
    result = rag.context(["space travel"], "travelToPlanet")
    assert result and result[0]["symbol"] == "travelToPlanet"
    assert result[0]["authority"] == "installed_target_adapted_source"
    assert result[0]["retrieval_score"] > 0
    assert rag.context(["trading"], "travelToPlanet") == []


def test_verified_java_donor_cannot_be_silently_dropped_by_owner_rag():
    from minecraft_mod_ai.authored_source_rag import VerifiedSourceRAG, bind_owner_source_context

    owner = SimpleNamespace(module_id="planet", config={"requirement_refs": ["req-planet"]})
    plan = {
        "capability_graph": {"requirement_bindings": [
            {"requirement_id": "req-planet", "capability": "space travel"},
        ]},
        "capabilities": [{
            "capability": "space travel", "mode": "source_transplant",
            "donor": {"files": [{"path": "src/main/java/PlanetTravel.java"}]},
        }],
    }
    with pytest.raises(ValueError, match="SOURCE_RAG_VERIFIED_DONOR_NOT_BOUND"):
        bind_owner_source_context(
            {}, owner=owner,
            request_scope={"requirements": [
                {"requirement_id": "req-planet", "statement": "Travel to planets"},
            ]},
            reuse_plan=plan, rag=VerifiedSourceRAG(), requirement="Travel to planets",
        )
