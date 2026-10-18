"""The flat semantic-review receipt must reach source RAG as pinned code."""
import hashlib

import pytest

from minecraft_mod_ai import authored_source_rag, curseforge_reuse_production as production
from minecraft_mod_ai.curseforge_semantic_source_review import SourceInspectionError
from minecraft_mod_ai.host_version_catalog import host_target


PATH = "common/src/main/java/donor/space/PlanetTravel.java"
CODE = b"package donor.space; public class PlanetTravel { public void travelToPlanet() {} }"
LICENSE = b"SPDX-License-Identifier: MIT\nCopyright donor"


def reference():
    return {"capability": "space travel", "reference_only": True,
            "source_url": "https://github.com/example/space", "pinned_sha": "a" * 40,
            "license_spdx": "MIT", "selected_file": PATH,
            "selected_symbol": "PlanetTravel", "proof_status": "NOT_RUN",
            "file_sha256": "sha256:" + hashlib.sha256(CODE).hexdigest()}


def test_flat_reviewed_reference_reaches_owning_coder_without_transplant(monkeypatch):
    calls = []

    def fetch(url, sha, *, repository):
        calls.append((url, sha, repository))
        return {PATH: CODE, "LICENSE": LICENSE}

    monkeypatch.setattr(production, "fetch_pinned_archive", fetch)
    adapter = host_target("26.1.2")
    plan = {"bound_target": {"minecraft_version": "26.1.2", "loader": "fabric"},
            "capabilities": [{"capability": "space travel", "mode": "fresh"}],
            "reference_sources": [reference()]}
    with authored_source_rag.verified_source_rag(
        plan, adapter=adapter, package_name="test.space", mod_id="space",
    ) as rag:
        snippets = rag.context(["space travel"], "travelToPlanet")
        assert snippets and "travelToPlanet" in snippets[0]["code"]
        assert snippets[0]["authority"] == "reference_only_requires_target_adaptation"
        assert snippets[0]["source_closure_complete"] is False
        assert snippets[0]["source_commit"] == "a" * 40
        assert rag.reference_failures == []
        assert rag.license_notices
        assert rag.context(["trading"], "travelToPlanet") == []
    assert calls == [(reference()["source_url"], "a" * 40, "example/space")]
    assert plan["capabilities"][0]["mode"] == "fresh"


@pytest.mark.parametrize("change", ["hash", "license", "symbol", "missing_file"])
def test_pinned_reference_cannot_bypass_byte_license_or_symbol_checks(monkeypatch, change):
    ref = reference()
    files = {PATH: CODE, "LICENSE": LICENSE}
    if change == "hash":
        files[PATH] += b" // changed"
    elif change == "license":
        files["LICENSE"] = b"All rights reserved"
    elif change == "symbol":
        ref["selected_symbol"] = "InventedPlanet"
    else:
        files.pop(PATH)
    monkeypatch.setattr(production, "fetch_pinned_archive", lambda *_a, **_kw: files)
    with pytest.raises(SourceInspectionError):
        production.read_pinned_source_reference(ref)


def test_flat_reference_keeps_provenance_in_source_evidence():
    from minecraft_mod_ai.source_reuse_evidence import build_source_reuse_evidence

    rows = build_source_reuse_evidence([], {"reference_sources": [reference()]}, {})
    assert len(rows) == 1
    row = rows[0]
    assert row.mode == "reference" and row.status == "reference_available"
    assert row.repository == "example/space"
    assert row.commit_sha == "a" * 40
    assert row.license_id == "MIT"
    assert row.files == ({"path": PATH, "sha256": reference()["file_sha256"]},)
    assert row.verification == {}
