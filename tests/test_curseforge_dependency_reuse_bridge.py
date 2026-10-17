"""Explicit dependency selections must pass provider recheck before project mutation."""
from __future__ import annotations

from minecraft_mod_ai.authored_reuse_bridge import materialize_verified_authored_sources


def _plan():
    return {
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "bound_target": {"minecraft_version": "26.2", "loader": "fabric"},
        "capabilities": [{"capability": "alien", "mode": "fresh", "source_id": ""}],
        "requested_curseforge_dependencies": [{
            "project_id": 456, "file_id": 123,
            "jar_path": "/host/cache/alien.jar",
            "license_policy_confirmed": True,
        }],
    }


def test_selected_dependency_is_reverified_with_authenticated_file_response(
    tmp_path, monkeypatch,
):
    from minecraft_mod_ai import curseforge_dependency_adoption as verifier
    from minecraft_mod_ai import curseforge_dependency_project as installer

    called = []

    def fetch(project, file):
        called.append(("provider", project, file))
        return {"id": file, "modId": project, "isAvailable": True}

    def verify(jar_path, **kwargs):
        called.append(("integrity", jar_path, kwargs["minecraft_version"]))
        assert kwargs["file_metadata"]["id"] == 123
        assert kwargs["license_policy_confirmed"] is True
        return {"schema_version": "mmm/curseforge-dependency-proof-v1"}

    def install(root, proof, *, minecraft_version):
        called.append(("install", minecraft_version))
        return {"status": "INSTALLED", "runtime_verified": False}

    monkeypatch.setattr(verifier, "fetch_exact_curseforge_file_metadata", fetch)
    monkeypatch.setattr(verifier, "verify_curseforge_dependency", verify)
    monkeypatch.setattr(installer, "install_verified_fabric_dependency", install)

    result = materialize_verified_authored_sources(
        str(tmp_path), _plan(),
        minecraft_version="26.2", loader="fabric",
    )
    assert result["donor_count"] == 0
    assert result["dependency_count"] == 1
    assert called == [
        ("provider", 456, 123),
        ("integrity", "/host/cache/alien.jar", "26.2"),
        ("install", "26.2"),
    ]


def test_catalogue_hits_alone_remain_fresh_and_do_not_install(tmp_path):
    plan = _plan()
    plan.pop("requested_curseforge_dependencies")
    plan["capabilities"][0]["catalogue_evidence"] = [
        {"source_id": "curseforge:456", "title": "Aliens"}
    ]
    result = materialize_verified_authored_sources(
        str(tmp_path), plan, minecraft_version="26.2", loader="fabric",
    )
    assert result["donor_count"] == 0
    assert result["dependency_count"] == 0
