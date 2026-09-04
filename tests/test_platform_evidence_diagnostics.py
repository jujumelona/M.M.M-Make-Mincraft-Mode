from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.platform_evidence_pipeline import (
    _ShallowProject,
    _build_target_evidence,
    _inspect_project_receipt_native_detailed,
    _required_dependency_closure,
)
from minecraft_mod_ai.spec import SpecValidationError


_SHA512 = "a" * 128


def _adapter() -> SimpleNamespace:
    return SimpleNamespace(
        minecraft_version="26.2",
        loader="fabric",
        adapter_id="test-fabric-26.2",
        deterministic_module_kinds=frozenset(),
        validate=lambda: None,
    )


def _version(
    project_id: str,
    *,
    sha512: str = _SHA512,
    dependencies: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": f"{project_id}Version1",
        "version_number": "1.0.0",
        "version_type": "release",
        "status": "listed",
        "date_published": "2026-09-04T00:00:00Z",
        "game_versions": ["26.2"],
        "loaders": ["fabric"],
        "dependencies": dependencies or [],
        "files": [
            {
                "filename": f"{project_id}.jar",
                "url": f"https://cdn.modrinth.com/data/{project_id}/versions/v1/{project_id}.jar",
                "size": 1024,
                "primary": True,
                "hashes": {
                    "sha1": "b" * 40,
                    "sha512": sha512,
                },
            }
        ],
    }


class _Client:
    def __init__(
        self,
        projects: dict[str, str],
        versions: dict[str, list[dict[str, object]]],
    ) -> None:
        self.projects = projects
        self.versions = versions

    def _get_json(self, url: str, params: object | None = None) -> object:
        del params
        prefix = "https://api.modrinth.com/v2/project/"
        assert url.startswith(prefix)
        tail = url[len(prefix):]
        if tail.endswith("/version"):
            project_id = tail[: -len("/version")]
            return self.versions[project_id]
        return {
            "id": tail,
            "slug": tail,
            "title": tail,
            "license": {"id": self.projects[tail]},
        }


def test_lgpl_is_rejected_for_source_reuse_but_allowed_for_dependency_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client(
        projects={"architectury": "LGPL-3.0-only"},
        versions={"architectury": [_version("architectury")]},
    )
    adapter = _adapter()

    source = _inspect_project_receipt_native_detailed(
        client,
        "architectury",
        adapter,
        role="source_reuse",
    )
    dependency = _inspect_project_receipt_native_detailed(
        client,
        "architectury",
        adapter,
        role="required_dependency",
        parent_project_id="root",
        dependency_path=("root", "architectury"),
    )

    assert source.verified is None
    assert source.failed_gate == "source_reuse_license"
    assert dependency.verified is not None
    assert dependency.license_id == "LGPL-3.0-only"

    trace = capsys.readouterr().out
    assert '"gate":"source_reuse_license"' in trace
    assert '"gate":"dependency_license"' in trace
    assert '"license_id":"LGPL-3.0-only"' in trace


def test_bad_sha512_is_reported_as_artifact_digest_not_generic_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client(
        projects={"broken": "MIT"},
        versions={"broken": [_version("broken", sha512="")]},
    )

    inspection = _inspect_project_receipt_native_detailed(
        client,
        "broken",
        _adapter(),
        role="source_reuse",
    )

    assert inspection.verified is None
    assert inspection.failed_gate == "artifact_digest"
    assert "SHA-512" in inspection.failure_reason
    trace = capsys.readouterr().out
    assert '"gate":"artifact_digest"' in trace
    assert "failed exact-target/license/digest gates" not in trace


def test_dependency_failure_contains_full_path_gate_reason_and_license() -> None:
    client = _Client(
        projects={"dep-a": ""},
        versions={"dep-a": [_version("dep-a")]},
    )

    with pytest.raises(SpecValidationError) as raised:
        _required_dependency_closure(
            client,
            _adapter(),
            ("dep-a",),
            inspection_cache={},
            root_project_id="root-mod",
        )

    message = str(raised.value)
    assert "root-mod -> dep-a" in message
    assert "gate=dependency_license" in message
    assert "license_id=<missing>" in message
    assert "failed exact-target/license/digest gates" not in message


def test_bad_optional_reuse_dependency_does_not_kill_platform_target() -> None:
    root_version = _version(
        "root-mod",
        dependencies=[
            {
                "project_id": "dep-a",
                "version_id": None,
                "dependency_type": "required",
            }
        ],
    )
    client = _Client(
        projects={"root-mod": "MIT", "dep-a": ""},
        versions={
            "root-mod": [root_version],
            "dep-a": [_version("dep-a")],
        },
    )
    adapter = _adapter()
    shallow = _ShallowProject(
        project_id="root-mod",
        versions=frozenset({"26.2"}),
        loaders=frozenset({"fabric"}),
        downloads=10,
        modified="2026-09-04T00:00:00Z",
    )

    evidence = _build_target_evidence(
        adapter,
        queries=("space combat",),
        shallow_by_query={"space combat": (shallow,)},
        client=client,
        target_research_fn=None,
        shallow_candidate_count=1,
        discovery_disabled=False,
    )

    assert evidence.reuse_coverage == 0
    assert evidence.residual_cost == 1
    assert evidence.composition_modes == (("space combat", "custom"),)
    assert any("dependency_closure" in error for error in evidence.discovery_errors)
