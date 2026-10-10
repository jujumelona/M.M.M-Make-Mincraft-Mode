"""The audit permits only the immutable HOST catalog's exact data-only push."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "mmm_debug_repo_audit", ROOT / ".github/scripts/debug_repo_audit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_host_catalog_sync_is_data_only() -> None:
    audit = _audit_module()
    relative = Path(".github/workflows/refresh-host-version-catalog.yml")
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert audit._is_scoped_host_catalog_publisher(relative, source)

    # Adding any new staged path, branch push, or broad git add invalidates
    # the exception rather than silently extending publication permissions.
    variants = (
        source.replace(
            "git push origin HEAD:main",
            "git push --force origin HEAD:main",
        ),
        source.replace(
            "git add minecraft_mod_ai/data/host_version_catalog.json "
            "minecraft_mod_ai/data/official_version_evidence.json",
            "git add .",
        ),
        source.replace(
            "git push origin HEAD:main",
            "git push origin HEAD:main\\n          git add README.md",
        ),
        source.replace(
            "python -m minecraft_mod_ai.populate_version_artifact_rules",
            "python -m minecraft_mod_ai.populate_version_artifact_rules_broken",
        ),
    )
    for altered in variants:
        assert not audit._is_scoped_host_catalog_publisher(relative, altered)


def test_other_workflows_cannot_publish_code() -> None:
    audit = _audit_module()
    source = (
        ROOT / ".github/workflows/refresh-host-version-catalog.yml"
    ).read_text(encoding="utf-8")
    assert not audit._is_scoped_host_catalog_publisher(
        Path(".github/workflows/main-ci.yml"), source
    )
    assert not audit.audit_workflow_definitions()
