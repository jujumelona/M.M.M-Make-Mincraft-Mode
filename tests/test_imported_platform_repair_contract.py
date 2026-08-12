from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.platform_release_contract as release_contract
import minecraft_mod_ai.platform_validation_contract as validation_contract
from minecraft_mod_ai.imported_platform_repair import (
    marker_path,
    read_valid_marker,
    write_marker,
)


def _adapter(adapter_id: str = "fabric-1.20.1") -> SimpleNamespace:
    return SimpleNamespace(
        adapter_id=adapter_id,
        minecraft_version="1.20.1",
        loader="fabric",
        yarn_mappings="1.20.1+build.10",
        java_version="17",
        fabric_loader="0.16.14",
        fabric_api="0.92.6+1.20.1",
        fabric_loom="1.10.5",
        gradle="8.12",
    )


def test_import_repair_marker_is_bound_to_adapter_and_archive(tmp_path: Path) -> None:
    adapter = _adapter()
    digest = "a" * 64
    write_marker(
        tmp_path,
        adapter=adapter,
        archive_sha256=digest,
        reason="incomplete exact toolchain metadata",
    )

    accepted = read_valid_marker(
        tmp_path,
        adapter=adapter,
        archive_sha256=digest,
    )
    assert accepted is not None
    assert accepted["release_evidence"] is False
    assert accepted["authority"] == "repair-entry-only"

    assert (
        read_valid_marker(
            tmp_path,
            adapter=adapter,
            archive_sha256="b" * 64,
        )
        is None
    )
    assert (
        read_valid_marker(
            tmp_path,
            adapter=_adapter("fabric-other"),
            archive_sha256=digest,
        )
        is None
    )


def test_validator_repair_admission_requires_approved_bound_digest(tmp_path: Path) -> None:
    adapter = _adapter()
    digest = "c" * 64
    write_marker(
        tmp_path,
        adapter=adapter,
        archive_sha256=digest,
        reason="repair required",
    )

    good_module = SimpleNamespace(
        _load_complete_project_proposal=lambda root, spec, findings: SimpleNamespace(
            existing_input_sha256=digest
        )
    )
    assert validation_contract._authorized_import_repair_marker(
        good_module,
        tmp_path,
        object(),
        adapter,
    ) is not None

    wrong_bound_module = SimpleNamespace(
        _load_complete_project_proposal=lambda root, spec, findings: SimpleNamespace(
            existing_input_sha256="d" * 64
        )
    )
    assert validation_contract._authorized_import_repair_marker(
        wrong_bound_module,
        tmp_path,
        object(),
        adapter,
    ) is None

    unbound_module = SimpleNamespace(
        _load_complete_project_proposal=lambda root, spec, findings: SimpleNamespace(
            existing_input_sha256=""
        )
    )
    assert validation_contract._authorized_import_repair_marker(
        unbound_module,
        tmp_path,
        object(),
        adapter,
    ) is None


def test_release_gate_blocks_unresolved_toolchain_before_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _adapter()
    calls: list[str] = []

    class Service:
        def _approved(self, proposal, approval_hash):
            return SimpleNamespace(spec=SimpleNamespace(platform=object()))

        def _existing_dir(self, project_root):
            return Path(project_root)

        def package_release(self, project_root, proposal, approval_hash, *args, **kwargs):
            calls.append("original")
            return {"status": "PACKAGED"}

    fake_module = SimpleNamespace(MMMToolService=Service)
    release_contract.install(fake_module)
    monkeypatch.setattr(release_contract, "adapter_for_lock_values", lambda lock: expected)

    def unresolved(root):
        raise ValueError("missing exact Gradle/Yarn metadata")

    monkeypatch.setattr(release_contract, "adapter_from_project", unresolved)

    with pytest.raises(RuntimeError, match="Release is blocked"):
        Service().package_release(str(tmp_path), {}, "approval")
    assert calls == []


def test_release_gate_clears_repair_marker_only_after_exact_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _adapter()
    digest = "e" * 64
    write_marker(
        tmp_path,
        adapter=expected,
        archive_sha256=digest,
        reason="repair required",
    )
    calls: list[str] = []

    class Service:
        def _approved(self, proposal, approval_hash):
            return SimpleNamespace(spec=SimpleNamespace(platform=object()))

        def _existing_dir(self, project_root):
            return Path(project_root)

        def package_release(self, project_root, proposal, approval_hash, *args, **kwargs):
            calls.append("original")
            return {"status": "PACKAGED"}

    fake_module = SimpleNamespace(MMMToolService=Service)
    release_contract.install(fake_module)
    monkeypatch.setattr(release_contract, "adapter_for_lock_values", lambda lock: expected)
    monkeypatch.setattr(release_contract, "adapter_from_project", lambda root: expected)

    result = Service().package_release(str(tmp_path), {}, "approval")
    assert result == {"status": "PACKAGED"}
    assert calls == ["original"]
    assert not marker_path(tmp_path).exists()
