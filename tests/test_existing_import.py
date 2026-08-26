from __future__ import annotations

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai import importer
from minecraft_mod_ai.importer import (
    ExistingProjectImportError,
    inspect_existing_project_archive,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline


def _write_zip(path: Path, entries: list[tuple[str, bytes | str]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)


def _jar_bytes(entries: list[tuple[str, bytes | str]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.txt",
        "safe/../../escape.txt",
        "/absolute.txt",
        r"C:\absolute.txt",
        r"safe\..\escape.txt",
    ],
)
def test_zip_slip_and_absolute_paths_are_rejected(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    _write_zip(archive_path, [(unsafe_path, b"no")])

    with pytest.raises(ExistingProjectImportError, match="path|Absolute"):
        inspect_existing_project_archive(archive_path)


def test_symlink_entries_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("project/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "../outside")

    with pytest.raises(ExistingProjectImportError, match="Symbolic links"):
        inspect_existing_project_archive(archive_path)


@pytest.mark.parametrize(
    "blocked_path",
    [
        "project/.env",
        "project/.ssh/id_rsa",
        "project/credentials.json",
        "project/private-key.pem",
        "project/.git/config",
        "project/.gradle/caches/state.bin",
        "project/build/output.jar",
        "project/run/options.txt",
    ],
)
def test_credentials_keys_vcs_and_generated_caches_are_rejected(
    tmp_path: Path,
    blocked_path: str,
) -> None:
    archive_path = tmp_path / "blocked.zip"
    _write_zip(archive_path, [(blocked_path, b"sensitive")])

    with pytest.raises(ExistingProjectImportError, match="not accepted"):
        inspect_existing_project_archive(archive_path)


def test_duplicate_and_case_colliding_paths_are_rejected(tmp_path: Path) -> None:
    exact_path = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning):
        _write_zip(exact_path, [("mod/file.txt", b"a"), ("mod/file.txt", b"b")])
    with pytest.raises(ExistingProjectImportError, match="duplicate"):
        inspect_existing_project_archive(exact_path)

    case_path = tmp_path / "case.zip"
    _write_zip(case_path, [("mod/File.txt", b"a"), ("mod/file.txt", b"b")])
    with pytest.raises(ExistingProjectImportError, match="case-colliding"):
        inspect_existing_project_archive(case_path)


def test_entry_and_size_limits_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_many = tmp_path / "too-many.zip"
    _write_zip(too_many, [("one.txt", b"1"), ("two.txt", b"2")])
    monkeypatch.setattr(importer, "MAX_ARCHIVE_ENTRIES", 1)
    with pytest.raises(ExistingProjectImportError, match="too many entries"):
        inspect_existing_project_archive(too_many)

    monkeypatch.setattr(importer, "MAX_ARCHIVE_ENTRIES", 10)
    monkeypatch.setattr(importer, "MAX_SINGLE_FILE_BYTES", 3)
    too_large = tmp_path / "too-large.zip"
    _write_zip(too_large, [("large.bin", b"1234")])
    with pytest.raises(ExistingProjectImportError, match="single-file limit"):
        inspect_existing_project_archive(too_large)

    monkeypatch.setattr(importer, "MAX_SINGLE_FILE_BYTES", 10)
    monkeypatch.setattr(importer, "MAX_TOTAL_UNCOMPRESSED_BYTES", 5)
    too_large_total = tmp_path / "too-large-total.zip"
    _write_zip(too_large_total, [("one.bin", b"123"), ("two.bin", b"456")])
    with pytest.raises(ExistingProjectImportError, match="total uncompressed"):
        inspect_existing_project_archive(too_large_total)


def test_zero_global_host_quotas_do_not_become_project_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "large-inventory.zip"
    _write_zip(
        archive_path,
        [
            (f"project/src/main/java/test/C{index}.java", "class C {}")
            for index in range(2_501)
        ],
    )
    monkeypatch.setattr(importer, "MAX_ARCHIVE_ENTRIES", 0)
    monkeypatch.setattr(importer, "MAX_TOTAL_UNCOMPRESSED_BYTES", 0)

    report = inspect_existing_project_archive(archive_path)

    assert report.file_count == 2_501
    assert len(report.source_files) == 2_501


def test_snapshot_is_stable_across_zip_order_and_container_metadata(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "second.zip"
    first_entries = [
        ("project/src/main/java/example/Main.java", b"class Main {}\n"),
        ("project/README.md", "hello\n"),
    ]
    _write_zip(first_path, first_entries)
    with zipfile.ZipFile(second_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in reversed(first_entries):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 2, 3, 4, 6))
            archive.writestr(info, content)

    first = inspect_existing_project_archive(first_path)
    second = inspect_existing_project_archive(second_path)

    assert first.source_snapshot_hash == second.source_snapshot_hash
    assert first.archive_sha256 != second.archive_sha256
    assert first.root_name == second.root_name == "project"


def test_source_project_metadata_and_assets_are_inventoried_and_safely_extracted(
    tmp_path: Path,
) -> None:
    metadata = {
        "schemaVersion": 1,
        "id": "frost_works",
        "name": "Frost Works",
        "version": "1.2.3",
        "depends": {"minecraft": "~1.20.1"},
        "mixins": ["frost_works.mixins.json"],
        "accessWidener": "frost_works.accesswidener",
    }
    archive_path = tmp_path / "source.zip"
    _write_zip(
        archive_path,
        [
            ("frost/build.gradle", "plugins {}\n"),
            ("frost/settings.gradle", "rootProject.name='frost'\n"),
            ("frost/src/main/java/example/Frost.java", "class Frost {}\n"),
            (
                "frost/src/main/resources/fabric.mod.json",
                json.dumps(metadata),
            ),
            (
                "frost/src/main/resources/frost_works.mixins.json",
                "{}",
            ),
            (
                "frost/src/main/resources/frost_works.accesswidener",
                "accessWidener v2 named\n",
            ),
            (
                "frost/src/main/resources/assets/frost_works/lang/en_us.json",
                "{}",
            ),
            (
                "frost/.minecraft_ai/world/frost_arena.world_design.json",
                "{}",
            ),
            ("frost/release/source-bundle.zip", b"not recursively opened"),
        ],
    )
    extract_root = tmp_path / "imports"

    report = inspect_existing_project_archive(
        archive_path,
        extract_root=extract_root,
    )

    assert report.input_kind == "source_project"
    assert report.has_sources is True
    assert report.has_gradle_project is True
    assert report.jar_only is False
    assert report.loader == "fabric"
    assert report.mod_id == "frost_works"
    assert report.mod_name == "Frost Works"
    assert report.mod_version == "1.2.3"
    assert report.minecraft_version == "~1.20.1"
    assert report.minecraft_versions == ("~1.20.1",)
    assert report.fabric_metadata_paths == ("src/main/resources/fabric.mod.json",)
    assert "src/main/resources/frost_works.mixins.json" in report.mixin_files
    assert "src/main/resources/frost_works.accesswidener" in report.access_widener_files
    assert report.asset_files
    assert report.world_files == (
        ".minecraft_ai/world/frost_arena.world_design.json",
    )
    assert report.release_bundles == ("release/source-bundle.zip",)
    assert report.trusted_generated_source is False
    assert report.extracted_to is not None
    extracted = Path(report.extracted_to)
    assert extracted.parent == extract_root.resolve()
    assert (extracted / "src/main/java/example/Frost.java").is_file()
    assert not (extracted / "frost").exists()
    json.dumps(report.to_dict())

    resumed = inspect_existing_project_archive(
        archive_path,
        extract_root=extract_root,
        expected_archive_sha256=report.archive_sha256,
    )
    assert resumed.extracted_to == report.extracted_to


def test_expected_archive_hash_is_checked_before_extraction(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "bound.zip"
    _write_zip(
        archive_path,
        [
            ("project/build.gradle", "plugins {}\n"),
            (
                "project/src/main/resources/fabric.mod.json",
                '{"schemaVersion":1,"id":"bound","version":"1.0.0"}',
            ),
            (
                "project/src/main/java/example/Bound.java",
                "class Bound {}\n",
            ),
        ],
    )
    extract_root = tmp_path / "imports"

    with pytest.raises(
        ExistingProjectImportError,
        match="changed after complete-plan approval",
    ):
        inspect_existing_project_archive(
            archive_path,
            extract_root=extract_root,
            expected_archive_sha256="sha256:" + ("0" * 64),
        )

    assert not extract_root.exists()


def test_tampered_completed_extraction_is_preserved_and_rebuilt(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "retry.zip"
    _write_zip(
        archive_path,
        [
            ("project/build.gradle", "plugins {}\n"),
            (
                "project/src/main/resources/fabric.mod.json",
                '{"schemaVersion":1,"id":"retry","version":"1.0.0"}',
            ),
            (
                "project/src/main/java/example/Retry.java",
                "class Retry {}\n",
            ),
        ],
    )
    extract_root = tmp_path / "imports"
    first = inspect_existing_project_archive(
        archive_path,
        extract_root=extract_root,
    )
    extracted = Path(str(first.extracted_to))
    source = extracted / "src/main/java/example/Retry.java"
    source.write_text("tampered\n", encoding="utf-8")

    second = inspect_existing_project_archive(
        archive_path,
        extract_root=extract_root,
        expected_archive_sha256=first.archive_sha256,
    )

    assert second.extracted_to == first.extracted_to
    assert source.read_text(encoding="utf-8") == "class Retry {}\n"
    preserved = extracted.with_name(extracted.name + ".incomplete-1")
    assert (preserved / "src/main/java/example/Retry.java").read_text(
        encoding="utf-8"
    ) == "tampered\n"


def test_jar_only_mod_metadata_is_inventoried_without_execution(tmp_path: Path) -> None:
    jar = _jar_bytes(
        [
            (
                "fabric.mod.json",
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "id": "jar_mod",
                        "name": "JAR Mod",
                        "version": "2.0.0",
                        "depends": {"minecraft": ["1.20.1", "1.20.2"]},
                    }
                ),
            ),
            ("assets/jar_mod/lang/en_us.json", "{}"),
        ]
    )
    archive_path = tmp_path / "jar-only.zip"
    _write_zip(archive_path, [("release/jar_mod.jar", jar)])

    report = inspect_existing_project_archive(archive_path)

    assert report.input_kind == "jar_only"
    assert report.jar_only is True
    assert report.mod_id == "jar_mod"
    assert report.minecraft_versions == ("1.20.1", "1.20.2")
    assert report.fabric_metadata_paths == ("jar_mod.jar!/fabric.mod.json",)
    assert report.jar_files == ("jar_mod.jar",)
    assert report.asset_files == ("jar_mod.jar!/assets/jar_mod/lang/en_us.json",)


def test_nested_archives_are_spooled_instead_of_retained_as_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jar = _jar_bytes(
        [
            (
                "fabric.mod.json",
                '{"schemaVersion":1,"id":"spooled","version":"1.0.0"}',
            ),
            ("payload.bin", b"x" * (3 * 1024 * 1024)),
        ]
    )
    archive_path = tmp_path / "spooled.zip"
    _write_zip(archive_path, [("release/spooled.jar", jar)])
    observed: list[bool] = []
    original = importer._inspect_nested_jar

    def recording(path, data, warnings):
        observed.append(isinstance(data, Path))
        assert isinstance(data, Path)
        assert data.is_file()
        return original(path, data, warnings)

    monkeypatch.setattr(importer, "_inspect_nested_jar", recording)

    report = inspect_existing_project_archive(archive_path)

    assert report.mod_id == "spooled"
    assert observed == [True]


def test_embedded_approved_proposal_is_validated_but_not_trusted_as_evidence(
    tmp_path: Path,
) -> None:
    awaiting = MinecraftModPipeline().plan("Create a frost item")
    approved = awaiting.approve(awaiting.approval_hash)
    archive_path = tmp_path / "generated-source.zip"
    _write_zip(
        archive_path,
        [
            (
                "project/.minecraft_ai/proposal.approved.json",
                json.dumps(approved.to_dict()),
            ),
            ("project/src/main/java/example/Main.java", "class Main {}\n"),
        ],
    )

    report = inspect_existing_project_archive(archive_path)

    assert report.embedded_proposal_status == "approved_valid"
    assert report.embedded_approval_hash == approved.approval_hash
    assert report.trusted_generated_source is False
    assert any("not independently attested" in warning for warning in report.warnings)


def test_tampered_embedded_proposal_is_reported_invalid(tmp_path: Path) -> None:
    awaiting = MinecraftModPipeline().plan("Create a frost item")
    approved = awaiting.approve(awaiting.approval_hash).to_dict()
    approved["requested_prompt"] = "tampered after approval"
    archive_path = tmp_path / "tampered.zip"
    _write_zip(
        archive_path,
        [
            (
                "project/.minecraft_ai/proposal.approved.json",
                json.dumps(approved),
            ),
            ("project/src/main/java/example/Main.java", "class Main {}\n"),
        ],
    )

    report = inspect_existing_project_archive(archive_path)

    assert report.embedded_proposal_status == "invalid"
    assert report.trusted_generated_source is False
    assert any("approval_hash does not match" in warning for warning in report.warnings)
