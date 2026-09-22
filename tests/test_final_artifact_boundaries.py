from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai import validator
from minecraft_mod_ai.final_artifact import (
    FinalArtifactError,
    _project_root,
    _read_jar_metadata,
    _write_json_receipt,
    append_github_outputs,
    build_authored_design_coverage_receipt,
    build_debug_fixture_coverage_receipt,
    sha256_file,
    verify_debug_fixture_source,
    write_downloadable_bundle,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.validator import validate_jar


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")


def _debug_coverage_inputs() -> dict[str, object]:
    return {
        "proposal_hash": "sha256:" + "1" * 64,
        "acceptance_tests": (
            "The debug fixture compiles against the selected target API.",
            "The debug fixture passes the deterministic verification pipeline.",
        ),
        "artifact_sha256": "sha256:" + "2" * 64,
        "source_validation": {
            "status": "PASS",
            "checks_run": 4,
            "findings": [],
        },
        "build_report": {"status": "PASS"},
        "jar_validation": {
            "status": "PASS",
            "checks_run": 6,
            "findings": [],
        },
        "gametest_passed": True,
        "unresolved_gates": (),
        "observable_acceptance": {
            "schema_version": "mmm/debug-source-acceptance-v1",
            "status": "PASS",
            "source_sha256": "sha256:" + "3" * 64,
            "findings": [],
        },
    }



def _authored_coverage_inputs() -> dict[str, object]:
    text = "# Economy\nCredits, trading, ship upgrades, and colonies.\n"
    raw = text.encode("utf-8")
    unit_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema_version": "mmm/authored-execution-manifest-v2",
        "source_text_sha256": unit_sha256,
        "source_bytes": len(raw),
        "unit_count": 1,
        "policy": "host_exact_task_queue_no_coder_file_planning",
        "units": [
            {
                "module_id": "authored_feature_001",
                "path": "src/main/java/example/AuthoredFeature001.java",
                "symbol": "AuthoredFeature001",
                "index": 1,
                "start_byte": 0,
                "end_byte": len(raw),
                "text_sha256": unit_sha256,
                "provides": "authored_feature_001_ready",
                "section": "Economy",
            }
        ],
        "entrypoint": {
            "owner": "host_scaffold",
            "path": "src/main/java/example/ExampleMod.java",
            "symbol": "ExampleMod",
            "feature_symbols": ["AuthoredFeature001"],
        },
    }
    manifest["manifest_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    prompt = "Build the saved space economy design."
    return {
        "proposal_hash": "sha256:" + "1" * 64,
        "requested_prompt": prompt,
        "authored_plan": {
            "schema_version": "mmm/authored-plan-v1",
            "requested_prompt": prompt,
            "text": text,
            "existing_input_sha256": "",
            "media_paths": [],
        },
        "authored_manifest": manifest,
        "artifact_sha256": "sha256:" + "2" * 64,
        "source_validation": {
            "status": "PASS",
            "checks_run": 12,
            "findings": [],
        },
        "build_report": {"status": "PASS"},
        "jar_validation": {
            "status": "PASS",
            "checks_run": 9,
            "findings": [],
        },
        "gametest_passed": True,
        "unresolved_gates": (),
    }


def test_saved_authored_coverage_binds_exact_design_and_verification() -> None:
    inputs = _authored_coverage_inputs()

    receipt = build_authored_design_coverage_receipt(**inputs)

    assert receipt["status"] == "PASS"
    assert receipt["coverage_mode"] == "saved_authored_design"
    assert receipt["verification"] == {
        "authored_design_binding": True,
        "source_validation": True,
        "build": True,
        "jar_validation": True,
        "gametest": True,
    }
    assert receipt["authored_design_binding"]["unit_count"] == 1
    assert [row["statement"] for row in receipt["requirements"]] == [
        inputs["requested_prompt"],
        inputs["authored_plan"]["text"].strip(),
    ]
    assert all(row["status"] == "PASS" for row in receipt["requirements"])


def test_saved_authored_coverage_fails_closed_when_manifest_text_is_tampered() -> None:
    inputs = _authored_coverage_inputs()
    tampered = json.loads(json.dumps(inputs["authored_manifest"]))
    tampered["units"][0]["text_sha256"] = "sha256:" + "f" * 64

    receipt = build_authored_design_coverage_receipt(
        **{**inputs, "authored_manifest": tampered}
    )

    assert receipt["status"] == "BLOCKED"
    assert receipt["verification"]["authored_design_binding"] is False
    assert any("text hash does not match" in finding for finding in receipt["findings"])
    assert all(row["status"] == "BLOCKED" for row in receipt["requirements"])

def test_debug_fixture_coverage_requires_all_real_verification_gates() -> None:
    inputs = _debug_coverage_inputs()
    receipt = build_debug_fixture_coverage_receipt(**inputs)
    assert receipt["status"] == "PASS"
    assert receipt["coverage_mode"] == "debug_fixture"
    assert receipt["verification"] == {
        "source_validation": True,
        "build": True,
        "jar_validation": True,
        "gametest": True,
        "observable_acceptance": True,
    }
    assert all(item["status"] == "PASS" for item in receipt["requirements"])

    for field, value in (
        ("source_validation", {"status": "FAIL", "checks_run": 4, "findings": []}),
        ("build_report", {"status": "FAIL"}),
        ("jar_validation", {"status": "FAIL", "checks_run": 6, "findings": []}),
        ("gametest_passed", False),
        (
            "observable_acceptance",
            {
                "schema_version": "mmm/debug-source-acceptance-v1",
                "status": "BLOCKED",
                "source_sha256": "sha256:" + "3" * 64,
                "findings": ["missing item registration"],
            },
        ),
        ("unresolved_gates", ("required-gate:debug_token:target_compile:missing",)),
    ):
        blocked = build_debug_fixture_coverage_receipt(
            **{**inputs, field: value}
        )
        assert blocked["status"] == "BLOCKED"


def _debug_source_contract() -> dict[str, object]:
    return {
        "schema_version": "mmm/debug-source-contract-v1",
        "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
        "identifier": "debug_token",
        "binding_field": "DEBUG_TOKEN",
        "semantic_kind": "item",
        "required_host_symbol_keys": [
            "register_item",
            "builtin_item_registry",
            "registries_item",
            "resource_key_create",
            "identifier_factory",
            "item_set_id",
        ],
        "required_host_symbol_specs": {
            "item_set_id": {
                "owner": "net.minecraft.world.item.Item$Properties",
                "name": "setId",
                "descriptor": (
                    "(Lnet/minecraft/resources/ResourceKey;)"
                    "Lnet/minecraft/world/item/Item$Properties;"
                ),
                "kind": "method",
                "static": False,
                "side": "common",
                "namespace": "minecraft",
            }
        },
        "forbidden_lifecycle_symbols": [
            "ModInitializer",
            "onInitialize",
        ],
    }


def _debug_host_facts() -> str:
    return json.dumps(
        {
            "host_revision": "sha256:" + "4" * 64,
            "api_symbols": {
                "register_item": {
                    "owner": "net.minecraft.core.Registry",
                    "name": "register",
                    "descriptor": (
                        "(Lnet/minecraft/core/Registry;"
                        "Lnet/minecraft/resources/ResourceKey;"
                        "Ljava/lang/Object;)Ljava/lang/Object;"
                    ),
                    "kind": "method",
                },
                "builtin_item_registry": {
                    "owner": "net.minecraft.core.registries.BuiltInRegistries",
                    "name": "ITEM",
                    "kind": "field",
                },
                "registries_item": {
                    "owner": "net.minecraft.core.registries.Registries",
                    "name": "ITEM",
                    "kind": "field",
                },
                "resource_key_create": {
                    "owner": "net.minecraft.resources.ResourceKey",
                    "name": "create",
                    "kind": "method",
                },
                "identifier_factory": {
                    "owner": "net.minecraft.resources.Identifier",
                    "name": "fromNamespaceAndPath",
                    "kind": "method",
                },
            },
        }
    )


def test_debug_fixture_source_acceptance_binds_real_host_item_symbols(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package dev.mmm.debugfixture;

import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.item.Item;

public final class DebugToken {
    public static final ResourceKey<Item> KEY = ResourceKey.create(
        Registries.ITEM,
        Identifier.fromNamespaceAndPath("mmm_debug_fixture", "debug_token")
    );
    public static final Item DEBUG_TOKEN = Registry.register(
        BuiltInRegistries.ITEM,
        KEY,
        new Item(new Item.Properties().setId(KEY))
    );
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    receipt = verify_debug_fixture_source(
        tmp_path,
        source_contract=_debug_source_contract(),
        host_facts_json=_debug_host_facts(),
    )

    assert receipt["status"] == "PASS"
    assert receipt["identifier_present"] is True
    assert receipt["binding_assignment_proven"] is True
    assert receipt["lifecycle_clear"] is True
    assert all(receipt["symbol_results"].values())


def test_debug_fixture_source_acceptance_rejects_missing_keyed_item_set_id(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package dev.mmm.debugfixture;

import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.item.Item;

public final class DebugToken {
    public static final ResourceKey<Item> KEY = ResourceKey.create(
        Registries.ITEM,
        Identifier.fromNamespaceAndPath("mmm_debug_fixture", "debug_token")
    );
    public static final Item DEBUG_TOKEN = Registry.register(
        BuiltInRegistries.ITEM,
        KEY,
        new Item(new Item.Properties())
    );
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    receipt = verify_debug_fixture_source(
        tmp_path,
        source_contract=_debug_source_contract(),
        host_facts_json=_debug_host_facts(),
    )

    assert receipt["status"] == "BLOCKED"
    assert receipt["symbol_results"]["item_set_id"] is False
    assert any("item_set_id" in item for item in receipt["findings"])
    assert receipt["diagnostics"]
    assert receipt["diagnostics"][0]["path"] == (
        "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    )
    assert any(
        "item_set_id" in item["message"]
        for item in receipt["diagnostics"]
    )


def test_debug_fixture_source_acceptance_rejects_unbound_registry_result(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package dev.mmm.debugfixture;

import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.item.Item;

public final class DebugToken {
    public static final ResourceKey<Item> KEY = ResourceKey.create(
        Registries.ITEM,
        Identifier.fromNamespaceAndPath("mmm_debug_fixture", "debug_token")
    );
    public static final Item DEBUG_TOKEN =
        new Item(new Item.Properties().setId(KEY));
    public static final Item OTHER = Registry.register(
        BuiltInRegistries.ITEM,
        KEY,
        DEBUG_TOKEN
    );
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    receipt = verify_debug_fixture_source(
        tmp_path,
        source_contract=_debug_source_contract(),
        host_facts_json=_debug_host_facts(),
    )

    assert receipt["status"] == "BLOCKED"
    assert receipt["identifier_present"] is True
    assert receipt["binding_assignment_proven"] is False
    assert all(receipt["symbol_results"].values())
    assert any("binding field" in item for item in receipt["findings"])


def test_debug_fixture_source_acceptance_rejects_compile_only_placeholder(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package dev.mmm.debugfixture;

public final class DebugToken {
    private static final String TOKEN = "debug_token_fixture";

    public static String getToken() {
        return TOKEN;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    receipt = verify_debug_fixture_source(
        tmp_path,
        source_contract=_debug_source_contract(),
        host_facts_json=_debug_host_facts(),
    )

    assert receipt["status"] == "BLOCKED"
    assert receipt["identifier_present"] is False
    assert set(receipt["symbol_results"].values()) == {False}
    assert any("required host symbol" in item for item in receipt["findings"])


def test_sha256_rejects_direct_and_parent_symlink_aliases(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    artifact = real_dir / "artifact.jar"
    artifact.write_bytes(b"artifact")
    assert sha256_file(artifact).startswith("sha256:")

    direct = tmp_path / "direct.jar"
    _symlink(direct, artifact)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        sha256_file(direct)

    alias = tmp_path / "alias"
    _symlink(alias, real_dir, directory=True)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        sha256_file(alias / "artifact.jar")


def test_project_root_rejects_parent_symlink_alias(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    project = real_parent / "project"
    project.mkdir(parents=True)
    alias_parent = tmp_path / "alias-parent"
    _symlink(alias_parent, real_parent, directory=True)

    with pytest.raises(FinalArtifactError, match="symbolic links"):
        _project_root(alias_parent / "project")


def test_receipt_write_rejects_symlink_target_and_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_receipt = real_dir / "receipt.json"
    real_receipt.write_text("{}\n", encoding="utf-8")

    direct = tmp_path / "receipt-link.json"
    _symlink(direct, real_receipt)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        _write_json_receipt(direct, {"status": "PASS"})

    alias = tmp_path / "alias"
    _symlink(alias, real_dir, directory=True)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        _write_json_receipt(alias / "new.json", {"status": "PASS"})


def test_github_output_rejects_unsafe_artifact_name(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "mod.jar").write_bytes(b"jar")
    output = tmp_path / "github-output.txt"

    with pytest.raises(FinalArtifactError, match="artifact name"):
        append_github_outputs(
            output,
            {
                "path": str(bundle_dir),
                "artifact": "../mod.jar",
                "artifact_sha256": "sha256:" + "0" * 64,
            },
        )


def test_github_output_rehashes_artifact_and_requires_receipt(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    artifact = bundle_dir / "mod.jar"
    artifact.write_bytes(b"jar")
    digest = sha256_file(artifact)
    output = tmp_path / "github-output.txt"

    with pytest.raises(FinalArtifactError, match="SHA-256"):
        append_github_outputs(
            output,
            {
                "path": str(bundle_dir),
                "artifact": "mod.jar",
                "artifact_sha256": "sha256:" + "0" * 64,
            },
        )

    with pytest.raises(FinalArtifactError, match="receipt"):
        append_github_outputs(
            output,
            {
                "path": str(bundle_dir),
                "artifact": "mod.jar",
                "artifact_sha256": digest,
            },
        )

    (bundle_dir / "artifact-receipt.json").write_text("{}\n", encoding="utf-8")
    append_github_outputs(
        output,
        {
            "path": str(bundle_dir),
            "artifact": "mod.jar",
            "artifact_sha256": digest,
        },
    )
    values = dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert values["artifact_sha256"] == digest
    assert Path(values["artifact_path"]) == artifact.resolve()
    assert Path(values["receipt_path"]) == (bundle_dir / "artifact-receipt.json").resolve()


@pytest.mark.parametrize("bad_receipt", ["build", "coverage", "runtime"])
def test_bundle_rejects_receipt_sha_mismatch(
    tmp_path: Path,
    bad_receipt: str,
) -> None:
    artifact = tmp_path / "mod.jar"
    artifact.write_bytes(b"jar")
    digest = sha256_file(artifact)
    wrong = "sha256:" + "0" * 64
    artifact_receipt = {
        "status": "PASS",
        "artifact_path": str(artifact),
        "sha256": digest,
    }
    build = {"status": "PASS", "artifact_sha256": digest}
    coverage = {"status": "PASS", "artifact_sha256": digest}
    runtime = {"status": "NOT_REQUIRED", "artifact_sha256": digest}
    if bad_receipt == "build":
        build["artifact_sha256"] = wrong
    elif bad_receipt == "coverage":
        coverage["artifact_sha256"] = wrong
    else:
        runtime["artifact_sha256"] = wrong

    with pytest.raises(FinalArtifactError, match="final artifact"):
        write_downloadable_bundle(
            tmp_path / "bundle",
            artifact_receipt=artifact_receipt,
            requirement_coverage=coverage,
            reuse_manifest={},
            build_receipt=build,
            runtime_receipt=runtime,
        )
    assert not (tmp_path / "bundle").exists()


def test_final_jar_metadata_rejects_windows_drive_entry(tmp_path: Path) -> None:
    jar = tmp_path / "unsafe.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr(
            "fabric.mod.json",
            '{"id":"demo_mod","depends":{"minecraft":"1.21.1"}}',
        )
        archive.writestr("C:/outside.txt", "unsafe")

    with pytest.raises(FinalArtifactError, match="unsafe path"):
        _read_jar_metadata(
            jar,
            loader="fabric",
            metadata_path="fabric.mod.json",
        )


def test_validate_jar_rejects_direct_symlink_before_archive_parsing(tmp_path: Path) -> None:
    spec = MinecraftModPipeline().plan("Create a frost item").spec
    target = tmp_path / "target.jar"
    target.write_bytes(b"not-even-a-zip")
    link = tmp_path / "linked.jar"
    _symlink(link, target)

    report = validate_jar(link, spec)
    assert report.status == "FAIL"
    assert report.findings
    assert report.findings[0].code == "JAR_MISSING"


def test_project_validator_has_one_canonical_boss_validator() -> None:
    source = inspect.getsource(validator.ProjectValidator)
    assert source.count("def _validate_boss(") == 1
