import inspect
import zipfile
from types import SimpleNamespace

from minecraft_mod_ai import mcp_tools


def test_release_contains_rebuildable_source_and_evidence_without_machine_outputs(tmp_path, monkeypatch):
    root = tmp_path / "project"
    wanted = ["src/main/java/demo/Token.java", "build.gradle", "gradlew",
              "gradle/wrapper/gradle-wrapper.jar", ".minecraft_ai/requirement-coverage.json",
              "build/gametest-report.xml", "build/mmm-gametest-attestation.xml",
              "src/main/resources/assets/demo/build/model.json"]
    unwanted = ["build/classes/Token.class", "bin/main/Token.class", ".classpath",
                ".project", ".settings/prefs", ".git/config", ".gradle/cache",
                "run/server.properties", ".minecraft_ai/trajectory-memory/index.sqlite3",
                "child/build/classes/Token.class", "build/libs/demo.jar"]
    for name in wanted + unwanted:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence or source", encoding="utf-8")
    report = SimpleNamespace(passed=True, to_dict=lambda: {"status": "PASS"})
    proposal = SimpleNamespace(spec=object(), calculate_hash=lambda: "sha256:proposal")
    service = mcp_tools.MMMToolService(workspace_root=tmp_path)
    monkeypatch.setattr(service, "_approved", lambda *a: proposal)
    monkeypatch.setattr(service.broker, "authorize", lambda *a: None)
    monkeypatch.setattr(mcp_tools, "approved_request", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_tools, "ScalableProjectValidator",
                        lambda **kw: SimpleNamespace(validate=lambda *a: report))
    monkeypatch.setattr(mcp_tools, "validate_jar", lambda *a: report)
    # An output inside the project must not include itself or old release archives.
    old = root / "releases/old.zip"
    old.parent.mkdir()
    old.write_bytes(b"old")
    result = inspect.unwrap(type(service).package_release)(service, str(root), {}, "approved",
        output_zip="project/releases/new.zip", jar_path=str(root / "build/libs/demo.jar"))
    with zipfile.ZipFile(result["release_zip"]) as archive:
        names = set(archive.namelist())
        assert {"source/" + p for p in wanted} <= names
        assert not {"source/" + p for p in unwanted} & names
        assert not any("releases/" in p for p in names)
        assert "binary/demo.jar" in names

def test_verified_release_packaging_reuses_completed_validation(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    source = root / "src/main/java/demo/Token.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo; final class Token {}\n", encoding="utf-8")
    jar = root / "build/libs/demo.jar"
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"verified-jar")

    proposal = SimpleNamespace(spec=object(), calculate_hash=lambda: "sha256:proposal")
    service = mcp_tools.MMMToolService(workspace_root=tmp_path)
    monkeypatch.setattr(service, "_approved", lambda *a: proposal)
    monkeypatch.setattr(service.broker, "authorize", lambda *a: None)
    monkeypatch.setattr(mcp_tools, "approved_request", lambda *a, **kw: None)
    monkeypatch.setattr(
        mcp_tools,
        "ScalableProjectValidator",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("verified packaging must not rerun source validation")
        ),
    )
    monkeypatch.setattr(
        mcp_tools,
        "validate_jar",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("verified packaging must not rerun JAR validation")
        ),
    )

    source_validation = {"status": "PASS", "checks_run": 1, "findings": []}
    jar_validation = {"status": "PASS", "checks_run": 1, "findings": []}
    attachment = tmp_path / "resource-pack.zip"
    attachment.write_bytes(b"resource-pack")
    attachment_sha = mcp_tools._sha256(attachment)
    result = service._package_release_from_verified_evidence(
        str(root),
        {},
        "approved",
        output_zip="verified-release.zip",
        jar_path=str(jar),
        source_validation=source_validation,
        jar_validation=jar_validation,
        expected_jar_sha256=mcp_tools._sha256(jar),
        additional_artifacts={
            "generated-resource-pack.zip": {
                "path": str(attachment),
                "sha256": attachment_sha,
            }
        },
        manifest_provenance={
            "proposal_scope": "complete",
        },
    )

    with zipfile.ZipFile(result["release_zip"]) as archive:
        names = set(archive.namelist())
        manifest = __import__("json").loads(
            archive.read("release-manifest.json").decode("utf-8")
        )
        assert archive.read("additional/generated-resource-pack.zip") == b"resource-pack"
    assert "binary/demo.jar" in names
    assert "release-manifest.json" in names
    assert manifest["additional_artifacts"] == {
        "generated-resource-pack.zip": attachment_sha
    }
    assert manifest["proposal_scope"] == "complete"

