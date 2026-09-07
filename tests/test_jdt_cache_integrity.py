from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import java_lsp
from minecraft_mod_ai import validation_execution_contract as validation


def _project(root: Path) -> Path:
    source = root / "src/main/java/demo/Main.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo; public final class Main {}\n", encoding="utf-8")
    (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    (root / "settings.gradle").write_text("rootProject.name='demo'\n", encoding="utf-8")
    (root / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx1g\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _clear_jdt_cache() -> None:
    with validation._CACHE_LOCK:
        validation._JDT_RESULTS.clear()


def _module(service_cls, *, java_files=java_lsp._java_files):
    return SimpleNamespace(
        JavaLanguageService=service_cls,
        _java_files=java_files,
        JDTLanguageServerError=java_lsp.JDTLanguageServerError,
    )


class _Service:
    def __init__(
        self,
        *,
        command: tuple[str, ...] = ("jdtls",),
        quiet: float = 0.0,
        page_files: int = 128,
        page_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.command = list(command)
        self.diagnostic_quiet_seconds = quiet
        self.diagnostic_page_max_files = page_files
        self.diagnostic_page_max_source_bytes = page_bytes
        self.calls = 0

    def diagnostics(
        self,
        project_root: str | Path,
        *,
        relative_files=None,
        timeout_seconds: int = 60,
    ):
        self.calls += 1
        root = Path(project_root).resolve()
        files = tuple(relative_files or ())
        return {
            "schema_version": "probe",
            "root": str(root),
            "files": files,
            "timeout_seconds": timeout_seconds,
        }


def test_same_snapshot_and_service_profile_reuses_jdt_result(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")

    class Service(_Service):
        pass

    validation._install_jdt_cache(_module(Service))
    service = Service(quiet=0.25)
    first = service.diagnostics(root, timeout_seconds=30)
    second = service.diagnostics(root, timeout_seconds=30)

    assert first == second
    assert service.calls == 1


def test_jdt_cache_does_not_cross_service_semantics(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")

    class Service(_Service):
        pass

    validation._install_jdt_cache(_module(Service))

    base = Service(command=("jdtls-a",), quiet=0.0)
    quiet = Service(command=("jdtls-a",), quiet=1.0)
    command = Service(command=("jdtls-b",), quiet=0.0)

    base.diagnostics(root, timeout_seconds=30)
    quiet.diagnostics(root, timeout_seconds=30)
    command.diagnostics(root, timeout_seconds=30)
    base.diagnostics(root, timeout_seconds=31)

    assert base.calls == 2
    assert quiet.calls == 1
    assert command.calls == 1


def test_unsafe_relative_file_is_rejected_before_original_diagnostics(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    outside = tmp_path / "Outside.java"
    outside.write_text("class Outside {}\n", encoding="utf-8")

    class Service(_Service):
        pass

    validation._install_jdt_cache(_module(Service))
    service = Service()

    with pytest.raises(ValueError, match="canonical|escaped"):
        service.diagnostics(root, relative_files=("../Outside.java",))
    assert service.calls == 0


def test_jdt_result_is_rejected_when_source_bytes_change_during_run(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    source = root / "src/main/java/demo/Main.java"

    class Service(_Service):
        def diagnostics(self, project_root, *, relative_files=None, timeout_seconds=60):
            result = super().diagnostics(
                project_root,
                relative_files=relative_files,
                timeout_seconds=timeout_seconds,
            )
            source.write_text(
                "package demo; public final class Main { int changed; }\n",
                encoding="utf-8",
            )
            return result

    validation._install_jdt_cache(_module(Service))
    service = Service()

    with pytest.raises(java_lsp.JDTLanguageServerError, match="inputs changed"):
        service.diagnostics(root)
    assert service.calls == 1
    with validation._CACHE_LOCK:
        assert not validation._JDT_RESULTS


def test_full_scope_jdt_rejects_java_file_set_change_during_run(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")

    class Service(_Service):
        def diagnostics(self, project_root, *, relative_files=None, timeout_seconds=60):
            result = super().diagnostics(
                project_root,
                relative_files=relative_files,
                timeout_seconds=timeout_seconds,
            )
            added = Path(project_root) / "src/main/java/demo/Added.java"
            added.write_text("package demo; final class Added {}\n", encoding="utf-8")
            return result

    validation._install_jdt_cache(_module(Service))
    service = Service()

    with pytest.raises(java_lsp.JDTLanguageServerError, match="source set changed"):
        service.diagnostics(root)
    assert service.calls == 1
    with validation._CACHE_LOCK:
        assert not validation._JDT_RESULTS
