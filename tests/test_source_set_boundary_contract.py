from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.source_set_boundary_contract import (
    SourceSetBoundaryError,
    assert_server_safe_source_sets,
    source_set_boundary_errors,
)
from minecraft_mod_ai.source_set_boundary_installation import install


def _write(root, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_common_source_cannot_import_project_client_source(tmp_path):
    _write(
        tmp_path,
        "src/client/java/example/client/ClientHooks.java",
        "package example.client; public final class ClientHooks {}",
    )
    _write(
        tmp_path,
        "src/main/java/example/CommonInit.java",
        "package example; import example.client.ClientHooks; public final class CommonInit { ClientHooks hook; }",
    )

    errors = source_set_boundary_errors(tmp_path)

    assert any("src/main/java/example/CommonInit.java" in item for item in errors)
    assert any("example.client.ClientHooks" in item for item in errors)
    with pytest.raises(SourceSetBoundaryError, match="common/server code must not reach client-only"):
        assert_server_safe_source_sets(tmp_path)


def test_server_source_cannot_import_external_minecraft_client_api(tmp_path):
    _write(
        tmp_path,
        "src/server/java/example/ServerInit.java",
        "package example; import net.minecraft.client.MinecraftClient; public final class ServerInit {}",
    )

    errors = source_set_boundary_errors(tmp_path)

    assert len(errors) == 1
    assert "server source imports client-only dependency net.minecraft.client.MinecraftClient" in errors[0]


def test_client_source_can_depend_on_common_source(tmp_path):
    _write(
        tmp_path,
        "src/main/java/example/CommonApi.java",
        "package example; public final class CommonApi {}",
    )
    _write(
        tmp_path,
        "src/client/java/example/client/ClientInit.java",
        "package example.client; import example.CommonApi; public final class ClientInit { CommonApi api; }",
    )

    assert source_set_boundary_errors(tmp_path) == ()
    assert_server_safe_source_sets(tmp_path)


def test_comments_and_literals_do_not_create_false_client_edges(tmp_path):
    _write(
        tmp_path,
        "src/client/java/example/client/ClientHooks.java",
        "package example.client; public final class ClientHooks {}",
    )
    _write(
        tmp_path,
        "src/main/java/example/CommonInit.java",
        '''package example;
public final class CommonInit {
  // import example.client.ClientHooks;
  String text = "example.client.ClientHooks";
  /* net.minecraft.client.MinecraftClient */
}
''',
    )

    assert source_set_boundary_errors(tmp_path) == ()


def test_same_package_client_type_is_rejected_without_import(tmp_path):
    _write(
        tmp_path,
        "src/client/java/example/ClientHooks.java",
        "package example; public final class ClientHooks {}",
    )
    _write(
        tmp_path,
        "src/main/java/example/CommonInit.java",
        "package example; public final class CommonInit { ClientHooks hooks; }",
    )

    errors = source_set_boundary_errors(tmp_path)

    assert any("same-package client source ClientHooks" in item for item in errors)


def test_installed_guard_fails_before_underlying_jdt_is_called(tmp_path):
    _write(
        tmp_path,
        "src/main/java/example/CommonInit.java",
        "package example; import net.minecraft.client.MinecraftClient; public final class CommonInit {}",
    )
    calls: list[str] = []

    class FakeJDTError(RuntimeError):
        pass

    class FakeService:
        def diagnostics(
            self,
            project_root,
            *,
            relative_files=None,
            timeout_seconds=60,
        ):
            calls.append(str(project_root))
            return {"diagnostics": []}

    module = SimpleNamespace(
        JavaLanguageService=FakeService,
        JDTLanguageServerError=FakeJDTError,
    )
    install(module)

    with pytest.raises(FakeJDTError, match="Java source-set preflight failed"):
        FakeService().diagnostics(tmp_path)
    assert calls == []


def test_package_runtime_exposes_source_set_guard_on_java_diagnostics():
    from minecraft_mod_ai import java_lsp

    assert getattr(
        java_lsp.JavaLanguageService.diagnostics,
        "__mmm_source_set_boundary__",
        False,
    ) is True
