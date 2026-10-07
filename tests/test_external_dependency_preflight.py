from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import api, complete_preflight_contract


def test_bound_target_toolchain_preflight_resolves_java_gradle_and_jdt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_jdk = tmp_path / "jdk25"
    owner_jdk = tmp_path / "owner25"
    for root in (project_jdk, owner_jdk):
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "java").write_text("", encoding="utf-8")
        (root / "bin" / "javac").write_text("", encoding="utf-8")

    adapter = SimpleNamespace(
        java_version="25",
        gradle="9.5.1",
        gradle_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.platform_catalog.adapter_for_lock_values",
        lambda _platform: adapter,
    )

    resolved: list[tuple[int, bool]] = []

    def resolve_java(major: int, *, require_compiler: bool = False):
        resolved.append((major, require_compiler))
        return project_jdk if len(resolved) == 1 else owner_jdk

    monkeypatch.setattr(
        "minecraft_mod_ai.java_lsp._resolve_project_java_home",
        resolve_java,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.java_lsp._java_major_version",
        lambda _home: 25,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.java_lsp._parse_java_major",
        lambda value: int(value),
    )

    gradle_calls: list[tuple[str, str, int]] = []

    class FakeGradleRunner:
        def __init__(self, _cache, *, download_timeout_seconds):
            assert download_timeout_seconds == 300

        def ensure_gradle(self, version, sha256, *, lock_timeout_seconds):
            gradle_calls.append((version, sha256, lock_timeout_seconds))
            return tmp_path / "gradle"

    monkeypatch.setattr(
        "minecraft_mod_ai.runner.GradleRunner",
        FakeGradleRunner,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.runner.production_gradle_cache_dir",
        lambda: tmp_path / "gradle-cache",
    )

    owner_calls: list[tuple[int, Path]] = []

    def owner_command(workspace, **kwargs):
        owner_calls.append((kwargs["required_major"], Path(kwargs["java_home"])))
        return [str(owner_jdk / "bin" / "java")]

    monkeypatch.setattr(
        "minecraft_mod_ai.jvm_owner_bootstrap.owner_command",
        owner_command,
    )

    complete_preflight_contract.validate_platform_toolchain_preflight(
        object(),
        SimpleNamespace(source_only=False, run_jdt=True),
    )

    assert resolved == [(25, True), (25, True)]
    assert gradle_calls == [("9.5.1", "a" * 64, 360)]
    assert owner_calls == [(25, owner_jdk.resolve())]


def test_source_only_without_jdt_skips_toolchain_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        "minecraft_mod_ai.platform_catalog.adapter_for_lock_values",
        lambda _platform: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    complete_preflight_contract.validate_platform_toolchain_preflight(
        object(),
        SimpleNamespace(source_only=True, run_jdt=False),
    )


def test_bound_toolchain_preflight_runs_before_production_compilation() -> None:
    source = inspect.getsource(api.CompleteModAISession.build)
    assert source.index("validate_platform_toolchain_preflight") < source.index(
        "_production_proposal(self, proposal)"
    )
