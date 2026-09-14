from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.java_diagnostics_fallback_contract as contract


def test_install_does_not_wrap_java_diagnostics(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...], int]] = []

    class FakeService:
        workspace_root = str(tmp_path)

        def java_diagnostics(
            self,
            project_root: str,
            relative_files: list[str] | None = None,
            timeout_seconds: int = 60,
        ) -> dict[str, object]:
            calls.append((project_root, tuple(relative_files or ()), timeout_seconds))
            return {
                "status": "UNAVAILABLE",
                "diagnostics": {},
                "error": "JDT diagnostics unavailable",
            }

    original = FakeService.java_diagnostics
    contract.install(FakeService)

    assert FakeService.java_diagnostics is original
    result = FakeService().java_diagnostics(
        str(tmp_path),
        relative_files=["src/main/java/example/Test.java"],
        timeout_seconds=7,
    )
    assert result["status"] == "UNAVAILABLE"
    assert calls == [
        (
            str(tmp_path),
            ("src/main/java/example/Test.java",),
            7,
        )
    ]
