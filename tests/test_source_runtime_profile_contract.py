from __future__ import annotations

import inspect
import json

from minecraft_mod_ai.complete_orchestrator_services import runtime_profile
from minecraft_mod_ai.runtime_manager import MinecraftRuntimeManager


def test_source_runtime_profile_matches_manager_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MMM_MINECRAFT_VERSION", "1.21.8")
    monkeypatch.setenv("MMM_LOADER", "fabric")
    monkeypatch.setenv("MMM_JAVA_VERSION", "21")

    profile_path = runtime_profile(tmp_path, 2048)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))

    assert set(payload["profiles"]) == {"fabric_target_disposable"}
    assert (
        inspect.signature(MinecraftRuntimeManager.__init__)
        .parameters["profile_name"]
        .default
        == "fabric_target_disposable"
    )
