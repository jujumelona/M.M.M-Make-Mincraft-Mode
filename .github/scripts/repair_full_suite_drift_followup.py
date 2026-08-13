from __future__ import annotations

import re
from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"missing expected text in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_function(path: str, name: str, source: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    pattern = re.compile(rf"(?ms)^def {re.escape(name)}\(.*?(?=^def |\Z)")
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"function {name} not found in {path}")
    target.write_text(
        text[: match.start()] + source.rstrip() + "\n\n" + text[match.end() :],
        encoding="utf-8",
    )


# Cell 2 now uses exact origin/main checkout/reset rather than a merge. This is the
# mechanism that prevents a partially stale Colab checkout from surviving setup.
replace_exact(
    "tests/test_notebook_registry_policy.py",
    '''    assert '"merge",' in cells["setup"]
    assert '"pull",' not in cells["setup"]
    assert "refs/remotes/origin/main" in cells["setup"]
    assert '"--untracked-files=no"' in cells["setup"]
''',
    '''    assert '"checkout",' in cells["setup"]
    assert '"reset",' in cells["setup"]
    assert '"--hard",' in cells["setup"]
    assert '"clean",' in cells["setup"]
    assert '"-fd"' in cells["setup"]
    assert '"pull",' not in cells["setup"]
    assert "refs/remotes/origin/main" in cells["setup"]
    assert '"--untracked-files=no"' in cells["setup"]
    assert "sys.modules.pop" in cells["setup"]
    assert "importlib.invalidate_caches()" in cells["setup"]
    assert "LATEST_MAIN_COMMIT" in cells["existing-input"]
    assert "CURRENT_ENGINE_COMMIT" in cells["existing-input"]
    assert '"reset",' in cells["existing-input"]
    assert '"--hard",' in cells["existing-input"]
    assert "sys.modules.pop" in cells["existing-input"]
    assert "importlib.invalidate_caches()" in cells["existing-input"]
''',
)

# Technology targets are dynamic. Validate exact binding to a fully resolved explicit
# target rather than asserting that every version other than the historical default is
# rejected.
replace_function(
    "tests/test_technology_radar.py",
    "test_target_and_authority_contract_are_exact_for_every_requirement",
    '''def test_target_and_authority_contract_are_exact_for_every_requirement() -> None:
    adapter = adapter_for_target("1.20.1", "fabric")
    target = {
        "edition": adapter.edition,
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
    }
    radar = build_technology_radar(
        "Add speech recognition to NPC dialogue.",
        target=target,
    )

    for requirement in radar["requirements"]:
        assert requirement["target"] == target
        assert requirement["authority"]["game_state_mutation"] == "server_only"
        assert (
            requirement["authority"]["client_messages"]
            == "schema_validated_and_rate_limited_by_server"
        )

    newer = adapter_for_target("1.21.1", "fabric")
    newer_target = {
        "edition": newer.edition,
        "minecraft_version": newer.minecraft_version,
        "loader": newer.loader,
        "mappings": newer.yarn_mappings,
        "java_version": newer.java_version,
        "fabric_loader": newer.fabric_loader,
        "fabric_api": newer.fabric_api,
    }
    newer_radar = build_technology_radar("AI NPC", target=newer_target)
    assert newer_radar["requirements"]
    assert all(
        requirement["target"] == newer_target
        for requirement in newer_radar["requirements"]
    )''',
)
