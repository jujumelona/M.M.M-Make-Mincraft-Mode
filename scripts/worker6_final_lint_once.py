from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def isolate_proof_test_from_scaffold(path: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if "def _worker6_isolate_scaffold(" in text:
        return
    anchor = "import minecraft_mod_ai.reuse_proof_executor as reuse_proof\n"
    if text.count(anchor) != 1:
        raise SystemExit(f"{path}: reuse_proof import anchor missing or ambiguous")
    fixture = (
        anchor
        + "\n\n@pytest.fixture(autouse=True)\n"
        + "def _worker6_isolate_scaffold(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        + "    monkeypatch.setattr(\n"
        + "        reuse_proof,\n"
        + "        \"scaffold_minimal_ephemeral_workspace\",\n"
        + "        lambda *args, **kwargs: None,\n"
        + "    )\n"
    )
    target.write_text(text.replace(anchor, fixture, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/dependency_resolver.py",
    '''    payload = "\\n".join(\n        (\n            repository,\n            coordinate,\n            configuration,\n            target_loader.casefold(),\n            target_minecraft,\n        )\n    )\n''',
    '''    payload = (\n        f"{repository}\\n{coordinate}\\n{configuration}\\n"\n        f"{target_loader.casefold()}\\n{target_minecraft}"\n    )\n''',
)

replace_once(
    "minecraft_mod_ai/reuse_artifacts.py",
    '''        for p in protected_paths:\n            if p.startswith("src/main/resources/assets/"):\n                parts = p.split("/")\n                if len(parts) > 4 and parts[4] not in ("minecraft", "c", "fabric", "neoforge", "forge"):\n                    if parts[4] not in owned_ns:\n                        owned_ns.append(parts[4])\n            elif p.startswith("src/main/resources/data/"):\n                parts = p.split("/")\n                if len(parts) > 4 and parts[4] not in ("minecraft", "c", "fabric", "neoforge", "forge"):\n                    if parts[4] not in owned_ns:\n                        owned_ns.append(parts[4])\n''',
    '''        for path in protected_paths:\n            if not path.startswith((\n                "src/main/resources/assets/",\n                "src/main/resources/data/",\n            )):\n                continue\n            parts = path.split("/")\n            if (\n                len(parts) > 4\n                and parts[4] not in ("minecraft", "c", "fabric", "neoforge", "forge")\n                and parts[4] not in owned_ns\n            ):\n                owned_ns.append(parts[4])\n''',
)

replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    '''        if path.endswith(".java") or path.endswith(".kt")\n''',
    '''        if path.endswith((".java", ".kt"))\n''',
)

for test_file in (
    "tests/test_worker6_reuse_fail_closed.py",
    "tests/test_worker6_reuse_hardening.py",
    "tests/test_worker6_reuse_authority.py",
):
    isolate_proof_test_from_scaffold(test_file)
