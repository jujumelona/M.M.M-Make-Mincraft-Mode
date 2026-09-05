from __future__ import annotations

from pathlib import Path


_OWNER = "target_profile_semantics.py"
_FORBIDDEN_POLICY_MARKERS = (
    "_NATIVE_NAME_MIN_VERSION",
    "def _uses_native_names",
    ">= (26, 1)",
    ">= (26,1)",
)


def test_target_version_policy_is_not_duplicated_across_production_modules() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    violations: list[str] = []
    for path in sorted(package_root.glob("*.py")):
        if path.name == _OWNER:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_POLICY_MARKERS:
            if marker in text:
                violations.append(f"{path.name}: {marker}")

    assert not violations, (
        "Minecraft target-version policy must route through target_profile_semantics; "
        + "; ".join(violations)
    )
