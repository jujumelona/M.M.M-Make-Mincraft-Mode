from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "templates"
PACKAGED = ROOT / "minecraft_mod_ai" / "templates"
MIRRORED_GROUPS = ("asset", "design", "translation")


def _yaml_files(base: Path, group: str) -> set[Path]:
    group_root = base / group
    if not group_root.exists():
        return set()
    return {path.relative_to(group_root) for path in group_root.rglob("*.yaml")}


def test_packaged_template_mirrors_match_canonical_templates() -> None:
    """Prevent runtime/package templates from silently drifting from canonical templates."""
    problems: list[str] = []

    for group in MIRRORED_GROUPS:
        canonical_files = _yaml_files(CANONICAL, group)
        packaged_files = _yaml_files(PACKAGED, group)

        missing = canonical_files - packaged_files
        if missing:
            problems.extend(
                f"missing packaged mirror: {group}/{path.as_posix()}"
                for path in sorted(missing)
            )

        for relative_path in sorted(canonical_files & packaged_files):
            canonical_path = CANONICAL / group / relative_path
            packaged_path = PACKAGED / group / relative_path
            if canonical_path.read_bytes() != packaged_path.read_bytes():
                problems.append(f"template drift: {group}/{relative_path.as_posix()}")

    assert not problems, "Template mirror integrity failed:\n" + "\n".join(problems)
