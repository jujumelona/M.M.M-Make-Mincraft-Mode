from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def install(module: Any) -> None:
    """Install only the corrected boss validator behavior."""

    module.ProjectValidator._validate_boss = _validate_boss


def _validate_boss(
    self: Any,
    root: Path,
    spec: Any,
    en: dict[str, Any],
    ko: dict[str, Any],
    en_path: Path,
    ko_path: Path,
    findings: list[Any],
) -> int:
    from . import validator as module

    boss = spec.boss
    assert boss is not None
    checks = 0
    package_path = Path(*spec.package_name.split("."))
    main_class = module._class_name(spec.mod_id) + "Mod"
    entity_class = module._class_name(boss.entity_id) + "ModEntity"
    renderer_class = module._class_name(boss.entity_id) + "ModRenderer"
    required = (
        root
        / f"src/main/resources/assets/{spec.mod_id}/textures/entity/{boss.entity_id}.png",
        root
        / f"src/main/resources/assets/{spec.mod_id}/models/item/{boss.entity_id}_spawn_egg.json",
        root
        / f"src/main/resources/data/{spec.mod_id}/loot_tables/entities/{boss.entity_id}.json",
        root / "src/main/java" / package_path / "entity" / f"{entity_class}.java",
        root / "src/main/java" / package_path / "client" / f"{renderer_class}.java",
        root / "src/main/java" / package_path / "client" / f"{main_class}Client.java",
        root / f".minecraft_ai/art_sources/{boss.entity_id}.bbmodel",
        root / f".minecraft_ai/art_sources/{boss.entity_id}.obj",
        root / f".minecraft_ai/art_sources/{boss.entity_id}.mtl",
    )
    for path in required:
        checks += 1
        if not path.is_file() or path.is_symlink():
            findings.append(
                module.Finding(
                    "MISSING_BOSS_ASSET",
                    "error",
                    self._rel(root, path),
                    f"Boss asset is missing for {boss.entity_id}.",
                )
            )
    checks += 1
    self._validate_png(root, required[0], findings)
    bbmodel = self._load_json(required[6], findings, root)
    checks += 1
    if bbmodel.get("model_identifier") != f"{spec.mod_id}:{boss.entity_id}":
        findings.append(
            module.Finding(
                "BAD_BBMODEL",
                "error",
                self._rel(root, required[6]),
                "Blockbench model identifier does not match the boss.",
            )
        )
    obj_text = (
        required[7].read_text(encoding="utf-8", errors="replace")
        if required[7].is_file()
        else ""
    )
    checks += 2
    if len(re.findall(r"^v\s", obj_text, re.MULTILINE)) < 8:
        findings.append(
            module.Finding(
                "BAD_OBJ",
                "error",
                self._rel(root, required[7]),
                "Boss OBJ has too few vertices.",
            )
        )
    if len(re.findall(r"^f\s", obj_text, re.MULTILINE)) < 6:
        findings.append(
            module.Finding(
                "BAD_OBJ",
                "error",
                self._rel(root, required[7]),
                "Boss OBJ has too few faces.",
            )
        )
    key = f"entity.{spec.mod_id}.{boss.entity_id}"
    for locale, translations, path in (
        ("en_us", en, en_path),
        ("ko_kr", ko, ko_path),
    ):
        checks += 1
        if key not in translations:
            findings.append(
                module.Finding(
                    "MISSING_BOSS_TRANSLATION",
                    "error",
                    self._rel(root, path),
                    f"{locale} is missing {key}.",
                )
            )
    return checks
