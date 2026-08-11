from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_for_lock_values, adapter_from_project


def install(module: Any) -> None:
    validator = module.ProjectValidator
    if getattr(validator.validate, "_mmm_dynamic_platform_validation", False):
        return

    original_validate = validator.validate
    original_content = validator._validate_content

    @wraps(original_validate)
    def validate(self: Any, root: Path, spec: Any):
        root = root.expanduser().resolve()
        try:
            expected = adapter_for_lock_values(spec.platform)
            actual = adapter_from_project(root)
        except Exception as exc:
            return module.ValidationReport(
                status="FAIL",
                checks_run=1,
                findings=(
                    module.Finding(
                        "PLATFORM_LOCK_INVALID",
                        "error",
                        ".minecraft_ai/platform-lock.json",
                        f"Project target is missing, mixed or unsupported: {exc}",
                    ),
                ),
            )
        if actual.adapter_id != expected.adapter_id:
            return module.ValidationReport(
                status="FAIL",
                checks_run=1,
                findings=(
                    module.Finding(
                        "PLATFORM_LOCK_MISMATCH",
                        "error",
                        ".minecraft_ai/platform-lock.json",
                        (
                            f"Project uses {actual.adapter_id} but approved proposal uses "
                            f"{expected.adapter_id}."
                        ),
                    ),
                ),
            )
        report = original_validate(self, root, spec)
        extra = list(report.findings)
        checks = report.checks_run
        pack = root / "src/main/resources/pack.mcmeta"
        checks += 1
        payload = self._load_json(pack, extra, root)
        pack_data = payload.get("pack") if isinstance(payload, dict) else None
        if (
            not isinstance(pack_data, dict)
            or pack_data.get("pack_format") != expected.resource_pack_format
        ):
            extra.append(
                module.Finding(
                    "BAD_RESOURCE_PACK_FORMAT",
                    "error",
                    self._rel(root, pack),
                    (
                        f"pack_format must equal {expected.resource_pack_format} for "
                        f"Minecraft {expected.minecraft_version}."
                    ),
                )
            )
        return module.ValidationReport(
            status=(
                "PASS" if not any(item.severity == "error" for item in extra) else "FAIL"
            ),
            checks_run=checks,
            findings=tuple(extra),
        )

    @wraps(original_content)
    def validate_content(
        self: Any,
        root: Path,
        spec: Any,
        content: Any,
        en: dict[str, Any],
        ko: dict[str, Any],
        en_path: Path,
        ko_path: Path,
        findings: list[Any],
    ) -> int:
        adapter = adapter_for_lock_values(spec.platform)
        if adapter.source_api_family != "fabric_1211":
            return original_content(
                self,
                root,
                spec,
                content,
                en,
                ko,
                en_path,
                ko_path,
                findings,
            )

        checks = 0
        prefix = "item" if content.kind is module.ContentKind.ITEM else "block"
        translation_key = f"{prefix}.{spec.mod_id}.{content.content_id}"
        for locale, translations, path in (
            ("en_us", en, en_path),
            ("ko_kr", ko, ko_path),
        ):
            checks += 1
            if translation_key not in translations:
                findings.append(
                    module.Finding(
                        "MISSING_TRANSLATION",
                        "error",
                        self._rel(root, path),
                        f"{locale} is missing {translation_key}.",
                    )
                )

        assets = root / f"src/main/resources/assets/{spec.mod_id}"
        data = root / f"src/main/resources/data/{spec.mod_id}"
        if content.kind is module.ContentKind.ITEM:
            required = (
                assets / "textures/item" / f"{content.content_id}.png",
                assets / "models/item" / f"{content.content_id}.json",
            )
        else:
            required = (
                assets / "textures/block" / f"{content.content_id}.png",
                assets / "models/block" / f"{content.content_id}.json",
                assets / "models/item" / f"{content.content_id}.json",
                assets / "blockstates" / f"{content.content_id}.json",
                data / "loot_table/blocks" / f"{content.content_id}.json",
            )
        for path in required:
            checks += 1
            if not path.is_file() or path.is_symlink():
                findings.append(
                    module.Finding(
                        "MISSING_RESOURCE",
                        "error",
                        self._rel(root, path),
                        f"Required resource is missing for {content.content_id}.",
                    )
                )
            elif path.suffix == ".png":
                checks += 1
                self._validate_png(root, path, findings)

        if content.recipe:
            recipe_path = data / "recipe" / f"{content.content_id}.json"
            checks += 1
            recipe = self._load_json(recipe_path, findings, root)
            result = recipe.get("result", {}) if isinstance(recipe, dict) else {}
            if (
                not isinstance(result, dict)
                or result.get("id") != f"{spec.mod_id}:{content.content_id}"
            ):
                findings.append(
                    module.Finding(
                        "BAD_RECIPE_RESULT",
                        "error",
                        self._rel(root, recipe_path),
                        f"Recipe result does not target {content.content_id}.",
                    )
                )
        return checks

    validate._mmm_dynamic_platform_validation = True
    validate_content._mmm_dynamic_platform_validation = True
    validator.validate = validate
    validator._validate_content = validate_content
