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

        if expected.source_api_family == "fabric_live_ai":
            return _validate_live_project(self, module, root, spec, expected)

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


def _validate_live_project(
    self: Any,
    module: Any,
    root: Path,
    spec: Any,
    adapter: Any,
):
    """Version-neutral source gate for officially bootstrapped future targets.

    Historical MMM validators encode exact 1.20/1.21 resource directories, recipe
    result schemas, pack-format integers and generated class names. None of those is
    a safe invariant for an unseen release. Live targets therefore validate only
    target identity, filesystem/security invariants and official project metadata at
    this stage. JDT, Gradle, GameTest, JAR and runtime gates remain mandatory later in
    the complete pipeline and are the authority for target-specific API correctness.
    """

    findings: list[Any] = []
    checks = 0
    if not root.is_dir() or root.is_symlink():
        return module.ValidationReport(
            status="FAIL",
            checks_run=1,
            findings=(
                module.Finding(
                    "PROJECT_ROOT_INVALID",
                    "error",
                    str(root),
                    "Project root must be a regular directory.",
                ),
            ),
        )

    for path in sorted(root.rglob("*")):
        checks += 1
        relative = self._rel(root, path)
        if path.is_symlink():
            findings.append(
                module.Finding(
                    "SYMLINK",
                    "error",
                    relative,
                    "Symlinks are not allowed.",
                )
            )
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError:
            findings.append(
                module.Finding(
                    "PATH_ESCAPE",
                    "error",
                    relative,
                    "Path escaped the staging root.",
                )
            )
            continue
        if path.is_file() and path.stat().st_size > self.policy.max_single_file_bytes:
            findings.append(
                module.Finding(
                    "FILE_TOO_LARGE",
                    "error",
                    relative,
                    "File exceeds MMM_MAX_SINGLE_FILE_BYTES host resource policy.",
                )
            )

    for path in sorted({*root.rglob("*.json"), *root.rglob("*.mcmeta")}):
        checks += 1
        self._load_json(path, findings, root)

    java_files = sorted(root.rglob("*.java"))
    checks += 1
    if not java_files:
        findings.append(
            module.Finding(
                "NO_JAVA",
                "error",
                ".",
                "No Java sources were generated.",
            )
        )
    for path in java_files:
        checks += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = self._rel(root, path)
        for token in self.FORBIDDEN_JAVA:
            if token in text:
                findings.append(
                    module.Finding(
                        "FORBIDDEN_JAVA_API",
                        "error",
                        relative,
                        f"Generated source contains forbidden API token {token!r}.",
                    )
                )

    properties = _read_properties(root / "gradle.properties")
    expected_properties = {
        "minecraft_version": adapter.minecraft_version,
        "loader_version": adapter.fabric_loader,
        "fabric_api_version": adapter.fabric_api,
        "loom_version": adapter.fabric_loom,
    }
    for key, expected_value in expected_properties.items():
        checks += 1
        actual_value = properties.get(key)
        # Fabric templates have used both fabric_version and fabric_api_version.
        if key == "fabric_api_version" and actual_value is None:
            actual_value = properties.get("fabric_version")
        if actual_value != expected_value:
            findings.append(
                module.Finding(
                    "LIVE_TOOLCHAIN_MISMATCH",
                    "error",
                    "gradle.properties",
                    f"{key} must match the approved official target {expected_value!r}.",
                )
            )

    fabric_path = root / "src/main/resources/fabric.mod.json"
    fabric = self._load_json(fabric_path, findings, root)
    checks += 1
    if isinstance(fabric, dict):
        for key, expected_value in (
            ("id", spec.mod_id),
            ("version", "${version}"),
            ("environment", "*"),
        ):
            checks += 1
            if fabric.get(key) != expected_value:
                findings.append(
                    module.Finding(
                        "BAD_FABRIC_METADATA",
                        "error",
                        self._rel(root, fabric_path),
                        f"{key} must equal {expected_value!r}.",
                    )
                )
        entrypoints = fabric.get("entrypoints")
        checks += 1
        if not isinstance(entrypoints, dict) or not _nonempty_entrypoint(entrypoints.get("main")):
            findings.append(
                module.Finding(
                    "BAD_ENTRYPOINTS",
                    "error",
                    self._rel(root, fabric_path),
                    "Live Fabric project must expose at least one main entrypoint.",
                )
            )
        depends = fabric.get("depends")
        checks += 1
        if not isinstance(depends, dict):
            findings.append(
                module.Finding(
                    "BAD_FABRIC_DEPENDS",
                    "error",
                    self._rel(root, fabric_path),
                    "depends must be an object.",
                )
            )
        else:
            required_ids = {"fabricloader", "minecraft", "java"}
            if not ({"fabric-api", "fabric"} & set(depends)):
                required_ids.add("fabric-api")
            for dependency_id in sorted(required_ids):
                checks += 1
                if dependency_id == "fabric-api" and (
                    "fabric-api" in depends or "fabric" in depends
                ):
                    continue
                value = depends.get(dependency_id)
                if not isinstance(value, str) or not value.strip():
                    findings.append(
                        module.Finding(
                            "BAD_FABRIC_DEPENDS",
                            "error",
                            self._rel(root, fabric_path),
                            f"Missing non-empty dependency predicate for {dependency_id}.",
                        )
                    )

    # pack.mcmeta is optional in an official Fabric project and its exact schema is
    # version-owned. If it exists, JSON validity was already checked above. Never
    # guess a future pack_format integer in MMM source code.

    status = "PASS" if not any(item.severity == "error" for item in findings) else "FAIL"
    return module.ValidationReport(
        status=status,
        checks_run=checks,
        findings=tuple(findings),
    )


def _read_properties(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _nonempty_entrypoint(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty_entrypoint(item) for item in value)
    if isinstance(value, dict):
        return any(_nonempty_entrypoint(item) for item in value.values())
    return False
